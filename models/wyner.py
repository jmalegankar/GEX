import torch as th
import torch.nn as nn

from typing import Optional, List, Tuple

import math

@th.jit.script
def sinusoidal_timestep_encoding(timesteps: th.Tensor, embed_dim: int) -> th.Tensor:
    """Convert integer timesteps (B,) or (B,1) to sinusoidal positional encoding (B, embed_dim)."""
    timesteps = timesteps.view(-1)  # ensure (B,)
    half = embed_dim // 2
    freqs = th.exp(-math.log(10000.0) * th.arange(half, dtype=th.float32, device=timesteps.device) / half)
    angles = timesteps.unsqueeze(1).float() * freqs.unsqueeze(0)  # (B, half)
    pe = th.cat([th.sin(angles), th.cos(angles)], dim=-1)  # (B, 2*half)
    if embed_dim % 2 == 1:
        pe = th.cat([pe, th.zeros(pe.size(0), 1, device=pe.device)], dim=-1)
    return pe


class WynerPriorNetwork(nn.Module):
    """Learned prior p_theta(z | w_{t-1}). Takes previous Wyner latent, outputs Gaussian params."""
    def __init__(self, latent_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        nn.init.zeros_(self.fc_logvar.bias)

    def forward(self, w_prev: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """w_prev: (B, 1, latent_dim) or (B, latent_dim). Returns (prior_mu, prior_logvar)."""
        x = w_prev.squeeze(1) if w_prev.dim() == 3 else w_prev
        h = self.net(x)
        return self.fc_mean(h), self.fc_logvar(h)


@th.jit.script
class WynerOutput:
    def __init__(
        self,
        w: th.Tensor,
        logvar: th.Tensor,
        recon: th.Tensor,
        recon_next: Optional[th.Tensor] = None,
        prior_mu: Optional[th.Tensor] = None,
        prior_logvar: Optional[th.Tensor] = None,
        posterior_mu: Optional[th.Tensor] = None,  # LBS: separate from w when w = h_t
    ):
        self.recon = recon
        self.recon_next = recon_next
        self.w = w
        self.logvar = logvar
        self.prior_mu = prior_mu
        self.prior_logvar = prior_logvar
        self.posterior_mu = posterior_mu


@th.jit.script
class WynerLoss:
    def __init__(
        self,
        kl_loss: th.Tensor,
        recon_loss: Optional[th.Tensor] = None,
        recon_next_loss: Optional[th.Tensor] = None,
    ):
        self.recon_loss = recon_loss
        self.recon_next_loss = recon_next_loss
        self.kl_loss = kl_loss

@th.jit.interface
class WynerInterface:
    def encode(self, w: th.Tensor, mu: th.Tensor, skips: Optional[List[th.Tensor]], timestep: Optional[th.Tensor] = None) -> Tuple[th.Tensor, th.Tensor]:
        pass

    def decode(self, z: th.Tensor, mu: th.Tensor, timestep: Optional[th.Tensor] = None) -> th.Tensor:
        pass

    def forward(self, w: th.Tensor, mu: th.Tensor, mu_next: Optional[th.Tensor], skips: Optional[List[th.Tensor]], timestep: Optional[th.Tensor] = None) -> WynerOutput:
        pass

    def loss(self, output: WynerOutput, recon_target: Optional[th.Tensor], recon_next_target: Optional[th.Tensor]) -> WynerLoss:
        pass


class WynerDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        latent_tokens: int,
        mu_dim: int,
        recon_dim: int,
        decode_hidden: int = 128,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.latent_tokens = latent_tokens
        self.mu_dim = mu_dim
        self.recon_dim = recon_dim
        self.decode_hidden = decode_hidden

        self.proj = nn.Linear(self.latent_dim, self.decode_hidden)
        
        self.fc1 = nn.Linear(self.decode_hidden * self.latent_tokens + self.mu_dim, self.decode_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.decode_hidden, self.decode_hidden)
        self.fc_out = nn.Linear(self.decode_hidden, self.recon_dim)
    
    def forward(self, z: th.Tensor, mu: th.Tensor) -> th.Tensor:
        z_proj = self.proj(z)
        z_proj = z_proj.view(z_proj.size(0), self.latent_tokens * self.decode_hidden)
        x = th.cat([z_proj, mu], dim=-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        recon = self.fc_out(x)
        return recon


class WynerVAE(nn.Module):
    def __init__(
        self,
        recon_dim: int,
        mu_dim: int,
        latent_dim: int,
        latent_tokens: int = 1,
        decode_hidden: int = 128,
        state_dim: int = 0,
        state_tokens: int = 0,
        free_bits: float = 0.0,
    ):
        super().__init__()
        assert latent_tokens >= 1, "Latent tokens must be at least 1."
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.state_tokens = state_tokens
        self.latent_tokens = latent_tokens

        self.mu_dim = mu_dim
        self.recon_dim = recon_dim
        self.free_bits = free_bits

        if self.state_tokens == 0 and self.latent_tokens == 1:
            self.gru = nn.GRUCell(input_size=mu_dim, hidden_size=latent_dim)
        else:
            raise NotImplementedError("Only single token encoding/decoding is implemented for now.")

        self.fc_mean = nn.Linear(self.latent_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.latent_dim, self.latent_dim)
        nn.init.zeros_(self.fc_logvar.bias)

        self.decoder = WynerDecoder(
            latent_dim=self.latent_dim,
            latent_tokens=self.latent_tokens,
            mu_dim=self.mu_dim,
            recon_dim=self.recon_dim,
            decode_hidden=decode_hidden,
        )
    
    def encode(self, w: th.Tensor, mu: th.Tensor, skips: Optional[List[th.Tensor]] = None, timestep: Optional[th.Tensor] = None) -> Tuple[th.Tensor, th.Tensor]:
        h = self.gru(mu, w.squeeze(1))
        z_mu = self.fc_mean(h)
        z_logvar = self.fc_logvar(h)
        return z_mu, z_logvar

    def decode(self, z: th.Tensor, mu: th.Tensor, timestep: Optional[th.Tensor] = None) -> th.Tensor:
        return self.decoder(z, mu)

    def forward(self, w: th.Tensor, mu: th.Tensor, mu_next: Optional[th.Tensor] = None, skips: Optional[List[th.Tensor]] = None, timestep: Optional[th.Tensor] = None) -> WynerOutput:
        z_mu, z_logvar = self.encode(w, mu, skips)
        std = th.exp(0.5 * z_logvar)
        eps = th.randn_like(std)
        z = z_mu + eps * std
        recon = self.decode(z, mu)
        recon_next = self.decode(z, mu_next) if mu_next is not None else None
        return WynerOutput(w=z_mu, logvar=z_logvar, recon=recon, recon_next=recon_next)

    def loss(self, output: WynerOutput, recon_target: Optional[th.Tensor] = None, recon_next_target: Optional[th.Tensor] = None) -> WynerLoss:
        kl_per_dim = -0.5 * (1 + output.logvar - output.w.pow(2) - output.logvar.exp())
        kl_loss = kl_per_dim.clamp_min(self.free_bits).sum(dim=-1)

        if recon_target is not None:
            recon_loss = nn.functional.mse_loss(output.recon, recon_target, reduction='none').mean(dim=-1)
        else:
            recon_loss = None

        if recon_next_target is not None and output.recon_next is not None:
            recon_next_loss = nn.functional.mse_loss(output.recon_next, recon_next_target, reduction='none').mean(dim=-1)
        else:
            recon_next_loss = None

        return WynerLoss(kl_loss=kl_loss, recon_loss=recon_loss, recon_next_loss=recon_next_loss)

class WynerIndependentDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        latent_tokens: int,
        recon_dim: int,
        decode_hidden: int = 128,
        pos_embed_dim: int = 16,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.latent_tokens = latent_tokens
        self.recon_dim = recon_dim
        self.decode_hidden = decode_hidden
        self.pos_embed_dim = pos_embed_dim

        self.proj = nn.Linear(self.latent_dim, self.decode_hidden)

        self.fc1 = nn.Linear(self.decode_hidden * self.latent_tokens + self.pos_embed_dim, self.decode_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.decode_hidden, self.decode_hidden)
        self.fc_out = nn.Linear(self.decode_hidden, self.recon_dim)

    def forward(self, z: th.Tensor, timestep_encoding: th.Tensor) -> th.Tensor:
        z_proj = self.proj(z)
        z_proj = z_proj.view(z_proj.size(0), self.latent_tokens * self.decode_hidden)
        x = th.cat([z_proj, timestep_encoding], dim=-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        recon = self.fc_out(x)
        return recon

class WynerIndependentVAE(nn.Module):
    def __init__(
        self,
        recon_dim: int,
        mu_dim: int,
        latent_dim: int,
        latent_tokens: int = 1,
        decode_hidden: int = 128,
        state_dim: int = 0,
        state_tokens: int = 0,
        free_bits: float = 0.0,
        pos_embed_dim: int = 16,
    ):
        super().__init__()
        assert latent_tokens >= 1, "Latent tokens must be at least 1."
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.state_tokens = state_tokens
        self.latent_tokens = latent_tokens

        self.mu_dim = mu_dim
        self.recon_dim = recon_dim
        self.free_bits = free_bits
        self.pos_embed_dim = pos_embed_dim

        # joint encoder: separate projections summed -> GRU input
        self.proj_t = nn.Linear(mu_dim, mu_dim)

        if self.state_tokens == 0 and self.latent_tokens == 1:
            self.gru = nn.GRUCell(input_size=mu_dim + pos_embed_dim, hidden_size=latent_dim)
        else:
            raise NotImplementedError("Only single token encoding/decoding is implemented for now.")

        self.fc_mean = nn.Linear(self.latent_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.latent_dim, self.latent_dim)
        nn.init.zeros_(self.fc_logvar.bias)

        # Learned prior: p_theta(z | w_{t-1})
        self.prior_net = WynerPriorNetwork(latent_dim, hidden_dim=decode_hidden)

        self.decoder = WynerIndependentDecoder(
            latent_dim=self.latent_dim,
            latent_tokens=self.latent_tokens,
            recon_dim=self.recon_dim,
            decode_hidden=decode_hidden,
            pos_embed_dim=pos_embed_dim,
        )

    def _gru_step(
        self, w: th.Tensor, mu: th.Tensor, timestep: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        # project mu and optionally mu_next, sum, and run GRU.
        gru_input = self.proj_t(mu)
        # concat positional embedding of timestep
        if timestep is not None:
            pos_emb = sinusoidal_timestep_encoding(timestep, self.pos_embed_dim)
        else:
            pos_emb = th.zeros(gru_input.size(0), self.pos_embed_dim, device=gru_input.device)
        gru_input = th.cat([gru_input, pos_emb], dim=-1)
        h = self.gru(gru_input, w.squeeze(1))
        return h

    def _timestep_encoding(self, timestep: Optional[th.Tensor], batch_size: int, device: th.device) -> th.Tensor:
        if timestep is not None:
            return sinusoidal_timestep_encoding(timestep, self.pos_embed_dim)
        return th.zeros(batch_size, self.pos_embed_dim, device=device)

    def encode(self, w: th.Tensor, mu: th.Tensor, skips: Optional[List[th.Tensor]] = None, timestep: Optional[th.Tensor] = None) -> Tuple[th.Tensor, th.Tensor]:
        h = self._gru_step(w, mu, timestep=timestep)
        z_mu = self.fc_mean(h)
        z_logvar = self.fc_logvar(h)
        return z_mu, z_logvar

    def decode(self, z: th.Tensor, mu: th.Tensor, timestep: Optional[th.Tensor] = None) -> th.Tensor:
        ts_enc = self._timestep_encoding(timestep, z.size(0), z.device)
        recon = self.decoder(z, ts_enc)
        return recon

    def forward(self, w: th.Tensor, mu: th.Tensor, mu_next: Optional[th.Tensor] = None, skips: Optional[List[th.Tensor]] = None, timestep: Optional[th.Tensor] = None) -> WynerOutput:
        # Compute learned prior from w_{t-1} BEFORE GRU update
        prior_mu, prior_logvar = self.prior_net(w)

        z_mu, z_logvar = self.encode(w, mu, skips, timestep=timestep)
        std = th.exp(0.5 * z_logvar)
        eps = th.randn_like(std)
        z = z_mu + eps * std

        ts_enc = self._timestep_encoding(timestep, z.size(0), z.device)
        recon = self.decoder(z, ts_enc)

        if mu_next is not None:
            # decoder_tp1 gets timestep + 1 encoding
            if timestep is not None:
                ts_enc_next = sinusoidal_timestep_encoding(timestep + 1, self.pos_embed_dim)
            else:
                ts_enc_next = th.zeros(z.size(0), self.pos_embed_dim, device=z.device)
            recon_next = self.decoder(z, ts_enc_next)
        else:
            recon_next = None

        return WynerOutput(w=z_mu, logvar=z_logvar, recon=recon, recon_next=recon_next,
                           prior_mu=prior_mu, prior_logvar=prior_logvar)

    def loss(self, output: WynerOutput, recon_target: Optional[th.Tensor] = None, recon_next_target: Optional[th.Tensor] = None) -> WynerLoss:
        # KL(q(z|...) || p_theta(z|w_{t-1})) — learned prior
        if output.prior_mu is not None and output.prior_logvar is not None:
            kl_per_dim = 0.5 * (
                output.prior_logvar - output.logvar
                + (output.logvar.exp() + (output.w - output.prior_mu).pow(2)) / output.prior_logvar.exp()
                - 1.0
            )
        else:
            # Fallback: KL against N(0, I)
            kl_per_dim = -0.5 * (1 + output.logvar - output.w.pow(2) - output.logvar.exp())
        kl_loss = kl_per_dim.clamp_min(self.free_bits).mean(dim=-1)

        if recon_target is not None:
            recon_loss = nn.functional.mse_loss(output.recon, recon_target, reduction='none').mean(dim=-1)
        else:
            recon_loss = None

        if recon_next_target is not None and output.recon_next is not None:
            recon_next_loss = nn.functional.mse_loss(output.recon_next, recon_next_target, reduction='none').mean(dim=-1)
        else:
            recon_next_loss = None

        return WynerLoss(kl_loss=kl_loss, recon_loss=recon_loss, recon_next_loss=recon_next_loss)
    
class WynerLBSVAE(nn.Module):
    """
    Wyner VAE with LBS-inspired separation of deterministic and stochastic paths.

    The GRU maintains a deterministic hidden state h_t that summarises episode history.
    The prior predicts from h_{t-1} (before seeing the current observation).
    The posterior updates from h_t (after the GRU incorporates μ_t), optionally
    conditioning on μ_{t+1} for the Wyner bidirectional constraint.

    Intrinsic reward = KL(posterior ‖ prior) = Bayesian surprise in latent space.
    """

    def __init__(
        self,
        recon_dim: int,
        mu_dim: int,
        latent_dim: int,
        latent_tokens: int = 1,
        decode_hidden: int = 128,
        state_dim: int = 0,      # unused, kept for interface compat
        state_tokens: int = 0,   # unused, kept for interface compat
        free_bits: float = 0.0,
        pos_embed_dim: int = 16,
    ):
        super().__init__()
        assert latent_tokens == 1, "Only single-token implemented."

        self.mu_dim = mu_dim
        self.latent_dim = latent_dim
        self.latent_tokens = latent_tokens
        self.recon_dim = recon_dim
        self.free_bits = free_bits
        self.pos_embed_dim = pos_embed_dim

        # ── Deterministic recurrence ──────────────────────────────
        # h_t = GRU(proj(μ_t) ‖ pos(t), h_{t-1})
        self.proj_mu_gru = nn.Linear(mu_dim, mu_dim)
        self.gru = nn.GRUCell(
            input_size=mu_dim + pos_embed_dim,
            hidden_size=latent_dim,          # h_t has same dim as z for simplicity
        )

        # ── Prior p(z_t | h_{t-1}) ────────────────────────────────
        # Predicts BEFORE seeing the current observation.
        self.prior_trunk = nn.Sequential(
            nn.Linear(latent_dim, decode_hidden),
            nn.ReLU(),
            nn.Linear(decode_hidden, decode_hidden),
            nn.ReLU(),
        )
        self.prior_fc_mu = nn.Linear(decode_hidden, latent_dim)
        self.prior_fc_logvar = nn.Linear(decode_hidden, latent_dim)
        nn.init.zeros_(self.prior_fc_logvar.bias)

        # ── Posterior q(z_t | h_t [, μ_{t+1}]) ───────────────────
        # Two projections summed: h_t always present, μ_{t+1} optional.
        # When μ_{t+1} is absent (rollout), only the h branch fires.
        self.post_proj_h = nn.Linear(latent_dim, decode_hidden)
        self.post_proj_mu_next = nn.Linear(mu_dim, decode_hidden)
        self.post_trunk = nn.Sequential(
            nn.ReLU(),
            nn.Linear(decode_hidden, decode_hidden),
            nn.ReLU(),
        )
        self.post_fc_mu = nn.Linear(decode_hidden, latent_dim)
        self.post_fc_logvar = nn.Linear(decode_hidden, latent_dim)
        nn.init.zeros_(self.post_fc_logvar.bias)

        # ── Decoder: z + timestep → reconstruction ────────────────
        self.decoder = WynerIndependentDecoder(
            latent_dim=latent_dim,
            latent_tokens=latent_tokens,
            recon_dim=recon_dim,
            decode_hidden=decode_hidden,
            pos_embed_dim=pos_embed_dim,
        )

    # ──────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────

    def _timestep_encoding(self, timestep: Optional[th.Tensor], batch_size: int, device: th.device) -> th.Tensor:
        if timestep is not None:
            return sinusoidal_timestep_encoding(timestep, self.pos_embed_dim)
        return th.zeros(batch_size, self.pos_embed_dim, device=device)

    def _gru_step(self, h_prev: th.Tensor, mu: th.Tensor, timestep: Optional[th.Tensor]) -> th.Tensor:
        """Run one GRU step: h_t = GRU(proj(μ_t) ‖ pos(t), h_{t-1})."""
        gru_input = self.proj_mu_gru(mu)
        pos_emb = self._timestep_encoding(timestep, mu.size(0), mu.device)
        gru_input = th.cat([gru_input, pos_emb], dim=-1)
        # h_prev may be (B, 1, D) from buffer — squeeze to (B, D)
        h = h_prev.squeeze(1) if h_prev.dim() == 3 else h_prev
        return self.gru(gru_input, h)

    def _prior(self, h_prev: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """Prior p(z | h_{t-1}): predict BEFORE seeing current obs."""
        h = h_prev.squeeze(1) if h_prev.dim() == 3 else h_prev
        feat = self.prior_trunk(h)
        return self.prior_fc_mu(feat), self.prior_fc_logvar(feat)

    def _posterior(self, h_t: th.Tensor, mu_next: Optional[th.Tensor] = None) -> Tuple[th.Tensor, th.Tensor]:
        """
        Posterior q(z | h_t [, μ_{t+1}]).

        During training (mu_next given):
            Wyner posterior — sees both the GRU-updated state and the
            next transition encoding, satisfying the bidirectional condition.

        During rollout (mu_next=None):
            LBS-style posterior — h_t already incorporates μ_t via GRU,
            so the KL against the prior still measures genuine surprise.
        """
        feat = self.post_proj_h(h_t)
        if mu_next is not None:
            feat = feat + self.post_proj_mu_next(mu_next)
        feat = self.post_trunk(feat)
        return self.post_fc_mu(feat), self.post_fc_logvar(feat)

    # ──────────────────────────────────────────────────────────────
    # Interface methods
    # ──────────────────────────────────────────────────────────────

    def encode(
        self,
        h_prev: th.Tensor,
        mu: th.Tensor,
        skips: Optional[List[th.Tensor]] = None,
        timestep: Optional[th.Tensor] = None,
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        Deterministic path only — used by policy for feature extraction.
        Returns (h_t, dummy) where h_t is the new GRU hidden state.

        This is all the policy needs: h_t goes through attention with μ
        to produce actor/critic features. No stochastic sampling here.
        """
        h_t = self._gru_step(h_prev, mu, timestep)
        # Return shape (B, latent_dim). Caller unsqueezes to (B, 1, D).
        return h_t, th.zeros(1, device=h_t.device)

    def decode(
        self,
        z: th.Tensor,
        mu: th.Tensor,           # IGNORED — self-sufficient decoder
        timestep: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        """Decode from z + timestep only. mu param kept for interface compat."""
        ts_enc = self._timestep_encoding(timestep, z.size(0), z.device)
        return self.decoder(z, ts_enc)

    def forward(
        self,
        h_prev: th.Tensor,
        mu: th.Tensor,
        mu_next: Optional[th.Tensor] = None,
        skips: Optional[List[th.Tensor]] = None,
        timestep: Optional[th.Tensor] = None,
    ) -> WynerOutput:
        """
        Full forward pass with prior, GRU step, posterior, sampling, decoding.

        Args:
            h_prev:   (B, 1, D) or (B, D) — deterministic state from previous step
            mu:       (B, mu_dim)          — current transition encoding
            mu_next:  (B, mu_dim) or None  — next transition encoding (training only)
            timestep: (B,) or None         — episode timestep index

        Returns:
            WynerOutput with:
              w            = h_t  (deterministic state for memory forwarding)
              posterior_mu = posterior mean (for z sampling / KL)
              logvar       = posterior log-variance
              prior_mu/prior_logvar = prior params
              recon / recon_next    = decoded reconstructions
        """
        # ── 1. Prior: predict from h_{t-1} BEFORE seeing μ_t ─────
        prior_mu, prior_logvar = self._prior(h_prev)

        # ── 2. GRU step: h_t = GRU(μ_t, h_{t-1}) ────────────────
        h_t = self._gru_step(h_prev, mu, timestep)

        # ── 3. Posterior: infer from h_t AFTER seeing μ_t ────────
        #    + optionally μ_{t+1} for Wyner bidirectional constraint
        post_mu, post_logvar = self._posterior(h_t, mu_next)

        # ── 4. Reparameterised sample ────────────────────────────
        std = th.exp(0.5 * post_logvar)
        z = post_mu + std * th.randn_like(std)

        # ── 5. Decode current timestep ───────────────────────────
        ts_enc = self._timestep_encoding(timestep, z.size(0), z.device)
        recon = self.decoder(z, ts_enc)

        # ── 6. Decode next timestep (training only) ──────────────
        recon_next: Optional[th.Tensor] = None
        if mu_next is not None:
            if timestep is not None:
                ts_enc_next = sinusoidal_timestep_encoding(timestep + 1, self.pos_embed_dim)
            else:
                ts_enc_next = th.zeros(z.size(0), self.pos_embed_dim, device=z.device)
            recon_next = self.decoder(z, ts_enc_next)

        # ── 7. Pack output ───────────────────────────────────────
        # w = h_t for memory forwarding (unsqueeze to match buffer shape)
        return WynerOutput(
            w=h_t.unsqueeze(1),          # (B, 1, D) — stored as memory
            logvar=post_logvar,          # (B, D)
            recon=recon,                 # (B, recon_dim)
            recon_next=recon_next,       # (B, recon_dim) or None
            prior_mu=prior_mu,           # (B, D)
            prior_logvar=prior_logvar,   # (B, D)
            posterior_mu=post_mu,        # (B, D) — separate from w!
        )

    def loss(
        self,
        output: WynerOutput,
        recon_target: Optional[th.Tensor] = None,
        recon_next_target: Optional[th.Tensor] = None,
    ) -> WynerLoss:
        """
        Compute KL(posterior ‖ prior) + reconstruction losses.

        KL uses the learned prior (not N(0,I)), with per-dimension free_bits
        and sum reduction — so free_bits=0.1 with latent_dim=64 gives an
        effective floor of 6.4 nats per sample.
        """
        # Posterior mean: use dedicated field, fall back to w for backward compat
        post_mu = output.posterior_mu if output.posterior_mu is not None else output.w.squeeze(1)

        if output.prior_mu is not None and output.prior_logvar is not None:
            # KL(q ‖ p) for two Gaussians
            kl_per_dim = 0.5 * (
                output.prior_logvar - output.logvar
                + (output.logvar.exp() + (post_mu - output.prior_mu).pow(2))
                  / output.prior_logvar.exp()
                - 1.0
            )
        else:
            # Fallback: KL against N(0, I)
            kl_per_dim = -0.5 * (1 + output.logvar - post_mu.pow(2) - output.logvar.exp())

        # Per-dimension free_bits, then SUM over latent dims (not mean!)
        kl_loss = kl_per_dim.clamp_min(self.free_bits).sum(dim=-1)  # (B,)

        # Reconstruction loss
        recon_loss: Optional[th.Tensor] = None
        if recon_target is not None:
            recon_loss = nn.functional.mse_loss(
                output.recon, recon_target, reduction='none'
            ).mean(dim=-1)  # (B,)

        recon_next_loss: Optional[th.Tensor] = None
        if recon_next_target is not None and output.recon_next is not None:
            recon_next_loss = nn.functional.mse_loss(
                output.recon_next, recon_next_target, reduction='none'
            ).mean(dim=-1)  # (B,)

        return WynerLoss(
            kl_loss=kl_loss,
            recon_loss=recon_loss,
            recon_next_loss=recon_next_loss,
        )

    def sample_z(self, output: WynerOutput) -> th.Tensor:
        """
        Sample z from the posterior stored in a WynerOutput.
        Convenience method for QA loss computation in the training loop.
        """
        post_mu = output.posterior_mu if output.posterior_mu is not None else output.w.squeeze(1)
        std = th.exp(0.5 * output.logvar)
        return post_mu + std * th.randn_like(std)

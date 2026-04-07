"""
wyner.py — redesigned for true SSM recurrence.

Memory contract:
  GRU models:   memory = GRU hidden h, shape (B, latent_dim)
  WynerMambaV2: memory = Mamba flat state, shape (B, flat_state_dim)

WynerOutput.w   = new memory state (what goes back into the buffer)
WynerOutput.z_mu = latent mean       (what goes to the features extractor)

encode() return contract:
  (z_mu: Tensor[B, latent_dim],  new_memory: Tensor[B, *])
"""

import torch as th
import torch.nn as nn
from typing import Optional, List, Tuple
from models.mamba_cell import MambaCell
import math


# ── Sinusoidal timestep encoding ─────────────────────────────────────────────

@th.jit.script
def sinusoidal_timestep_encoding(timesteps: th.Tensor, embed_dim: int) -> th.Tensor:
    timesteps = timesteps.view(-1)
    half = embed_dim // 2
    freqs = th.exp(
        -math.log(10000.0) * th.arange(half, dtype=th.float32, device=timesteps.device) / half
    )
    angles = timesteps.unsqueeze(1).float() * freqs.unsqueeze(0)
    pe = th.cat([th.sin(angles), th.cos(angles)], dim=-1)
    if embed_dim % 2 == 1:
        pe = th.cat([pe, th.zeros(pe.size(0), 1, device=pe.device)], dim=-1)
    return pe


# ── Shared prior network ──────────────────────────────────────────────────────

class WynerPriorNetwork(nn.Module):
    """Learned prior p_theta(z | context). context is always latent_dim."""
    def __init__(self, latent_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.fc_mean   = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        nn.init.zeros_(self.fc_logvar.bias)

    def forward(self, context: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        x = context.squeeze(1) if context.dim() == 3 else context
        h = self.net(x)
        return self.fc_mean(h), self.fc_logvar(h).clamp(-8, 8)


# ── Data classes ─────────────────────────────────────────────────────────────

@th.jit.script
class WynerOutput:
    def __init__(
        self,
        w:            th.Tensor,                    # new memory state  (B, flat_state_dim or latent_dim)
        z_mu:         th.Tensor,                    # latent mean        (B, latent_dim)
        logvar:       th.Tensor,                    # latent log-var     (B, latent_dim)
        recon:        th.Tensor,
        recon_next:   Optional[th.Tensor] = None,
        prior_mu:     Optional[th.Tensor] = None,
        prior_logvar: Optional[th.Tensor] = None,
    ):
        self.w            = w
        self.z_mu         = z_mu
        self.logvar       = logvar
        self.recon        = recon
        self.recon_next   = recon_next
        self.prior_mu     = prior_mu
        self.prior_logvar = prior_logvar


@th.jit.script
class WynerLoss:
    def __init__(
        self,
        kl_loss:       th.Tensor,
        recon_loss:      Optional[th.Tensor] = None,
        recon_next_loss: Optional[th.Tensor] = None,
    ):
        self.kl_loss        = kl_loss
        self.recon_loss     = recon_loss
        self.recon_next_loss = recon_next_loss


@th.jit.interface
class WynerInterface:
    def encode(
        self,
        w: th.Tensor,
        mu: th.Tensor,
        skips: Optional[List[th.Tensor]],
        timestep: Optional[th.Tensor] = None,
    ) -> Tuple[th.Tensor, th.Tensor]:                  # (z_mu, new_memory_state)
        pass

    def decode(
        self,
        z: th.Tensor,
        mu: Optional[th.Tensor],
        timestep: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        pass

    def forward(
        self,
        w: th.Tensor,
        mu: th.Tensor,
        mu_next: Optional[th.Tensor],
        skips: Optional[List[th.Tensor]],
        timestep: Optional[th.Tensor] = None,
    ) -> WynerOutput:
        pass

    def loss(
        self,
        output: WynerOutput,
        recon_target: Optional[th.Tensor],
        recon_next_target: Optional[th.Tensor],
    ) -> WynerLoss:
        pass


# ── Shared decoders ───────────────────────────────────────────────────────────

class WynerDecoder(nn.Module):
    """mu-conditioned decoder — used by WynerVAE."""
    def __init__(self, latent_dim: int, mu_dim: int, recon_dim: int, decode_hidden: int = 128):
        super().__init__()
        self.proj   = nn.Linear(latent_dim, decode_hidden)
        self.fc1    = nn.Linear(decode_hidden + mu_dim, decode_hidden)
        self.fc2    = nn.Linear(decode_hidden, decode_hidden)
        self.fc_out = nn.Linear(decode_hidden, recon_dim)
        self.act    = nn.ReLU()

    def forward(self, z: th.Tensor, mu: th.Tensor) -> th.Tensor:
        x = th.cat([self.proj(z), mu], dim=-1)
        return self.fc_out(self.act(self.fc2(self.act(self.fc1(x)))))


class WynerIndependentDecoder(nn.Module):
    """Timestep-conditioned decoder — decodes at arbitrary future timesteps."""
    def __init__(self, latent_dim: int, recon_dim: int, decode_hidden: int = 128, pos_embed_dim: int = 16):
        super().__init__()
        self.proj   = nn.Linear(latent_dim, decode_hidden)
        self.fc1    = nn.Linear(decode_hidden + pos_embed_dim, decode_hidden)
        self.fc2    = nn.Sequential(
            nn.Linear(decode_hidden, decode_hidden), nn.ReLU(),
            nn.Linear(decode_hidden, decode_hidden), nn.ReLU(),
        )
        self.fc_out = nn.Linear(decode_hidden, recon_dim)
        self.act    = nn.ReLU()

    def forward(self, z: th.Tensor, ts_enc: th.Tensor) -> th.Tensor:
        x = th.cat([self.proj(z), ts_enc], dim=-1)
        return self.fc_out(self.fc2(self.act(self.fc1(x))))


# ─────────────────────────────────────────────────────────────────────────────
# Model 1: WynerVAE  (GRU, simple, no positional encoding, no learned prior)
# flat_state_dim = latent_dim  (GRU hidden IS the memory)
# ─────────────────────────────────────────────────────────────────────────────

class WynerVAE(nn.Module):
    def __init__(
        self,
        recon_dim:     int,
        mu_dim:        int,
        latent_dim:    int,
        decode_hidden: int   = 128,
        free_bits:     float = 0.0,
        **_,                                           # absorb unused kwargs cleanly
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.mu_dim     = mu_dim
        self.free_bits  = free_bits

        self.gru       = nn.GRUCell(input_size=mu_dim, hidden_size=latent_dim)
        self.fc_mean   = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)
        nn.init.zeros_(self.fc_logvar.bias)
        self.decoder   = WynerDecoder(latent_dim, mu_dim, recon_dim, decode_hidden)

    @property
    def flat_state_dim(self) -> int:
        return self.latent_dim

    def _step(self, w: th.Tensor, mu: th.Tensor) -> th.Tensor:
        return self.gru(mu, w.squeeze(1) if w.dim() == 3 else w)

    def encode(
        self,
        w: th.Tensor,
        mu: th.Tensor,
        skips: Optional[List[th.Tensor]] = None,
        timestep: Optional[th.Tensor] = None,
    ) -> Tuple[th.Tensor, th.Tensor]:
        h    = self._step(w, mu)
        z_mu = self.fc_mean(h)
        return z_mu, h                                 # (z_mu, new_memory=h)

    def decode(
        self,
        z: th.Tensor,
        mu: Optional[th.Tensor] = None,
        timestep: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        assert mu is not None, "WynerVAE.decode requires mu"
        return self.decoder(z, mu)

    def forward(
        self,
        w: th.Tensor,
        mu: th.Tensor,
        mu_next: Optional[th.Tensor] = None,
        skips: Optional[List[th.Tensor]] = None,
        timestep: Optional[th.Tensor] = None,
    ) -> WynerOutput:
        h        = self._step(w, mu)
        z_mu     = self.fc_mean(h)
        z_logvar = self.fc_logvar(h).clamp(-8, 8)
        z        = z_mu + th.randn_like(z_mu) * th.exp(0.5 * z_logvar)
        recon      = self.decoder(z, mu)
        recon_next = self.decoder(z, mu_next) if mu_next is not None else None
        return WynerOutput(w=h, z_mu=z_mu, logvar=z_logvar, recon=recon, recon_next=recon_next)

    def loss(
        self,
        output: WynerOutput,
        recon_target: Optional[th.Tensor] = None,
        recon_next_target: Optional[th.Tensor] = None,
    ) -> WynerLoss:
        kl = -0.5 * (1 + output.logvar - output.z_mu.pow(2) - output.logvar.exp())
        kl_loss = kl.clamp_min(self.free_bits).sum(dim=-1)
        r  = nn.functional.mse_loss(output.recon, recon_target, reduction='none').mean(-1) if recon_target is not None else None
        rn = nn.functional.mse_loss(output.recon_next, recon_next_target, reduction='none').mean(-1) if (recon_next_target is not None and output.recon_next is not None) else None
        return WynerLoss(kl_loss=kl_loss, recon_loss=r, recon_next_loss=rn)


# ─────────────────────────────────────────────────────────────────────────────
# Model 2: WynerIndependentVAE  (GRU + pos-embed + learned prior)
# flat_state_dim = latent_dim
# ─────────────────────────────────────────────────────────────────────────────

class WynerIndependentVAE(nn.Module):
    def __init__(
        self,
        recon_dim:     int,
        mu_dim:        int,
        latent_dim:    int,
        decode_hidden: int   = 128,
        free_bits:     float = 0.0,
        pos_embed_dim: int   = 16,
        **_,
    ):
        super().__init__()
        self.latent_dim    = latent_dim
        self.mu_dim        = mu_dim
        self.free_bits     = free_bits
        self.pos_embed_dim = pos_embed_dim

        self.proj_t    = nn.Linear(mu_dim, mu_dim)
        self.gru       = nn.GRUCell(input_size=mu_dim + pos_embed_dim, hidden_size=latent_dim)
        self.fc_mean   = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)
        nn.init.zeros_(self.fc_logvar.bias)
        self.prior_net = WynerPriorNetwork(latent_dim, hidden_dim=decode_hidden)
        self.decoder   = WynerIndependentDecoder(latent_dim, recon_dim, decode_hidden, pos_embed_dim)

    @property
    def flat_state_dim(self) -> int:
        return self.latent_dim

    def _ts(self, timestep: Optional[th.Tensor], B: int, device: th.device) -> th.Tensor:
        if timestep is not None:
            return sinusoidal_timestep_encoding(timestep, self.pos_embed_dim)
        return th.zeros(B, self.pos_embed_dim, device=device)

    def _step(self, w: th.Tensor, mu: th.Tensor, timestep: Optional[th.Tensor] = None) -> th.Tensor:
        h_prev = w.squeeze(1) if w.dim() == 3 else w
        x = th.cat([self.proj_t(mu), self._ts(timestep, mu.size(0), mu.device)], dim=-1)
        return self.gru(x, h_prev)

    def encode(
        self,
        w: th.Tensor,
        mu: th.Tensor,
        skips: Optional[List[th.Tensor]] = None,
        timestep: Optional[th.Tensor] = None,
    ) -> Tuple[th.Tensor, th.Tensor]:
        h    = self._step(w, mu, timestep)
        z_mu = self.fc_mean(h)
        return z_mu, h                                 # (z_mu, new_memory=h)

    def decode(
        self,
        z: th.Tensor,
        mu: Optional[th.Tensor] = None,
        timestep: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        return self.decoder(z, self._ts(timestep, z.size(0), z.device))

    def forward(
        self,
        w: th.Tensor,
        mu: th.Tensor,
        mu_next: Optional[th.Tensor] = None,
        skips: Optional[List[th.Tensor]] = None,
        timestep: Optional[th.Tensor] = None,
    ) -> WynerOutput:
        prior_mu, prior_logvar = self.prior_net(w.squeeze(1) if w.dim() == 3 else w)
        h        = self._step(w, mu, timestep)
        z_mu     = self.fc_mean(h)
        z_logvar = self.fc_logvar(h).clamp(-8, 8)
        z        = z_mu + th.randn_like(z_mu) * th.exp(0.5 * z_logvar)
        B, dev   = z.size(0), z.device
        recon    = self.decoder(z, self._ts(timestep, B, dev))
        recon_next = (
            self.decoder(z, self._ts(timestep + 1 if timestep is not None else None, B, dev))
            if mu_next is not None else None
        )
        return WynerOutput(
            w=h, z_mu=z_mu, logvar=z_logvar,
            recon=recon, recon_next=recon_next,
            prior_mu=prior_mu, prior_logvar=prior_logvar,
        )

    def loss(
        self,
        output: WynerOutput,
        recon_target: Optional[th.Tensor] = None,
        recon_next_target: Optional[th.Tensor] = None,
    ) -> WynerLoss:
        if output.prior_mu is not None and output.prior_logvar is not None:
            kl = 0.5 * (
                output.prior_logvar - output.logvar
                + (output.logvar.exp() + (output.z_mu - output.prior_mu).pow(2))
                  / output.prior_logvar.exp()
                - 1.0
            )
        else:
            kl = -0.5 * (1 + output.logvar - output.z_mu.pow(2) - output.logvar.exp())
        kl_loss = kl.clamp_min(self.free_bits).mean(dim=-1)
        r  = nn.functional.mse_loss(output.recon, recon_target, reduction='none').mean(-1) if recon_target is not None else None
        rn = nn.functional.mse_loss(output.recon_next, recon_next_target, reduction='none').mean(-1) if (recon_next_target is not None and output.recon_next is not None) else None
        return WynerLoss(kl_loss=kl_loss, recon_loss=r, recon_next_loss=rn)


# ─────────────────────────────────────────────────────────────────────────────
# Model 3: WynerMambaV2  (true SSM recurrence)
#
# Memory = full Mamba flat state (B, flat_state_dim).
# flat_state_dim >> latent_dim — use SMALL Mamba params to keep it manageable.
#
# Recommended defaults:  d_state=16, d_conv=2, expand=2, n_heads=1
# → flat_state_dim ≈ 1200  for latent_dim=64
#
# Prior is conditioned on the full SSM state via a projection:
#   flat_state → latent_dim → WynerPriorNetwork → (prior_mu, prior_logvar)
# ─────────────────────────────────────────────────────────────────────────────

class WynerMambaV2(nn.Module):
    def __init__(
        self,
        recon_dim:     int,
        mu_dim:        int,
        latent_dim:    int,
        decode_hidden: int   = 128,
        free_bits:     float = 0.0,
        pos_embed_dim: int   = 16,
        # Mamba hyperparams — keep small to control flat_state_dim
        d_state:       int   = 16,
        d_conv:        int   = 2,
        expand:        int   = 2,
        n_heads:       int   = 1,
        **_,
    ):
        super().__init__()
        self.latent_dim    = latent_dim
        self.mu_dim        = mu_dim
        self.pos_embed_dim = pos_embed_dim
        self.free_bits     = free_bits

        self.proj_t = nn.Linear(mu_dim, mu_dim)

        # Mamba cell: input = mu_proj + pos_embed, output = hidden_dim = latent_dim
        self.mamba_cell = MambaCell(
            input_dim  = mu_dim + pos_embed_dim,
            hidden_dim = latent_dim,
            d_state    = d_state,
            d_conv     = d_conv,
            expand     = expand,
            n_heads    = n_heads,
        )
        self._flat_state_dim: int = self.mamba_cell.flat_state_dim

        # z computed from Mamba output y: (B, latent_dim) → (B, latent_dim)
        self.fc_mean   = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)
        nn.init.zeros_(self.fc_logvar.bias)

        # Prior conditioned on the full SSM flat state
        #   flat_state → latent_dim → WynerPriorNetwork
        self.prior_flat_proj = nn.Linear(self._flat_state_dim, latent_dim)
        self.prior_net       = WynerPriorNetwork(latent_dim, hidden_dim=decode_hidden)

        self.decoder = WynerIndependentDecoder(latent_dim, recon_dim, decode_hidden, pos_embed_dim)

    @property
    def flat_state_dim(self) -> int:
        return self._flat_state_dim

    def _ts(self, timestep: Optional[th.Tensor], B: int, device: th.device) -> th.Tensor:
        if timestep is not None:
            return sinusoidal_timestep_encoding(timestep, self.pos_embed_dim)
        return th.zeros(B, self.pos_embed_dim, device=device)

    def _step(
        self,
        w: th.Tensor,                                  # (B, flat_state_dim) — true SSM state
        mu: th.Tensor,
        timestep: Optional[th.Tensor] = None,
    ) -> Tuple[th.Tensor, th.Tensor]:
        """One Mamba step. Returns (y: B×latent_dim, new_flat_state: B×flat_state_dim)."""
        w_flat = w.squeeze(1) if w.dim() == 3 else w  # handle legacy (B,1,*) if passed
        x = th.cat([self.proj_t(mu), self._ts(timestep, mu.size(0), mu.device)], dim=-1)
        y, new_flat_state = self.mamba_cell(x, w_flat)   # mamba_cell.forward returns (y, h_new)
        return y, new_flat_state

    def _prior(self, w: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        w_flat  = w.squeeze(1) if w.dim() == 3 else w
        context = self.prior_flat_proj(w_flat)         # (B, latent_dim)
        return self.prior_net(context)

    # ── Interface ──────────────────────────────────────────────────────────

    def encode(
        self,
        w: th.Tensor,
        mu: th.Tensor,
        skips: Optional[List[th.Tensor]] = None,
        timestep: Optional[th.Tensor] = None,
    ) -> Tuple[th.Tensor, th.Tensor]:
        """Returns (z_mu, new_flat_state). new_flat_state is stored in the buffer."""
        y, new_flat = self._step(w, mu, timestep)
        z_mu = self.fc_mean(y)
        return z_mu, new_flat

    def decode(
        self,
        z: th.Tensor,
        mu: Optional[th.Tensor] = None,               # unused, kept for interface compat
        timestep: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        return self.decoder(z, self._ts(timestep, z.size(0), z.device))

    def forward(
        self,
        w: th.Tensor,
        mu: th.Tensor,
        mu_next: Optional[th.Tensor] = None,
        skips: Optional[List[th.Tensor]] = None,
        timestep: Optional[th.Tensor] = None,
    ) -> WynerOutput:
        prior_mu, prior_logvar = self._prior(w)

        y, new_flat = self._step(w, mu, timestep)
        z_mu     = self.fc_mean(y)
        z_logvar = self.fc_logvar(y).clamp(-8, 8)
        z        = z_mu + th.randn_like(z_mu) * th.exp(0.5 * z_logvar)

        B, dev   = z.size(0), z.device
        recon    = self.decoder(z, self._ts(timestep, B, dev))
        recon_next = (
            self.decoder(z, self._ts(
                timestep + 1 if timestep is not None else None, B, dev
            ))
            if mu_next is not None else None
        )

        return WynerOutput(
            w=new_flat,                                # ← SSM flat state → buffer
            z_mu=z_mu,                                 # ← latent mean → features extractor
            logvar=z_logvar,
            recon=recon,
            recon_next=recon_next,
            prior_mu=prior_mu,
            prior_logvar=prior_logvar,
        )

    def loss(
        self,
        output: WynerOutput,
        recon_target: Optional[th.Tensor] = None,
        recon_next_target: Optional[th.Tensor] = None,
    ) -> WynerLoss:
        if output.prior_mu is not None and output.prior_logvar is not None:
            kl = 0.5 * (
                output.prior_logvar - output.logvar
                + (output.logvar.exp() + (output.z_mu - output.prior_mu).pow(2))
                  / output.prior_logvar.exp()
                - 1.0
            )
        else:
            kl = -0.5 * (1 + output.logvar - output.z_mu.pow(2) - output.logvar.exp())
        kl_loss = kl.clamp_min(self.free_bits).mean(dim=-1)
        r  = nn.functional.mse_loss(output.recon, recon_target, reduction='none').mean(-1) if recon_target is not None else None
        rn = nn.functional.mse_loss(output.recon_next, recon_next_target, reduction='none').mean(-1) if (recon_next_target is not None and output.recon_next is not None) else None
        return WynerLoss(kl_loss=kl_loss, recon_loss=r, recon_next_loss=rn)
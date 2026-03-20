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
    ):
        self.recon = recon
        self.recon_next = recon_next
        self.w = w
        self.logvar = logvar
        self.prior_mu = prior_mu
        self.prior_logvar = prior_logvar


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
        free_bits: float = 0.5,
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
        self.proj_tp1 = nn.Linear(mu_dim, mu_dim)

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
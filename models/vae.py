import torch as th
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, List, Tuple

from .utils import sc_kl_uniform, uniformity_loss, sc_sample
from .embeddings import EmbeddingInterface
from .config import SCVAEConfig, GaussianVAEConfig
from .encoders import ConvEncoder


@th.jit.script
class VAEOutput:
    def __init__(
        self,
        recon: th.Tensor,
        mu: th.Tensor,
        rho: th.Tensor,
        recon_target: th.Tensor,
        skips: Optional[List[th.Tensor]] = None,
        fwd_pred: Optional[th.Tensor] = None,
        fwd_target: Optional[th.Tensor] = None,
    ):
        self.recon        = recon
        self.mu           = mu
        self.rho          = rho
        self.recon_target = recon_target
        self.skips        = skips
        self.fwd_pred     = fwd_pred
        self.fwd_target   = fwd_target


@th.jit.script
class VAELoss:
    def __init__(
        self,
        recon_loss: th.Tensor,
        kl_loss: th.Tensor,
        aux_loss: Optional[th.Tensor] = None,
        fwd_loss: Optional[th.Tensor] = None,
    ):
        self.recon_loss = recon_loss
        self.kl_loss    = kl_loss
        self.aux_loss   = aux_loss
        self.fwd_loss   = fwd_loss


@th.jit.interface
class VAEInterface:
    def encode(
        self, s_t: th.Tensor, a_t: th.Tensor, s_tp1: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, Optional[List[th.Tensor]]]:
        pass

    def decode(
        self, z: th.Tensor, skips: Optional[List[th.Tensor]]
    ) -> th.Tensor:
        pass

    def build_target(
        self, s_t: th.Tensor, a_t: th.Tensor, s_tp1: th.Tensor
    ) -> th.Tensor:
        pass

    def forward(
        self, s_t: th.Tensor, a_t: th.Tensor, s_tp1: th.Tensor
    ) -> VAEOutput:
        pass

    def loss(self, output: VAEOutput) -> VAELoss:
        pass

    def intrinsic_reward(
        self, s_t: th.Tensor, a_t: th.Tensor, s_tp1: th.Tensor
    ) -> th.Tensor:
        pass


class ForwardPredictor(nn.Module):
    """Predicts h_{t+1} from (z, h_t, a_emb). Used to compute intrinsic reward."""

    def __init__(self, z_dim: int, h_dim: int, a_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + h_dim + a_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, h_dim),
        )

    def forward(self, z: th.Tensor, h_t: th.Tensor, a_emb: th.Tensor) -> th.Tensor:
        return self.net(th.cat([z, h_t, a_emb], dim=-1))


class TransitionSCVAE(nn.Module):
    """
    Spherical Cauchy VAE over (s_t, a_t, s_{t+1}) transitions.
    """

    def __init__(self, embedding: EmbeddingInterface, cfg: SCVAEConfig):
        super().__init__()

        self.cfg        = cfg
        self.embedding  = embedding
        self.latent_dim = cfg.latent_dim

        self.conv = ConvEncoder(
            embedding.meta().out_channels,
            cfg.conv_channels,
        )

        E = self.conv.out_dim
        A = cfg.action_embed_dim
        H = cfg.hidden_dim

        self.action_embed = nn.Sequential(
            nn.Linear(cfg.act_dim, A),
            nn.ReLU(),
        )

        self.encoder_trunk = nn.Sequential(
            nn.Linear(2 * E + A, H),
            nn.LayerNorm(H),
            nn.ReLU(),
            nn.Linear(H, H),
            nn.LayerNorm(H),
            nn.ReLU(),
        )

        self.fc_mu  = nn.Linear(H, cfg.latent_dim)
        self.fc_rho = nn.Linear(H, 1)
        # Initialize rho bias so sigmoid(-2.0) ≈ 0.12, starting near uniform on S^{d-1}
        nn.init.constant_(self.fc_rho.bias, -2.0)

        self._recon_dim = 2 * E + A

        self.decoder_trunk = nn.Sequential(
            nn.Linear(cfg.latent_dim, H),
            nn.ReLU(),
            nn.Linear(H, H),
            nn.ReLU(),
        )
        self.state_t_head    = nn.Linear(H, E)
        self.action_head     = nn.Linear(H, A)
        self.state_next_head = nn.Linear(H, E)

        # Forward predictor: (z, h_s, a_emb) → ĥ_sn
        self.forward_predictor = ForwardPredictor(
            z_dim=cfg.latent_dim, h_dim=E, a_dim=A, hidden=H,
        )


    # ---------------------------------------------------------
    # Private helpers
    # ---------------------------------------------------------

    def _embed(self, obs: th.Tensor) -> th.Tensor:
        return self.embedding(obs)

    def _encode_parts(
        self,
        s_t: th.Tensor,
        a_t: th.Tensor,
        s_next: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        h_s   = self.conv(self._embed(s_t))
        h_sn  = self.conv(self._embed(s_next))
        a = a_t.float()
        if a.dim() == 1:
            a = a.unsqueeze(-1)  # (B,) → (B, 1) for Discrete actions
        a_emb = self.action_embed(a)
        return h_s, a_emb, h_sn

    def build_target(
        self,
        s_t: th.Tensor,
        a_t: th.Tensor,
        s_next: th.Tensor,
    ) -> th.Tensor:
        # stop-gradient target (intentional — encoder only learns through KL)
        with th.no_grad():
            h_s, a_emb, h_sn = self._encode_parts(s_t, a_t, s_next)
        return th.cat([h_s, a_emb, h_sn], dim=-1)
    
    def encode_state(self, s_t: th.Tensor) -> th.Tensor:
        return self._embed(s_t)

    # ---------------------------------------------------------
    # Interface methods
    # ---------------------------------------------------------

    def encode(
        self,
        s_t: th.Tensor,
        a_t: th.Tensor,
        s_next: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor, Optional[List[th.Tensor]]]:

        h_s, a_emb, h_sn = self._encode_parts(s_t, a_t, s_next)

        h = self.encoder_trunk(
            th.cat([h_s, a_emb, h_sn], dim=-1)
        )

        mu  = F.normalize(self.fc_mu(h), p=2, dim=-1)
        rho = th.sigmoid(self.fc_rho(h))

        skips: Optional[List[th.Tensor]] = None 

        return mu, rho, skips

    def decode(
        self,
        z: th.Tensor,
        skips: Optional[List[th.Tensor]],          
    ) -> th.Tensor:
        h = self.decoder_trunk(z)
        s_t_recon    = self.state_t_head(h)
        a_recon      = self.action_head(h)
        s_next_recon = self.state_next_head(h)
        out = th.cat([s_t_recon, a_recon, s_next_recon], dim=-1)
        assert out.shape[-1] == self._recon_dim, f"Reconstruction dimension mismatch: {out.shape[-1]} != {self._recon_dim}"
        return out

    def forward(
        self,
        s_t: th.Tensor,
        a_t: th.Tensor,
        s_next: th.Tensor,
    ) -> VAEOutput:

        h_s, a_emb, h_sn = self._encode_parts(s_t, a_t, s_next)

        h = self.encoder_trunk(th.cat([h_s, a_emb, h_sn], dim=-1))
        mu  = F.normalize(self.fc_mu(h), p=2, dim=-1)
        rho = th.sigmoid(self.fc_rho(h))

        z = sc_sample(mu, rho) if self.training else mu

        skips: Optional[List[th.Tensor]] = None
        recon        = self.decode(z, skips)
        recon_target = th.cat([h_s.detach(), a_emb.detach(), h_sn.detach()], dim=-1)

        # Forward prediction: predict h_{t+1} from (z, h_t, a_emb)
        fwd_pred   = self.forward_predictor(z, h_s, a_emb)
        fwd_target = h_sn.detach()  # stop-gradient target

        return VAEOutput(recon, mu, rho, recon_target, skips, fwd_pred, fwd_target)

    def intrinsic_reward(
        self,
        s_t: th.Tensor,
        a_t: th.Tensor,
        s_next: th.Tensor,
    ) -> th.Tensor:
        """Compute intrinsic reward = ||fwd_pred - sg(h_sn)||² per sample."""
        with th.no_grad():
            h_s, a_emb, h_sn = self._encode_parts(s_t, a_t, s_next)
            h = self.encoder_trunk(th.cat([h_s, a_emb, h_sn], dim=-1))
            mu = F.normalize(self.fc_mu(h), p=2, dim=-1)
            # Use mu directly (no sampling) for stable intrinsic reward
            fwd_pred = self.forward_predictor(mu, h_s, a_emb)
            return (fwd_pred - h_sn).pow(2).sum(dim=-1)

    def loss(self, out: VAEOutput) -> VAELoss:

        l_recon = F.mse_loss(out.recon, out.recon_target)

        l_kl = sc_kl_uniform(
            out.rho,
            self.latent_dim,
        ).clamp_min(self.cfg.free_bits).mean()

        l_uniform = uniformity_loss(
            out.mu,
            t=self.cfg.uniformity_t,
        )

        l_fwd: Optional[th.Tensor] = None
        if out.fwd_pred is not None and out.fwd_target is not None:
            l_fwd = F.mse_loss(out.fwd_pred, out.fwd_target)

        return VAELoss(
            recon_loss = l_recon,
            kl_loss    = l_kl,
            aux_loss   = self.cfg.gamma * l_uniform,
            fwd_loss   = l_fwd,
        )


class TransitionGaussianVAE(nn.Module):
    """
    Gaussian VAE baseline over (s_t, a_t, s_{t+1}) transitions.
    Same architecture as TransitionSCVAE but with:
      - fc_mu (NO L2-norm)
      - fc_logvar
      - Standard Gaussian reparameterization: z = mu + std * eps
      - KL to N(0,I) with optional free-bits per dimension
    """

    def __init__(self, embedding: EmbeddingInterface, cfg: GaussianVAEConfig):
        super().__init__()

        self.cfg        = cfg
        self.embedding  = embedding
        self.latent_dim = cfg.latent_dim

        self.conv = ConvEncoder(
            embedding.meta().out_channels,
            cfg.conv_channels,
        )

        E = self.conv.out_dim
        A = cfg.action_embed_dim
        H = cfg.hidden_dim

        self.action_embed = nn.Sequential(
            nn.Linear(cfg.act_dim, A),
            nn.ReLU(),
        )

        self.encoder_trunk = nn.Sequential(
            nn.Linear(2 * E + A, H),
            nn.LayerNorm(H),
            nn.ReLU(),
            nn.Linear(H, H),
            nn.LayerNorm(H),
            nn.ReLU(),
        )

        self.fc_mu     = nn.Linear(H, cfg.latent_dim)
        self.fc_logvar = nn.Linear(H, cfg.latent_dim)

        self._recon_dim = 2 * E + A

        self.decoder_trunk = nn.Sequential(
            nn.Linear(cfg.latent_dim, H),
            nn.ReLU(),
            nn.Linear(H, H),
            nn.ReLU(),
        )
        self.state_t_head    = nn.Linear(H, E)
        self.action_head     = nn.Linear(H, A)
        self.state_next_head = nn.Linear(H, E)

        # Forward predictor: (z, h_s, a_emb) → ĥ_sn
        self.forward_predictor = ForwardPredictor(
            z_dim=cfg.latent_dim, h_dim=E, a_dim=A, hidden=H,
        )

    # ---------------------------------------------------------
    # Private helpers
    # ---------------------------------------------------------

    def _embed(self, obs: th.Tensor) -> th.Tensor:
        return self.embedding(obs)

    def _encode_parts(
        self,
        s_t: th.Tensor,
        a_t: th.Tensor,
        s_next: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        h_s   = self.conv(self._embed(s_t))
        h_sn  = self.conv(self._embed(s_next))
        a = a_t.float()
        if a.dim() == 1:
            a = a.unsqueeze(-1)
        a_emb = self.action_embed(a)
        return h_s, a_emb, h_sn

    def build_target(
        self,
        s_t: th.Tensor,
        a_t: th.Tensor,
        s_next: th.Tensor,
    ) -> th.Tensor:
        with th.no_grad():
            h_s, a_emb, h_sn = self._encode_parts(s_t, a_t, s_next)
        return th.cat([h_s, a_emb, h_sn], dim=-1)

    def encode_state(self, s_t: th.Tensor) -> th.Tensor:
        return self._embed(s_t)

    # ---------------------------------------------------------
    # Interface methods
    # ---------------------------------------------------------

    def encode(
        self,
        s_t: th.Tensor,
        a_t: th.Tensor,
        s_next: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor, Optional[List[th.Tensor]]]:
        h_s, a_emb, h_sn = self._encode_parts(s_t, a_t, s_next)
        h = self.encoder_trunk(th.cat([h_s, a_emb, h_sn], dim=-1))
        mu     = self.fc_mu(h)        # NO L2-norm
        logvar = self.fc_logvar(h)
        skips: Optional[List[th.Tensor]] = None
        return mu, logvar, skips

    def decode(
        self,
        z: th.Tensor,
        skips: Optional[List[th.Tensor]],
    ) -> th.Tensor:
        h = self.decoder_trunk(z)
        s_t_recon    = self.state_t_head(h)
        a_recon      = self.action_head(h)
        s_next_recon = self.state_next_head(h)
        out = th.cat([s_t_recon, a_recon, s_next_recon], dim=-1)
        return out

    def forward(
        self,
        s_t: th.Tensor,
        a_t: th.Tensor,
        s_next: th.Tensor,
    ) -> VAEOutput:
        h_s, a_emb, h_sn = self._encode_parts(s_t, a_t, s_next)

        h      = self.encoder_trunk(th.cat([h_s, a_emb, h_sn], dim=-1))
        mu     = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        if self.training:
            std = th.exp(0.5 * logvar)
            z   = mu + std * th.randn_like(std)
        else:
            z = mu

        skips: Optional[List[th.Tensor]] = None
        recon        = self.decode(z, skips)
        recon_target = th.cat([h_s.detach(), a_emb.detach(), h_sn.detach()], dim=-1)

        # Forward prediction
        fwd_pred   = self.forward_predictor(z, h_s, a_emb)
        fwd_target = h_sn.detach()

        # Store logvar in rho slot for the loss function
        return VAEOutput(recon, mu, logvar, recon_target, skips, fwd_pred, fwd_target)

    def intrinsic_reward(
        self,
        s_t: th.Tensor,
        a_t: th.Tensor,
        s_next: th.Tensor,
    ) -> th.Tensor:
        """Compute intrinsic reward = ||fwd_pred - sg(h_sn)||² per sample."""
        with th.no_grad():
            h_s, a_emb, h_sn = self._encode_parts(s_t, a_t, s_next)
            h  = self.encoder_trunk(th.cat([h_s, a_emb, h_sn], dim=-1))
            mu = self.fc_mu(h)
            fwd_pred = self.forward_predictor(mu, h_s, a_emb)
            return (fwd_pred - h_sn).pow(2).sum(dim=-1)

    def loss(self, out: VAEOutput) -> VAELoss:
        # out.rho holds logvar for Gaussian
        logvar = out.rho

        l_recon = F.mse_loss(out.recon, out.recon_target)

        # KL to N(0,I) per dimension, with optional free-bits clamp
        # KL_j = -0.5 * (1 + logvar_j - mu_j^2 - exp(logvar_j))
        kl_per_dim = -0.5 * (1.0 + logvar - out.mu.pow(2) - logvar.exp())
        if self.cfg.free_bits > 0:
            kl_per_dim = kl_per_dim.clamp_min(self.cfg.free_bits)
        l_kl = kl_per_dim.sum(dim=-1).mean()

        l_fwd: Optional[th.Tensor] = None
        if out.fwd_pred is not None and out.fwd_target is not None:
            l_fwd = F.mse_loss(out.fwd_pred, out.fwd_target)

        return VAELoss(
            recon_loss = l_recon,
            kl_loss    = l_kl,
            aux_loss   = None,
            fwd_loss   = l_fwd,
        )
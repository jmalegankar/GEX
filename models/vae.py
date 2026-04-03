import torch as th
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, List, Tuple, overload

from .utils import sc_kl_uniform, uniformity_loss, sc_sample
from .embeddings import EmbeddingInterface
from .config import SCVAEConfig
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
    ):
        self.recon        = recon
        self.mu           = mu
        self.rho          = rho
        self.recon_target = recon_target
        self.skips        = skips


@th.jit.script
class VAELoss:
    def __init__(
        self,
        recon_loss: th.Tensor,
        kl_loss: th.Tensor,
        aux_loss: Optional[th.Tensor] = None,
    ):
        self.recon_loss = recon_loss
        self.kl_loss    = kl_loss
        self.aux_loss   = aux_loss


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

        # self.action_embed = nn.Sequential(
        #     nn.Linear(cfg.act_dim, A),
        #     nn.ReLU(),
        # )
        self.action_embed = nn.Embedding(cfg.act_dim, A)

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
        # a = a_t.float()
        # if a.dim() == 1:
        #     a = a.unsqueeze(-1)  # (B,) → (B, 1) for Discrete actions
        # Convert float back to int
        a = a_t.long()
        a_emb = self.action_embed(a).view(-1, self.cfg.action_embed_dim)
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
    
    def encode_direct(
        self,
        encoded: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor, Optional[List[th.Tensor]]]:
        
        h = self.encoder_trunk(encoded)

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

        mu, rho, skips = self.encode(s_t, a_t, s_next)

        z = sc_sample(mu, rho) if self.training else mu

        recon        = self.decode(z, skips)
        recon_target = self.build_target(s_t, a_t, s_next)

        return VAEOutput(recon, mu, rho, recon_target, skips)

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

        return VAELoss(
            recon_loss = l_recon,
            kl_loss    = l_kl,
            aux_loss   = self.cfg.gamma * l_uniform,
        )
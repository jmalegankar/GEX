from __future__ import annotations

from typing import NamedTuple, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from obs_embeddings import ObservationEmbedding
from config import SCVAEConfig


# ===================================================================
# Spherical Cauchy helpers
# ===================================================================

_KL_SERIES_CACHE = {}

def _mobius_add(a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    a_sq = (a * a).sum(-1, keepdim=True)
    x_sq = (x * x).sum(-1, keepdim=True)
    ax   = (a * x).sum(-1, keepdim=True)
    num  = (1 + 2 * ax + x_sq) * a + (1 - a_sq) * x
    den  = 1 + 2 * ax + a_sq * x_sq
    return F.normalize(num / (den + 1e-8), p=2, dim=-1)


def _sc_sample(mu: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    xi = F.normalize(torch.randn_like(mu), p=2, dim=-1)
    return _mobius_add(rho * mu, xi)


def _sc_z_of_rho(rho: torch.Tensor) -> torch.Tensor:
    return 4.0 * rho / (1.0 + rho).pow(2)

def _get_kl_series_terms(dim: int, max_terms: int, device, dtype):
    """
    Returns tensor of shape (K,) containing:
        coeff_k * (psi(d-1+k) - psi(d-1))
    Cached per (dim, max_terms, device, dtype).
    """
    key = (dim, max_terms, device.type, str(dtype))

    if key in _KL_SERIES_CACHE:
        return _KL_SERIES_CACHE[key]

    d_minus_1 = float(dim - 1)
    a = d_minus_1 / 2.0

    k = torch.arange(1, max_terms + 1, device=device, dtype=dtype)

    a_t = torch.tensor(a, device=device, dtype=dtype)

    log_coeff = (
        torch.lgamma(a_t + k)
        - torch.lgamma(a_t)
        - torch.lgamma(k + 1.0)
    )

    coeff = torch.exp(log_coeff)

    psi_base = torch.digamma(torch.tensor(d_minus_1, device=device, dtype=dtype))
    psi_diff = torch.digamma(d_minus_1 + k) - psi_base

    scalar_terms = coeff * psi_diff  # (K,)

    _KL_SERIES_CACHE[key] = scalar_terms

    return scalar_terms

def _sc_kl_uniform(
    rho: torch.Tensor,
    dim: int,
    *,
    max_terms: int = 64,
) -> torch.Tensor:

    if dim < 2:
        raise ValueError("dim must be >= 2")

    if rho.dim() == 1:
        rho = rho.unsqueeze(-1)

    eps = 1e-7
    rho = rho.clamp(0.0, 1.0 - eps)

    device = rho.device
    dtype = rho.dtype

    d_minus_1 = float(dim - 1)

    # ---- First term ----
    log_ratio = torch.log1p(-rho) - torch.log1p(rho)
    term1 = d_minus_1 * log_ratio

    # ---- Prefactor ----
    ratio = (1.0 - rho) / (1.0 + rho)
    pref = d_minus_1 * ratio.pow(d_minus_1)

    # ---- z(rho) ----
    z = 4.0 * rho / (1.0 + rho).pow(2)  # (B,1)

    # ---- Cached scalar series terms ----
    scalar_terms = _get_kl_series_terms(dim, max_terms, device, dtype)  # (K,)

    k = torch.arange(1, max_terms + 1, device=device, dtype=dtype)

    # (B,K)
    z_pow = z.pow(k)

    series = (z_pow * scalar_terms).sum(dim=-1, keepdim=True)

    kl = term1 + pref * series

    return kl.clamp_min(0.0)


# ===================================================================
# Output type
# ===================================================================

class SCVAEForwardOutput(NamedTuple):
    recon: torch.Tensor
    mu: torch.Tensor
    rho: torch.Tensor
    z: torch.Tensor
    recon_target: torch.Tensor


# ===================================================================
# Conv block
# ===================================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ===================================================================
# TransitionSCVAE
# ===================================================================

class TransitionSCVAE(nn.Module):
    """
    Env-agnostic Spherical Cauchy VAE.

    Contract:
      embedding(obs) -> (B, C, H, W) float tensor
      SCVAE never tries to infer obs layout.
    """

    def __init__(
        self,
        embedding: ObservationEmbedding,
        cfg: SCVAEConfig,
        sample_input_shape: Tuple[int, ...],  # raw obs shape WITHOUT batch
    ):
        super().__init__()
        self.embedding = embedding
        self.cfg = cfg
        self.latent_dim = cfg.latent_dim

        # --- conv encoder ---
        layers = []
        in_ch = embedding.out_channels
        for ch in cfg.conv_channels:
            layers.append(ConvBlock(in_ch, ch))
            in_ch = ch
        self.conv = nn.Sequential(*layers)

        # --- infer conv feature dim ---
        with torch.no_grad():
            dummy = torch.zeros((1, *sample_input_shape))
            dummy_emb = self._embed(dummy)
            dummy_feat = self.conv(dummy_emb)
            self._feat_dim = dummy_feat.flatten(1).shape[1]

        # --- action embedding ---
        self.action_embed = nn.Embedding(cfg.n_actions, cfg.action_embed_dim)

        # --- bottleneck ---
        self.fc = nn.Sequential(
            nn.Linear(2 * self._feat_dim + cfg.action_embed_dim, cfg.hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(cfg.hidden_dim, cfg.latent_dim)
        self.fc_rho = nn.Linear(cfg.hidden_dim, 1)

        # --- decoder ---
        self.decoder_trunk = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
        )
        self.state_t_head = nn.Linear(cfg.hidden_dim, self._feat_dim)
        self.action_head = nn.Linear(cfg.hidden_dim, cfg.action_embed_dim)
        self.state_next_head = nn.Linear(cfg.hidden_dim, self._feat_dim)

    # ============================================================
    # Private helpers (kept above public API as requested)
    # ============================================================

    def _embed(self, obs: torch.Tensor) -> torch.Tensor:
        # embedding must handle dtype/layout; enforce float output
        x = self.embedding(obs)
        if x.dim() != 4:
            raise ValueError(f"embedding must return (B,C,H,W), got shape {tuple(x.shape)}")
        return x

    def _encode_parts(self, s_t: torch.Tensor, a_t: torch.Tensor, s_next: torch.Tensor):
        h_s = self.conv(self._embed(s_t)).flatten(1)
        h_sn = self.conv(self._embed(s_next)).flatten(1)
        a_emb = self.action_embed(a_t.long())
        return h_s, a_emb, h_sn

    def _build_target(self, s_t: torch.Tensor, a_t: torch.Tensor, s_next: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            h_s, a_emb, h_sn = self._encode_parts(s_t, a_t, s_next)
        return torch.cat([h_s, a_emb, h_sn], dim=-1)

    def _decode_z(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_trunk(z)
        return torch.cat(
            [self.state_t_head(h), self.action_head(h), self.state_next_head(h)],
            dim=-1,
        )

    # ============================================================
    # Public API
    # ============================================================

    def encode(self, s_t: torch.Tensor, a_t: torch.Tensor, s_next: torch.Tensor):
        h_s, a_emb, h_sn = self._encode_parts(s_t, a_t, s_next)
        h = self.fc(torch.cat([h_s, a_emb, h_sn], dim=-1))

        mu = F.normalize(self.fc_mu(h), p=2, dim=-1)
        rho = torch.sigmoid(self.fc_rho(h))
        rho = self.cfg.rho_min + rho * (self.cfg.rho_max - self.cfg.rho_min)
        return mu, rho

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self._decode_z(z)

    def forward(self, s_t: torch.Tensor, a_t: torch.Tensor, s_next: torch.Tensor) -> SCVAEForwardOutput:
        mu, rho = self.encode(s_t, a_t, s_next)
        z = _sc_sample(mu, rho) if self.training else mu
        recon = self._decode_z(z)
        recon_target = self._build_target(s_t, a_t, s_next)
        return SCVAEForwardOutput(recon, mu, rho, z, recon_target)

    def loss(self, out: SCVAEForwardOutput) -> Tuple[torch.Tensor, torch.Tensor]:
        l_recon = F.mse_loss(out.recon, out.recon_target)
        l_kl = _sc_kl_uniform(
            out.rho,
            self.latent_dim,
            max_terms=self.cfg.kl_max_terms,
        ).mean()
        return l_recon, l_kl
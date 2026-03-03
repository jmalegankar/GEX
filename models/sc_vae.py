from __future__ import annotations

from typing import NamedTuple, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.obs_embeddings import ObservationEmbedding
from models.config import SCVAEConfig


# ===================================================================
# Spherical Cauchy helpers
# ===================================================================

_KL_SERIES_CACHE: dict = {}


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


def _get_kl_series_terms(
    dim: int,
    max_terms: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Cached (K,) tensor of  coeff_k * psi_diff_k  for k = 1..max_terms.
    Independent of rho and batch size, so safe to cache per (dim, device, dtype).
    """
    key = (dim, max_terms, str(device), str(dtype))

    if key in _KL_SERIES_CACHE:
        return _KL_SERIES_CACHE[key]

    d_minus_1 = float(dim - 1)
    a         = d_minus_1 / 2.0
    a_t       = torch.tensor(a, device=device, dtype=dtype)

    k = torch.arange(1, max_terms + 1, device=device, dtype=dtype)

    # log( (a)_k / k! )  via lgamma
    log_coeff = torch.lgamma(a_t + k) - torch.lgamma(a_t) - torch.lgamma(k + 1.0)
    coeff     = torch.exp(log_coeff)

    psi_base = torch.digamma(torch.tensor(d_minus_1, device=device, dtype=dtype))
    psi_diff = torch.digamma(d_minus_1 + k) - psi_base   # ψ(d-1+k) - ψ(d-1)

    scalar_terms = coeff * psi_diff   # (K,)

    _KL_SERIES_CACHE[key] = scalar_terms
    return scalar_terms


def _sc_kl_asymptotic(
    rho: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    """
    Proposition 2 (paper §3.2.2): asymptotic KL for rho → 1.

    KL ≈ (d-1) * log((1+ρ)/(1-ρ)) + ψ((d-1)/2) - ψ(d-1)

    Used when rho > RHO_ASYMP_THRESHOLD (= 0.9) because z(ρ) → 1
    causes the power series to converge too slowly.
    """
    device    = rho.device
    dtype     = rho.dtype
    d_minus_1 = float(dim - 1)

    log_term   = d_minus_1 * (torch.log1p(rho) - torch.log1p(-rho))  # (d-1)*log((1+ρ)/(1-ρ))
    correction = (
        torch.digamma(torch.tensor(d_minus_1 / 2.0, device=device, dtype=dtype))
        - torch.digamma(torch.tensor(d_minus_1,     device=device, dtype=dtype))
    )  # scalar

    return (log_term + correction).clamp_min(0.0)


# Threshold from paper: "for ρ > 0.9 we use the asymptotic approximation"
_RHO_ASYMP = 0.9


def _sc_kl_uniform(
    rho: torch.Tensor,
    dim: int,
    *,
    max_terms: int = 64,
) -> torch.Tensor:
    """
    KL( spCauchy_d(·|μ,ρ) ‖ Uniform(S^{d-1}) ) — Theorem 1.

    Vectorised over batch; series terms cached per (dim, device, dtype).
    For ρ > 0.9 falls back to Proposition 2 asymptotic to avoid slow
    convergence when z(ρ) → 1.

    Args:
        rho:       (B,) or (B,1) tensor, values in [0, 1)
        dim:       latent dimension d  (must be ≥ 2)
        max_terms: series truncation for the ρ ≤ 0.9 branch
    Returns:
        kl: (B,1) non-negative tensor
    """
    if dim < 2:
        raise ValueError("dim must be >= 2")

    rho = rho.to(dtype=torch.float32)
    if rho.dim() == 1:
        rho = rho.unsqueeze(-1)   # (B,1)

    eps = 1e-7
    rho = rho.clamp(0.0, 1.0 - eps)

    device    = rho.device
    dtype     = rho.dtype
    d_minus_1 = float(dim - 1)

    # ----------------------------------------------------------------
    # Masks for the two branches
    # ----------------------------------------------------------------
    high_rho = rho > _RHO_ASYMP   # (B,1) bool

    kl = torch.zeros_like(rho)

    # ----------------------------------------------------------------
    # Branch A — power series (ρ ≤ 0.9)
    # ----------------------------------------------------------------
    if high_rho.logical_not().any():
        rho_lo = rho.clone()
        rho_lo[high_rho] = 0.0   # dummy safe value; result masked out below

        log_ratio = torch.log1p(-rho_lo) - torch.log1p(rho_lo)
        term1     = d_minus_1 * log_ratio

        ratio = (1.0 - rho_lo) / (1.0 + rho_lo)
        pref  = d_minus_1 * ratio.pow(d_minus_1)

        z = 4.0 * rho_lo / (1.0 + rho_lo).pow(2)   # (B,1)

        scalar_terms = _get_kl_series_terms(dim, max_terms, device, dtype)  # (K,)
        k     = torch.arange(1, max_terms + 1, device=device, dtype=dtype)
        z_pow = z.pow(k)                             # (B,K)

        series = (z_pow * scalar_terms).sum(dim=-1, keepdim=True)  # (B,1)
        kl_lo  = (term1 + pref * series).clamp_min(0.0)

        kl = torch.where(high_rho, kl, kl_lo)

    # ----------------------------------------------------------------
    # Branch B — asymptotic (ρ > 0.9, Proposition 2)
    # ----------------------------------------------------------------
    if high_rho.any():
        rho_hi = rho.clone()
        rho_hi[~high_rho] = 0.5   # dummy safe value; result masked out below

        kl_hi = _sc_kl_asymptotic(rho_hi, dim)
        kl    = torch.where(high_rho, kl_hi, kl)

    return kl

# ==================================================================
# Uniformity Loss
# ==================================================================
def _uniformity_loss(mu: torch.Tensor, t: float = 2.0) -> torch.Tensor:
    """
    Wang & Isola (2020) uniformity loss on S^{d-1}.

    L_uniform = log E[exp(-t * ||mu_i - mu_j||^2)]  for i != j

    For unit vectors: ||a-b||^2 = 2(1 - a·b)
    Minimising this spreads mu's uniformly across the sphere.

    Args:
        mu: (B, d) unit vectors
        t:  kernel bandwidth (default 2.0 from paper)
    Returns:
        scalar (lower = more uniform)
    """
    sq_dists = 2.0 - 2.0 * (mu @ mu.T)   # (B, B)
    mask     = ~torch.eye(mu.size(0), dtype=torch.bool, device=mu.device)
    return torch.log(torch.exp(-t * sq_dists[mask]).mean())


# ===================================================================
# Output type
# ===================================================================

class SCVAEForwardOutput(NamedTuple):
    recon:        torch.Tensor
    mu:           torch.Tensor
    rho:          torch.Tensor
    z:            torch.Tensor
    recon_target: torch.Tensor


# ===================================================================
# Conv block
# ===================================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU(),
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
      embedding(obs) -> (B, C, H, W) float
      SCVAE never tries to infer obs layout.
    """

    def __init__(
        self,
        embedding: ObservationEmbedding,
        cfg: SCVAEConfig,
        sample_input_shape: Tuple[int, ...],
    ):
        super().__init__()
        self.embedding  = embedding
        self.cfg        = cfg
        self.latent_dim = cfg.latent_dim

        # conv encoder
        layers = []
        in_ch  = embedding.out_channels
        for ch in cfg.conv_channels:
            layers.append(ConvBlock(in_ch, ch))
            in_ch = ch
        self.conv = nn.Sequential(*layers)

        # infer conv feature dim
        with torch.no_grad():
            dummy          = torch.zeros((1, *sample_input_shape))
            dummy_feat     = self.conv(self._embed(dummy))
            self._feat_dim = dummy_feat.flatten(1).shape[1]

        # action embedding
        self.action_embed = nn.Embedding(cfg.n_actions, cfg.action_embed_dim)

        # bottleneck
        self.fc = nn.Sequential(
            nn.Linear(2 * self._feat_dim + cfg.action_embed_dim, cfg.hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu  = nn.Linear(cfg.hidden_dim, cfg.latent_dim)
        self.fc_rho = nn.Linear(cfg.hidden_dim, 1)

        # decoder
        self.decoder_trunk = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.hidden_dim), nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.ReLU(),
        )
        self.state_t_head    = nn.Linear(cfg.hidden_dim, self._feat_dim)
        self.action_head     = nn.Linear(cfg.hidden_dim, cfg.action_embed_dim)
        self.state_next_head = nn.Linear(cfg.hidden_dim, self._feat_dim)

    def _embed(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.embedding(obs)
        if x.dim() != 4:
            raise ValueError(f"embedding must return (B,C,H,W), got {tuple(x.shape)}")
        return x

    def _encode_parts(self, s_t, a_t, s_next):
        h_s  = self.conv(self._embed(s_t)).flatten(1)
        h_sn = self.conv(self._embed(s_next)).flatten(1)
        a_emb = self.action_embed(a_t.long())
        return h_s, a_emb, h_sn

    def _build_target(self, s_t, a_t, s_next) -> torch.Tensor:
        with torch.no_grad():
            h_s, a_emb, h_sn = self._encode_parts(s_t, a_t, s_next)
        return torch.cat([h_s, a_emb, h_sn], dim=-1)

    def _decode_z(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_trunk(z)
        return torch.cat(
            [self.state_t_head(h), self.action_head(h), self.state_next_head(h)],
            dim=-1,
        )

    def encode_state(self, s_t: torch.Tensor) -> torch.Tensor:
        """Returns flattened conv features for s_t. Used by policy."""
        with torch.no_grad():
            return self.conv(self._embed(s_t)).flatten(1)  # (B, feat_dim)

    def encode(self, s_t, a_t, s_next):
        h_s, a_emb, h_sn = self._encode_parts(s_t, a_t, s_next)
        h   = self.fc(torch.cat([h_s, a_emb, h_sn], dim=-1))
        mu  = F.normalize(self.fc_mu(h), p=2, dim=-1)
        rho = torch.sigmoid(self.fc_rho(h))
        rho = self.cfg.rho_min + rho * (self.cfg.rho_max - self.cfg.rho_min)
        return mu, rho

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self._decode_z(z)

    def forward(self, s_t, a_t, s_next) -> SCVAEForwardOutput:
        mu, rho      = self.encode(s_t, a_t, s_next)
        z            = _sc_sample(mu, rho) if self.training else mu
        recon        = self._decode_z(z)
        recon_target = self._build_target(s_t, a_t, s_next)
        return SCVAEForwardOutput(recon, mu, rho, z, recon_target)

    def loss(self, out: SCVAEForwardOutput) -> Tuple[torch.Tensor, torch.Tensor]:
        l_recon = F.mse_loss(out.recon, out.recon_target)
        l_kl    = _sc_kl_uniform(
            out.rho,
            self.latent_dim,
            max_terms=self.cfg.kl_max_terms,
        ).mean()
        l_uniform = _uniformity_loss(out.mu, t=self.cfg.uniformity_t)
        return l_recon, l_kl, l_uniform
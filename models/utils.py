import torch
import torch.nn.functional as F
import numpy as np

from typing import Tuple

# ===================================================================
# Spherical Cauchy helpers
# ===================================================================

@torch.jit.script
def _mobius_add(a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    a_sq = (a * a).sum(-1, keepdim=True)
    x_sq = (x * x).sum(-1, keepdim=True)
    ax   = (a * x).sum(-1, keepdim=True)
    num  = (1 + 2 * ax + x_sq) * a + (1 - a_sq) * x
    den  = 1 + 2 * ax + a_sq * x_sq
    return F.normalize(num / (den + 1e-8), p=2, dim=-1)


@torch.jit.script
def sc_sample(mu: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    xi = F.normalize(torch.randn_like(mu), p=2, dim=-1)
    return _mobius_add(rho * mu, xi)


@torch.jit.script
def _sc_kl_asymptotic(rho: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Proposition 2 (paper §3.2.2): KL ≈ (d-1)*log((1+ρ)/(1-ρ)) + ψ((d-1)/2) - ψ(d-1)
    Verified against paper eq. directly. Valid when ρ > 0.9 (z → 1).
    """
    d_minus_1  = float(dim - 1)
    # (d-1) * log((1+ρ)/(1-ρ)) — positive ✓
    log_term   = d_minus_1 * (torch.log1p(rho) - torch.log1p(-rho))
    # ψ((d-1)/2) - ψ(d-1) — negative but small relative to log_term ✓
    correction = (
        torch.digamma(torch.tensor(d_minus_1 / 2.0, device=rho.device, dtype=rho.dtype))
        - torch.digamma(torch.tensor(d_minus_1,     device=rho.device, dtype=rho.dtype))
    )
    return (log_term + correction).clamp_min(0.0)

def _gauss_legendre_01(n: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gauss-Legendre nodes and weights mapped to [0, 1].
    Cached per (n, device, dtype).
    """

    nodes_np, weights_np = np.polynomial.legendre.leggauss(n)
    # map [-1,1] -> [0,1]: t = (s+1)/2, w -> w/2
    t = torch.tensor((nodes_np + 1.0) / 2.0, device=device, dtype=dtype)
    w = torch.tensor(weights_np / 2.0,        device=device, dtype=dtype)
    return t, w

_LEGENDRE_POINTS: int = 64  # default for KL quadrature branch
_LEGENDRE_TENSORS = _gauss_legendre_01(_LEGENDRE_POINTS, torch.device('cpu'), torch.float32)  # cached on CPU, moved to target device in function


@torch.jit.script
def _sc_kl_quadrature(
    rho: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    """
    Proposition 1 (paper §3.2.2): quadrature-based KL for ρ ≤ 0.9.

    KL = (d-1)*log((1-ρ)/(1+ρ))
       + (d-1) * ∫_0^1 [t^{d-2}/(1-t)] * [1 - ((1-ρ)²/((1+ρ)²-4ρt))^{(d-1)/2}] dt

    The integrand has a removable singularity at t=1 (both numerator 1/(1-t)
    and the bracket → 0, limit is finite: 2(d-1)ρ/(1-ρ)²). Standard G-L
    handles this stably with enough points.

    This avoids the pref=((1-ρ)/(1+ρ))^{d-1} underflow in the power series
    branch, which collapses to 0 for dim≥16 at moderate ρ.
    """
    device, dtype = rho.device, rho.dtype
    d_minus_1     = float(dim - 1)
    alpha         = d_minus_1 / 2.0

    t_gl, w_gl = _LEGENDRE_TENSORS
    if t_gl.device != device or t_gl.dtype != dtype:
        t_gl = t_gl.to(device=device, dtype=dtype)
        w_gl = w_gl.to(device=device, dtype=dtype)

    # broadcast: rho (B,1), t (1,Q) -> (B,Q)
    t_gl = t_gl.unsqueeze(0)    # (1, Q)
    w_gl = w_gl.unsqueeze(0)    # (1, Q)

    # ((1-ρ)² / ((1+ρ)² - 4ρt))^α
    denom = (1.0 + rho).pow(2) - 4.0 * rho * t_gl   # (B, Q)
    denom = denom.clamp_min(1e-10)
    ratio = (1.0 - rho).pow(2) / denom               # (B, Q), in (0,1]
    inner = 1.0 - ratio.pow(alpha)                   # (B, Q), in [0,1)

    # t^{d-2} / (1-t): clamp near t=1 (singularity is removable, inner→0 there)
    t_pow        = t_gl.pow(max(d_minus_1 - 1.0, 0.0))   # (1, Q)
    one_minus_t  = (1.0 - t_gl).clamp_min(1e-10)          # (1, Q)
    integrand    = (t_pow / one_minus_t) * inner           # (B, Q)

    integral = (w_gl * integrand).sum(dim=-1, keepdim=True)  # (B, 1)

    # term1 is negative (log of value < 1); term2 is positive and larger
    term1 = d_minus_1 * (torch.log1p(-rho) - torch.log1p(rho))
    term2 = d_minus_1 * integral
    return (term1 + term2).clamp_min(0.0)


_RHO_ASYMP = 0.9

@torch.jit.script
def sc_kl_uniform(rho: torch.Tensor, dim: int) -> torch.Tensor:
    """
    KL( spCauchy_d(·|μ,ρ) ‖ Uniform(S^{d-1}) )

    Branches (matching paper §3.2.2):
      ρ ≤ 0.9  →  Proposition 1 (Gauss-Legendre quadrature) — stable for all d
      ρ >  0.9 →  Proposition 2 (asymptotic)

    Args:
        rho:           (B,) or (B,1) in [0, 1)
        dim:           latent dimension d ≥ 2
        n_quad_points: G-L quadrature points for the low-ρ branch
    Returns:
        kl: (B,1) non-negative tensor
    """
    if dim < 2:
        raise ValueError("dim must be >= 2")

    rho = rho.to(dtype=torch.float32)
    if rho.dim() == 1:
        rho = rho.unsqueeze(-1)
    rho = rho.clamp(0.0, 1.0 - 1e-7)

    high_rho = rho > _RHO_ASYMP   # (B,1)
    kl       = torch.zeros_like(rho)

    # ── Branch A: quadrature (ρ ≤ 0.9) ──────────────────────────
    if high_rho.logical_not().any():
        rho_lo            = rho.clone()
        rho_lo[high_rho]  = 0.5
        kl_lo             = _sc_kl_quadrature(rho_lo, dim)
        kl                = torch.where(high_rho, kl, kl_lo)

    # ── Branch B: asymptotic (ρ > 0.9) ───────────────────────────
    if high_rho.any():
        rho_hi = rho.clone(); rho_hi[~high_rho] = 0.5
        kl     = torch.where(high_rho, _sc_kl_asymptotic(rho_hi, dim), kl)

    return kl


@torch.jit.script
def uniformity_loss(mu: torch.Tensor, t: float = 2.0) -> torch.Tensor:
    sq_dists = 2.0 - 2.0 * (mu @ mu.T)
    mask     = ~torch.eye(mu.size(0), dtype=torch.bool, device=mu.device)
    return torch.log(torch.exp(-t * sq_dists[mask]).mean())
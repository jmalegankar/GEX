"""
Multichannel LMU — Gated Write variant (HiPPO-LegT backbone).

Gate types (controlled by gate_type arg to LMUCell):
─────────────────────────────────────────────────────
  'softsign_sum'   (default, recommended)
      gate  = softsign(u_x)          = u_x / (1 + |u_x|)
      innov = softsign(u_x − pred)   = (u_x−pred) / (1 + |u_x−pred|)
      write = W_pre(gate + innov) + pred

      Null condition 1 (u_x=0): gate=0, innov=softsign(-pred)
          → write = W_pre(softsign(-pred)) + pred  ≈ pred early training
          → NOT exactly pred. Holds approximately when ||pred|| is small
            (early training), breaks gracefully as pred grows.
      Null condition 2 (u_x=pred): gate=softsign(pred), innov=0
          → write = W_pre(softsign(pred)) + pred  ≠ pred (same as above)
      r_intr = ||gate + innov||₂  ∈ [0, 2√C]    (softsign ∈ (-1,1))

      Why use this: faster convergence than tanh_product (empirically,
      ~5.76M vs ~8.26M steps on S11 no-wrapper). Softsign's heavier tail
      (~20% more gradient at moderate activations vs tanh) reduces gradient
      vanishing. The approximate null conditions are fine because β=0.

  'tanh_product'   (original, both null conditions exact)
      gate  = tanh(u_x)
      innov = tanh(u_x − pred)
      write = W_pre(gate ⊙ innov) + pred

      Null condition 1 (u_x=0):    gate=0  → write=pred  ✓ exactly
      Null condition 2 (u_x=pred): innov=0 → write=pred  ✓ exactly
      r_intr = ||gate ⊙ innov||₂  ∈ [0, √C]    (tanh ∈ (-1,1))

      Use when exact null conditions matter (e.g. if β > 0 and r_intr is
      actively used as a reward signal — the exact conditions make r_intr
      a clean prediction-error metric in that case).

  'none'   (no gating — plain additive write, [OLD] baseline)
      write = u_x + u_h + u_m   (no W_pre, no gate, no innov)
      r_intr = ||u_x||₂          (raw encoder output magnitude)

      Use for ablation only. Fastest convergence (~4M steps on S11) but
      lower final performance ceiling (0.93 vs 0.96 for gated variants).
      Memory is always written regardless of prediction quality, which
      prevents m_norm collapse but loses the world-model signal.

Unified r_intr formula (works for all gate types):
    r_intr = ||u_actual − pred||₂
    For tanh_product:  = ||W_pre(gate ⊙ innov)||₂ = ||gate ⊙ innov||₂
    For softsign_sum:  = ||W_pre(gate + innov)||₂  = ||gate + innov||₂
    For none:          = ||u_x + pred − pred||₂     = ||u_x||₂
    The W_pre isometry (||Wv||=||v||) makes the first two clean.

m_norm anti-collapse residual:
    All gated variants suffer from m_norm collapse as E_h learns to predict
    u_x (innov→0, writes→pred-only, memory stagnates). Fixed by injecting
    a small fraction of u_x directly into the write, bypassing the gate:
        u_actual += residual_scale * u_x.detach()
    residual_scale=0.05 is the default. Set to 0.0 to disable (e.g. 'none'
    mode doesn't need it — always writes u_x directly).

OrthoLayer (W_pre) notes:
    - 'none' mode does not use W_pre at all (no-op for optimizer exclusion).
    - 'tanh_product' and 'softsign_sum' both require W_pre excluded from Adam.
    - See optimizer integration note in module docstring.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import cont2discrete
from typing import Literal, Tuple

from torch.nn.utils import spectral_norm


# ─────────────────────────────────────────────────────────────────────────────
# OrthoLayer
# ─────────────────────────────────────────────────────────────────────────────

class OrthoLayer(nn.Module):
    """
    Orthogonal linear layer (no bias) maintained via Cayley-map Riemannian updates.

    No bias is required: W_pre(0) = 0 exactly, which makes r_intr = ||u_actual - pred||
    equal to ||W_pre(innovation)||₂ = ||innovation||₂ (isometry). A bias would add
    a constant offset and break this equality.

    CRITICAL: exclude from main Adam optimizer. See module docstring.
    """

    def __init__(self, size: int):
        super().__init__()
        self.weights = nn.Parameter(torch.eye(size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weights   # (B, C) @ (C, C) → (B, C)

    def ortho_update(self, lr: float) -> None:
        """
        Riemannian gradient step on O(C) via Cayley retraction.
        Call AFTER loss.backward(), BEFORE optimizer.step().
        Zeros the gradient so Adam never touches this parameter.
        """
        with torch.no_grad():
            if self.weights.grad is None:
                return
            G = self.weights.grad
            W = self.weights
            A = G @ W.t() - W @ G.t()   # skew-symmetric Riemannian gradient
            I = torch.eye(W.size(0), device=W.device, dtype=W.dtype)
            W_new = torch.linalg.solve(I + lr * A, (I - lr * A) @ W)
            self.weights.copy_(W_new)
            self.weights.grad.zero_()

    @torch.no_grad()
    def reorthogonalize(self) -> None:
        """Hard SVD reset. Use when orthogonality_error() > 1e-3."""
        U, _, Vh = torch.linalg.svd(self.weights, full_matrices=False)
        self.weights.copy_(U @ Vh)

    @torch.no_grad()
    def orthogonality_error(self) -> float:
        """Diagnostic: ‖WᵀW − I‖_F. Should stay < 1e-3."""
        I = torch.eye(self.weights.size(0), device=self.weights.device)
        return (self.weights.t() @ self.weights - I).norm().item()


# ─────────────────────────────────────────────────────────────────────────────
# HiPPO-LegT matrices
# ─────────────────────────────────────────────────────────────────────────────

def get_AB(d: int, theta: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build HiPPO-LegT (A, B) from Voelker 2019 Eq. 2, then ZOH-discretize.
    Returns float32 arrays of shape (d, d) and (d, 1).
    """
    Q = np.arange(d, dtype=float)
    R = (2 * Q + 1)[:, None]
    j, i = np.meshgrid(Q, Q)
    A = R * np.where(i < j, -1.0, (-1.0) ** (i - j + 1))
    A /= theta
    B = R * ((-1.0) ** Q)[:, None]
    B /= theta
    C_dummy = np.zeros((1, d))
    D_dummy = np.zeros((1,))
    Ad, Bd, _, _, _ = cont2discrete((A, B, C_dummy, D_dummy), dt=1.0, method='zoh')
    return Ad.astype(np.float32), Bd.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# LMUCell
# ─────────────────────────────────────────────────────────────────────────────

GateType = Literal['softsign_sum', 'tanh_product', 'none']


class LMUCell(nn.Module):
    """
    One step of the multichannel LMU — HiPPO-LegT backbone.

    Args:
        input_size:      C — encoder output dimension
        hidden_size:     n — hidden state dimension
        memory_size:     d — Legendre polynomial order
        theta:           sliding window length (LegT)
        gate_type:       'softsign_sum' | 'tanh_product' | 'none'
        residual_scale:  anti-collapse floor for gated variants (0.0 = off)

    Returns (from forward):
        h_new   (B, n)    — new hidden state
        m_new   (B, d, C) — new Legendre memory
        r_intr  (B,)      — prediction error, IN GRAPH
        gate    (B, C)    — detached diagnostic
        innov   (B, C)    — detached diagnostic
        u_x     (B, C)    — detached diagnostic

    For gate_type='none': gate and innov are zeros_like(u_x) (no meaning).
    """

    def __init__(
        self,
        input_size:     int,
        hidden_size:    int,
        memory_size:    int,
        theta:          float,
        gate_type:      GateType = 'softsign_sum',
        residual_scale: float    = 0.05,
    ):
        super().__init__()
        self.input_size     = input_size
        self.hidden_size    = hidden_size
        self.memory_size    = memory_size
        self.gate_type      = gate_type
        self.residual_scale = residual_scale

        # ── LegT memory matrices ──────────────────────────────────────────
        Ad, Bd = get_AB(memory_size, theta)
        self.register_buffer('A', torch.from_numpy(Ad))   # (d, d)
        self.register_buffer('B', torch.from_numpy(Bd))   # (d, 1)

        # ── Encoding parameters ───────────────────────────────────────────
        self.e_x    = nn.Parameter(torch.empty(input_size))
        self.E_h    = spectral_norm(nn.Linear(hidden_size, input_size, bias=False))
        self.e_m    = nn.Parameter(torch.zeros(memory_size))

        # ── W_pre: only used for gated variants ───────────────────────────
        # Still created for 'none' mode so that optimizer exclusion logic in
        # policies.py doesn't need to be conditioned on gate_type.
        # For 'none', W_pre.forward is never called.
        self.W_pre = OrthoLayer(input_size)

        # ── Dynamic read head ─────────────────────────────────────────────
        self.W_query = nn.Linear(hidden_size, memory_size, bias=False)
        nn.init.orthogonal_(self.W_query.weight, gain=0.01)

        # ── Hidden state kernels ──────────────────────────────────────────
        self.W_x = nn.Linear(input_size,  hidden_size, bias=True)
        self.W_h = spectral_norm(nn.Linear(hidden_size, hidden_size, bias=False))
        self.W_m = nn.Linear(input_size,  hidden_size, bias=False)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.uniform_(self.e_x, -1.0, 1.0)
        nn.init.xavier_normal_(self.E_h.weight)
        # e_m stays zero: silent write at episode start
        for layer in (self.W_x, self.W_h, self.W_m):
            nn.init.xavier_normal_(layer.weight)
        nn.init.zeros_(self.W_x.bias)

    # ── Gate implementations ──────────────────────────────────────────────────

    @staticmethod
    def _softsign(x: torch.Tensor) -> torch.Tensor:
        return x / (1.0 + x.abs())

    def _compute_write(
        self,
        u_x:  torch.Tensor,   # (B, C)
        u_h:  torch.Tensor,   # (B, C)
        u_m:  torch.Tensor,   # (B, C)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute u_actual, pred, gate, innov for the configured gate_type.

        Returns:
            u_actual : (B, C) — value to write into Legendre memory
            pred     : (B, C) — prediction (u_h + u_m)
            gate     : (B, C) — gate activation (zeros for 'none')
            innov    : (B, C) — innovation activation (zeros for 'none')

        r_intr is computed by the caller as ||u_actual - pred||₂.
        """
        pred = u_h + u_m

        if self.gate_type == 'softsign_sum':
            gate  = self._softsign(u_x)
            innov = self._softsign(u_x - pred)
            innovation_vec = gate + innov                        # (B, C)
            u_actual = self.W_pre(innovation_vec) + pred        # (B, C)

        elif self.gate_type == 'tanh_product':
            gate  = torch.tanh(u_x)
            innov = torch.tanh(u_x - pred)
            innovation_vec = gate * innov                        # (B, C)
            u_actual = self.W_pre(innovation_vec) + pred        # (B, C)

        elif self.gate_type == 'none':
            # Plain additive write — no gating, no W_pre
            u_actual = u_x + pred                               # (B, C)
            gate  = torch.zeros_like(u_x)
            innov = torch.zeros_like(u_x)

        else:
            raise ValueError(f"Unknown gate_type '{self.gate_type}'. "
                             f"Choose from: softsign_sum, tanh_product, none")

        # Anti-collapse residual for gated variants.
        # Prevents m_norm from collapsing to zero as E_h learns to predict u_x.
        # Adds a small fraction of u_x that bypasses the gate.
        # No-op for 'none' (already writes u_x directly; residual_scale=0 default).
        if self.gate_type != 'none' and self.residual_scale > 0.0:
            u_actual = u_actual + self.residual_scale * u_x.detach()

        return u_actual, pred, gate, innov

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        x:      torch.Tensor,   # (B, C)
        h_prev: torch.Tensor,   # (B, n)
        m_prev: torch.Tensor,   # (B, d, C)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns: h_new, m_new, r_intr, gate, innov, u_x

        r_intr = ||u_actual - pred||₂  (unified formula for all gate types)
               ∈ [0, √C]  for tanh_product  (tanh ∈ (-1,1), W_pre isometry)
               ∈ [0, 2√C] for softsign_sum  (softsign ∈ (-1,1))
               = ||u_x||₂ for none          (raw encoder magnitude)

        gate, innov, u_x are .detach()-ed — for diagnostics only.
        r_intr stays in the compute graph (needed if β > 0).
        """

        # ── Step 1: encode input ──────────────────────────────────────────
        e_x_n = F.normalize(self.e_x, dim=0)                    # (C,) unit norm
        u_x   = x * e_x_n                                        # (B, C)
        u_h   = self.E_h(h_prev)                                 # (B, C)
        u_m   = torch.einsum('d,bdc->bc', self.e_m, m_prev)     # (B, C)

        # ── Step 2: gated write ───────────────────────────────────────────
        u_actual, pred, gate, innov = self._compute_write(u_x, u_h, u_m)

        # ── Step 3: intrinsic reward ──────────────────────────────────────
        # Unified: r_intr = ||u_actual - pred||₂
        # For tanh_product: = ||W_pre(gate⊙innov)||₂ = ||gate⊙innov||₂
        # For softsign_sum: = ||W_pre(gate+innov)||₂ = ||gate+innov||₂
        # For none:         = ||u_x||₂
        r_intr = (u_actual - pred).norm(dim=-1)                  # (B,)

        # ── Step 4: Legendre memory update ────────────────────────────────
        Am    = torch.einsum('ij,bjc->bic', self.A, m_prev)     # (B, d, C)
        Bu    = self.B * u_actual.unsqueeze(1)                   # (B, d, C)
        m_new = Am + Bu                                          # (B, d, C)

        # ── Step 5: dynamic read head ─────────────────────────────────────
        C_t = F.normalize(self.W_query(h_prev), dim=-1)         # (B, d)
        y   = torch.einsum('bd,bdc->bc', C_t, m_new)            # (B, C)

        # ── Step 6: hidden update ─────────────────────────────────────────
        h_new = torch.tanh(
            self.W_x(x) + self.W_h(h_prev) + self.W_m(y)
        )                                                        # (B, n)

        return h_new, m_new, r_intr, gate.detach(), innov.detach(), u_x.detach()

    def initial_state(
        self, n: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(n, self.hidden_size, device=device)
        m = torch.zeros(n, self.memory_size, self.input_size, device=device)
        return h, m


# ─────────────────────────────────────────────────────────────────────────────
# LMU — sequence wrapper
# ─────────────────────────────────────────────────────────────────────────────

class LMU(nn.Module):
    """
    Sequence wrapper around LMUCell.
    Input:  x (B, T, C)
    Output: out (B, T, n), r_intrs (B, T), final_state (h, m)
    """
    def __init__(
        self,
        input_size:     int,
        hidden_size:    int,
        memory_size:    int,
        theta:          float,
        gate_type:      GateType = 'softsign_sum',
        residual_scale: float    = 0.05,
    ):
        super().__init__()
        self.cell = LMUCell(
            input_size, hidden_size, memory_size, theta,
            gate_type=gate_type, residual_scale=residual_scale,
        )

    def forward(
        self,
        x:     torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T, _ = x.shape
        h, m = state if state is not None else self.cell.initial_state(B, x.device)
        outs, r_intrs = [], []
        for t in range(T):
            h, m, r, _, _, _ = self.cell(x[:, t], h, m)
            outs.append(h)
            r_intrs.append(r)
        return torch.stack(outs, dim=1), torch.stack(r_intrs, dim=1), (h, m)
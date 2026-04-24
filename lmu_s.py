"""
HiPPO-LegS Memory Cell for RL — Drop-in replacement for LMUCell (lmu_t.py).

Why LegS instead of LegT for Craftax:
───────────────────────────────────────
LegT (the LMU) integrates over a sliding window [t-θ, t]. Information from
step 0 decays as exp(-t/θ). For θ=200 and t=5000 (Craftax long episodes),
that's exp(-25) ≈ 1.4e-11 — gone. The recipe the agent learned at step 10 is
unrecoverable at step 5000.

LegS (Gu et al., HiPPO NeurIPS 2020) integrates over [0, t] with uniform
weight, rescaling at each step. The continuous ODE is:

    d/dt c(t) = -1/t · A · c(t) + 1/t · B · u(t)

where A is the lower-triangular LegS matrix (no θ hyperparameter). The key
property: dilating time by α maps t→αt and leaves the trajectory of c
invariant. This means LegS is timescale-robust — it works for episodes of any
length without θ to tune.

Discrete-time forward Euler recurrence (Gu et al. eq 4):

    m_t = (I - A/t) m_{t-1} + B/t · u_t
        = m_{t-1} + (1/t)(B · u_t - A · m_{t-1})

where t is the current step counter within the episode (1-indexed, reset to 1
at episode boundary). m_0 = 0 by convention.

The A matrix for LegS (Gu et al. 2020 Theorem 2):

    A_nk = { sqrt((2n+1)(2k+1))  if n > k
           { n+1                  if n == k
           { 0                    if n < k

B_n = sqrt(2n+1). A is lower triangular.

Gate types (same as lmu_t.py — identical interface):
─────────────────────────────────────────────────────
  'softsign_sum'   (default, recommended)
  'tanh_product'   (exact null conditions)
  'none'           (no gating — ablation only)

See lmu_t.py docstring for full gate type documentation.

API difference from LMUCell: forward() takes an extra argument t (B,) int.
The step counter is tracked externally and reset to 1 at episode boundaries.

Integration: change one import line in policies.py.
    from lmu_ppo.lmu_t import LMUCell          # LegT
    from lmu_ppo.lmu_s import LegSCell as LMUCell  # LegS
Plus threading t through forward calls. See legs-integration artifact.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Tuple

from torch.nn.utils import spectral_norm

# OrthoLayer and GateType are identical to lmu_t — import directly,
# no duplication.
from lmu_t import OrthoLayer, GateType


# ─────────────────────────────────────────────────────────────────────────────
# LegS A and B matrices
# ─────────────────────────────────────────────────────────────────────────────

def get_AB_legs(d: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct HiPPO-LegS (A, B) matrices. No θ parameter.

    A is upper triangular (transposed from LegT's lower triangular).
    B_n = sqrt(2n+1).

    Per Gu et al. 2020, Theorem 2. These are the CONTINUOUS-TIME matrices.
    The discrete-time recurrence uses them via:
        m_t = m_{t-1} - (A/t) m_{t-1} + (B/t) u_t
            = (I - A/t) m_{t-1} + (B/t) u_t

    Returns float32 arrays:
        A: (d, d) — upper triangular, NOT ZOH discretized (time-varying)
        B: (d, 1) — column vector
    """
    n = np.arange(d, dtype=np.float64)

    # A_nk = sqrt((2n+1)(2k+1)) for n > k, (n+1) for n==k, 0 for n < k
    # Note: upper triangular means row n depends on columns k < n
    # Wait — let me be precise from Gu et al. Theorem 2:
    # The CONTINUOUS matrix A has:
    #   A_nk = sqrt((2n+1)(2k+1))  for k < n   (lower triangular part)
    #   A_nn = n+1
    # So A is actually LOWER triangular (same convention as LegT).
    # B_n = sqrt(2n+1)
    # The discrete recurrence is:
    #   c_t = c_{t-1} - (1/t)(A c_{t-1} - B u_t)
    #       = (I - A/t) c_{t-1} + (B/t) u_t

    # Build A (lower triangular + diagonal)
    i, k = np.meshgrid(n, n, indexing='ij')  # i=row, k=col
    A = np.where(
        i > k,
        np.sqrt((2 * i + 1) * (2 * k + 1)),
        np.where(i == k, i + 1, 0.0)
    )  # (d, d) lower triangular + diagonal

    B = np.sqrt(2 * n + 1)[:, np.newaxis]  # (d, 1)

    return A.astype(np.float32), B.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# LegSCell
# ─────────────────────────────────────────────────────────────────────────────

class LegSCell(nn.Module):
    """
    One step of the multichannel HiPPO-LegS recurrent cell.

    State:
        h:    (B, hidden_size)    — nonlinear hidden state
        m:    (B, memory_size, C) — polynomial memory coefficients
        t:    (B,) int            — step counter within episode (caller-managed)

    The step counter is NOT stored in the cell. It is tracked by the caller
    (lmu_ppo.collect_rollouts) and passed in at each forward call. Reset to 1
    at episode boundaries. This avoids state management issues across envs.

    Args:
        input_size:      C — encoder output dimension
        hidden_size:     n — hidden state dimension
        memory_size:     d — Legendre polynomial order
        gate_type:       'softsign_sum' | 'tanh_product' | 'none'
        residual_scale:  anti-collapse floor for gated variants (0.0 = off)
        discretization:  'euler' (default) | 'bilinear' (more stable at t>1000)

    Returns (from forward): h_new, m_new, r_intr, gate, innov, u_x
    Identical return signature to LMUCell. Only forward() differs (adds t arg).
    """

    def __init__(
        self,
        input_size:     int,
        hidden_size:    int,
        memory_size:    int,
        gate_type:      GateType = 'softsign_sum',
        residual_scale: float    = 0.05,
        discretization: str      = 'euler',
    ):
        super().__init__()
        self.input_size     = input_size
        self.hidden_size    = hidden_size
        self.memory_size    = memory_size
        self.gate_type      = gate_type
        self.residual_scale = residual_scale
        self.discretization = discretization

        # LegS A, B — continuous-time, no ZOH. Time-varying discretization
        # is applied per-step in _legs_update.
        A, B = get_AB_legs(memory_size)
        self.register_buffer('A', torch.from_numpy(A))   # (d, d)
        self.register_buffer('B', torch.from_numpy(B))   # (d, 1)

        # ── Encoding (identical to LMUCell) ──────────────────────────────
        self.e_x    = nn.Parameter(torch.empty(input_size))
        self.E_h    = spectral_norm(nn.Linear(hidden_size, input_size, bias=False))
        self.e_m    = nn.Parameter(torch.zeros(memory_size))
        self.W_pre  = OrthoLayer(input_size)

        # ── Dynamic read head (identical to LMUCell) ─────────────────────
        self.W_query = nn.Linear(hidden_size, memory_size, bias=False)
        nn.init.orthogonal_(self.W_query.weight, gain=0.01)

        # ── Hidden state kernels (identical to LMUCell) ───────────────────
        self.W_x = nn.Linear(input_size,  hidden_size, bias=True)
        self.W_h = spectral_norm(nn.Linear(hidden_size, hidden_size, bias=False))
        self.W_m = nn.Linear(input_size,  hidden_size, bias=False)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.uniform_(self.e_x, -1.0, 1.0)
        nn.init.xavier_normal_(self.E_h.weight)
        for layer in (self.W_x, self.W_h, self.W_m):
            nn.init.xavier_normal_(layer.weight)
        nn.init.zeros_(self.W_x.bias)

    # ── Gate (identical logic to LMUCell._compute_write) ─────────────────────

    @staticmethod
    def _softsign(x: torch.Tensor) -> torch.Tensor:
        return x / (1.0 + x.abs())

    def _compute_write(
        self,
        u_x: torch.Tensor,
        u_h: torch.Tensor,
        u_m: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Identical to LMUCell._compute_write. Returns (u_actual, pred, gate, innov).
        r_intr = ||u_actual - pred||₂ computed by caller.
        """
        pred = u_h + u_m

        if self.gate_type == 'softsign_sum':
            gate  = self._softsign(u_x)
            innov = self._softsign(u_x - pred)
            u_actual = self.W_pre(gate + innov) + pred

        elif self.gate_type == 'tanh_product':
            gate  = torch.tanh(u_x)
            innov = torch.tanh(u_x - pred)
            u_actual = self.W_pre(gate * innov) + pred

        elif self.gate_type == 'none':
            u_actual = u_x + pred
            gate  = torch.zeros_like(u_x)
            innov = torch.zeros_like(u_x)

        else:
            raise ValueError(f"Unknown gate_type '{self.gate_type}'.")

        if self.gate_type != 'none' and self.residual_scale > 0.0:
            u_actual = u_actual + self.residual_scale * u_x.detach()

        return u_actual, pred, gate, innov

    # ── LegS memory update (the only part that differs from LMUCell) ──────────

    def _legs_update(
        self,
        m_prev:   torch.Tensor,   # (B, d, C)
        u_actual: torch.Tensor,   # (B, C)
        t:        torch.Tensor,   # (B,) — step counter, clamped >= 1
    ) -> torch.Tensor:
        """
        LegS discrete recurrence:

            Forward Euler:
                m_t = m_{t-1} + (1/t)(B · u_t  −  A · m_{t-1})

            Bilinear (Tustin, more stable for large t):
                (I + A/2t) m_t = (I − A/2t) m_{t-1} + (B/t) · u_t

        t: (B,) per-env step counter. Clamped to >= 1 to avoid div/0.
        As t grows, A/t → 0 and the memory updates more slowly — this is
        how LegS stretches its window to cover [0, t].

        For t=1 (first step): m_1 = m_0 + B·u_1 = B·u_1 (same as LegT).
        """
        t_safe = t.float().clamp(min=1.0)        # (B,)
        inv_t  = (1.0 / t_safe).view(-1, 1, 1)  # (B, 1, 1)

        Am = torch.einsum('ij,bjc->bic', self.A, m_prev)   # (B, d, C)
        Bu = self.B * u_actual.unsqueeze(1)                 # (B, d, C)

        if self.discretization == 'euler':
            # m_t = m_{t-1} + inv_t * (Bu - Am)
            return m_prev + inv_t * (Bu - Am)

        else:  # bilinear
            # Solve per-env: (I + A/2t) m_t = (I - A/2t) m_{t-1} + B/t · u
            # O(d³) per env per step — correct but expensive; use for t > 1000.
            B_size = m_prev.shape[0]
            d = self.memory_size
            I_d = torch.eye(d, device=m_prev.device, dtype=m_prev.dtype)
            m_new_list = []
            for b in range(B_size):
                half_At = self.A / (2.0 * t_safe[b])
                lhs = I_d + half_At
                rhs = ((I_d - half_At) @ m_prev[b]
                       + (self.B / t_safe[b]) * u_actual[b])
                m_new_list.append(torch.linalg.solve(lhs, rhs))
            return torch.stack(m_new_list, dim=0)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        x:      torch.Tensor,   # (B, C)
        h_prev: torch.Tensor,   # (B, hidden)
        m_prev: torch.Tensor,   # (B, d, C)
        t:      torch.Tensor,   # (B,) int/float — step within episode (1-indexed)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns: h_new, m_new, r_intr, gate, innov, u_x

        Identical return signature to LMUCell.forward.
        Only difference: requires t argument (per-env step counter).

        r_intr = ||u_actual - pred||₂  (unified, same as LMUCell)
        gate, innov, u_x are .detach()-ed diagnostics.
        r_intr stays in graph.

        t must be 1-indexed. Caller zeros h, m and resets t=1 at episode end.
        """

        # ── Step 1: encode ────────────────────────────────────────────────
        e_x_n = F.normalize(self.e_x, dim=0)                    # (C,)
        u_x   = x * e_x_n                                        # (B, C)
        u_h   = self.E_h(h_prev)                                 # (B, C)
        u_m   = torch.einsum('d,bdc->bc', self.e_m, m_prev)     # (B, C)

        # ── Step 2: gated write ───────────────────────────────────────────
        u_actual, pred, gate, innov = self._compute_write(u_x, u_h, u_m)

        # ── Step 3: intrinsic reward ──────────────────────────────────────
        r_intr = (u_actual - pred).norm(dim=-1)                  # (B,)

        # ── Step 4: LegS memory update (only difference from LMUCell) ────
        m_new = self._legs_update(m_prev, u_actual, t)           # (B, d, C)

        # ── Step 5: dynamic read head ─────────────────────────────────────
        C_t = F.normalize(self.W_query(h_prev), dim=-1)         # (B, d)
        y   = torch.einsum('bd,bdc->bc', C_t, m_new)            # (B, C)

        # ── Step 6: hidden update ─────────────────────────────────────────
        h_new = torch.tanh(
            self.W_x(x) + self.W_h(h_prev) + self.W_m(y)
        )                                                        # (B, hidden)

        return h_new, m_new, r_intr, gate.detach(), innov.detach(), u_x.detach()

    def initial_state(
        self, n: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Zero initial state. Matches LMUCell.initial_state signature."""
        h = torch.zeros(n, self.hidden_size, device=device)
        m = torch.zeros(n, self.memory_size, self.input_size, device=device)
        return h, m
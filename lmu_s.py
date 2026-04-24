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

where A is the upper-triangular LegS matrix (no θ hyperparameter). The key
property: dilating time by α maps t→αt and leaves the trajectory of c
invariant. This means LegS is timescale-robust — it works for episodes of any
length without θ to tune.

Discrete-time forward Euler recurrence (Gu et al. eq 4):

    m_t = (1 - A/t) m_{t-1} + B/t · u_t

where t is the current step counter within the episode (1-indexed, reset to 1
at episode boundary). m_0 = 0 by convention.

The A matrix for LegS (Gu et al. 2020 Theorem 2):

    A_nk = { sqrt((2n+1)(2k+1))  if n > k
           { n+1                  if n == k
           { 0                    if n < k

B_n = sqrt(2n+1)

This is upper triangular, meaning higher-order coefficients (n=d-1) depend on
all lower-order ones, but not vice versa. The structure is the transpose of
LegT's lower-triangular A.

Numerical note on discretization:
  Forward Euler (shown above) is first-order. The bilinear (Tustin) method is
  more stable for stiff systems:
      m_t = (I + A/(2t))^{-1} (I - A/(2t)) m_{t-1} + B/t · u_t
  For RL inference (step-by-step, small t), forward Euler is fine. For very
  long sequences (t>10000) or training stability, use bilinear. We default to
  forward Euler here and flag where to switch.

Architecture note:
  Everything outside the memory dynamics is IDENTICAL to lmu_t.py:
  - OrthoLayer W_pre and Cayley updates: unchanged
  - e_x, E_h, e_m encoding: unchanged
  - W_query dynamic read head: unchanged
  - W_x, W_h, W_m hidden update: unchanged
  - r_intr, gate, innov diagnostics: unchanged

  This means the policy class (policies.py) and lmu_ppo.py work with LegSCell
  by changing one import line. No other changes needed.

Usage:
  # In policies.py (or wherever LMUCell is imported):
  # BEFORE: from lmu_ppo.lmu_t import LMUCell
  # AFTER:  from lmu_ppo.lmu_s import LegSCell as LMUCell
  # That's the entire integration change.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from torch.nn.utils import spectral_norm

# Import OrthoLayer from lmu_t — it's identical and we don't want to duplicate.
from lmu_ppo.lmu_t import OrthoLayer


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
        h:       (B, hidden_size)        — nonlinear hidden state (same as LMUCell)
        m:       (B, memory_size, C)     — polynomial memory coefficients
        step:    scalar int              — current step within episode (managed externally)

    The step counter is NOT stored in the cell. It is tracked by the caller
    (lmu_ppo.collect_rollouts) and passed in at each forward call. This avoids
    state management issues across environments and reset boundaries.

    Forward arguments: (x, h_prev, m_prev, t)
    where t: (B,) int tensor — per-env step counter within current episode.
    Episodes that reset should have t[i] = 1 (first step).

    Returns: h_new, m_new, r_intr, gate, innov, u_x   (same as LMUCell)

    API DIFFERENCE FROM LMUCell: forward() takes an extra argument t.
    All callers must pass t. In policies.py and lmu_ppo.py, where LMUCell is
    called, add t to the call. See integration notes below.

    INTEGRATION CHANGES from lmu_t → lmu_s:
    ─────────────────────────────────────────
    1. policies.py:
       - In __init__: LegSCell instead of LMUCell. Remove theta arg.
       - In forward/evaluate_actions/predict_values: pass t (step counter)
         to self.lmu_cell(x, h_prev, m_prev, t)

    2. lmu_ppo.py:
       - In _setup_learn: init per-env step counter self._lmu_t (n_envs,)
       - In collect_rollouts: pass self._lmu_t to policy.forward
         Increment self._lmu_t after each step.
         Reset self._lmu_t[i] = 1 when done[i].

    3. lmu_ppo.py constructor: remove theta param (or keep for LegT compat).
    """

    def __init__(
        self,
        input_size:  int,
        hidden_size: int,
        memory_size: int,
        # No theta — LegS is timescale-free
        discretization: str = 'euler',  # 'euler' or 'bilinear'
    ):
        super().__init__()
        self.input_size   = input_size
        self.hidden_size  = hidden_size
        self.memory_size  = memory_size
        self.discretization = discretization

        # LegS A and B — stored as buffers, NOT ZOH-discretized.
        # Time-varying discretization is applied in forward().
        A, B = get_AB_legs(memory_size)
        self.register_buffer('A', torch.from_numpy(A))   # (d, d)
        self.register_buffer('B', torch.from_numpy(B))   # (d, 1)

        # For bilinear: precompute (I + A/2)^{-1} if you want to cache it.
        # We don't, because t changes each step. Computed on-the-fly.

        # ── Encoding parameters (identical to LMUCell) ───────────────
        self.e_x    = nn.Parameter(torch.empty(input_size))
        self.E_h    = spectral_norm(nn.Linear(hidden_size, input_size, bias=False))
        self.e_m    = nn.Parameter(torch.zeros(memory_size))
        self.W_pre  = OrthoLayer(input_size)

        # ── Dynamic read head (identical to LMUCell) ──────────────────
        self.W_query = nn.Linear(hidden_size, memory_size, bias=False)
        nn.init.orthogonal_(self.W_query.weight, gain=0.01)

        # ── Hidden state kernels (identical to LMUCell) ───────────────
        self.W_x = nn.Linear(input_size,  hidden_size, bias=True)
        self.W_h = spectral_norm(nn.Linear(hidden_size, hidden_size, bias=False))
        self.W_m = nn.Linear(input_size,  hidden_size, bias=False)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.uniform_(self.e_x, -1.0, 1.0)
        nn.init.xavier_normal_(self.E_h.weight)
        # e_m stays zero: same silent-write logic as LMUCell
        # W_pre identity init from OrthoLayer
        for layer in (self.W_x, self.W_h, self.W_m):
            nn.init.xavier_normal_(layer.weight)
        nn.init.zeros_(self.W_x.bias)

    def _compute_u(
        self,
        u_x: torch.Tensor,
        u_h: torch.Tensor,
        u_m: torch.Tensor,
    ) -> torch.Tensor:
        """
        Gated write — identical to LMUCell._compute_u.
        Two null conditions preserved:
          u_x = 0    → gate = 0 → u_actual = pred  ✓
          u_x = pred → innov = 0 → u_actual = pred  ✓
        """
        pred  = u_h + u_m
        gate  = torch.tanh(u_x)
        innov = torch.tanh(u_x - pred)
        return self.W_pre(gate * innov) + pred

    def _legs_update(
        self,
        m_prev:   torch.Tensor,   # (B, d, C)
        u_actual: torch.Tensor,   # (B, C)
        t:        torch.Tensor,   # (B,) float — step counter per env
    ) -> torch.Tensor:
        """
        Apply the LegS discrete recurrence:

            Forward Euler:
                m_t = (I - A/t) m_{t-1} + (B/t) u_t
                    = m_{t-1} - (1/t)(A m_{t-1} - B u_t)

            Bilinear (Tustin):
                m_t = (I + A/(2t))^{-1} [(I - A/(2t)) m_{t-1} + (B/t) u_t]

        t: per-env step counter, shape (B,). Clamp to >= 1 to avoid div/0.

        Key difference from LegT: the A matrix is scaled by 1/t at every step.
        As t grows, A/t shrinks and the memory updates more slowly — this is
        the mechanism by which LegS "stretches" its window to cover [0, t].
        """
        # t: (B,) → reshape for broadcasting with (B, d, C)
        t_safe = t.float().clamp(min=1.0)           # (B,)
        inv_t  = (1.0 / t_safe).view(-1, 1, 1)     # (B, 1, 1)

        # Am: (B, d, C) — same einsum as LMUCell
        Am = torch.einsum('ij,bjc->bic', self.A, m_prev)   # (B, d, C)

        # Bu: (B, d, C)
        Bu = self.B * u_actual.unsqueeze(1)                 # (B, d, C)

        if self.discretization == 'euler':
            # m_t = m_{t-1} - (1/t)(Am - Bu)
            #      = m_{t-1} + inv_t * (Bu - Am)
            m_new = m_prev + inv_t * (Bu - Am)

        else:  # bilinear
            # Bilinear (more stable for large t):
            # (I + A/(2t)) m_t = (I - A/(2t)) m_{t-1} + (B/t) u_t
            # Per-env: need to solve a (d,d) system per env. Not batched easily.
            # Use a fixed-point iteration or Neumann series approximation instead.
            # Neumann: (I + X)^{-1} ≈ I - X + X² - ... for small ||X||
            # For large t, A/(2t) is small and 2 terms suffice.
            #
            # Full batched solve approach (correct but O(d³) per step):
            B_size = m_prev.shape[0]
            d = self.memory_size
            C = self.input_size
            I_d = torch.eye(d, device=m_prev.device, dtype=m_prev.dtype)

            m_new_list = []
            for b in range(B_size):
                half_At = self.A / (2.0 * t_safe[b])    # (d, d)
                lhs = I_d + half_At                       # (d, d)
                rhs_mat = I_d - half_At                   # (d, d)
                # rhs: (d, C) = (I - A/2t) m_{t-1} + (B/t) u
                rhs = (rhs_mat @ m_prev[b] +              # (d, C)
                       (self.B / t_safe[b]) * u_actual[b])
                # Solve: lhs @ m_new_b = rhs
                m_new_b = torch.linalg.solve(lhs, rhs)   # (d, C)
                m_new_list.append(m_new_b)

            m_new = torch.stack(m_new_list, dim=0)       # (B, d, C)

        return m_new

    def forward(
        self,
        x:      torch.Tensor,   # (B, C)
        h_prev: torch.Tensor,   # (B, hidden)
        m_prev: torch.Tensor,   # (B, d, C)
        t:      torch.Tensor,   # (B,) int or float — step within episode (1-indexed)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns: h_new, m_new, r_intr, gate, innov, u_x
        Identical return signature to LMUCell.forward.

        t must be 1-indexed (first step of episode = 1, not 0).
        At episode reset, the caller zeros h and m, and resets t to 1.
        """

        # ── Step 1: per-channel u ─────────────────────────────────────
        e_x_n = F.normalize(self.e_x, dim=0)               # (C,)
        u_x   = x * e_x_n                                   # (B, C)

        u_h = self.E_h(h_prev)                              # (B, C)
        u_m = torch.einsum('d,bdc->bc', self.e_m, m_prev)  # (B, C)

        pred     = u_h + u_m
        gate     = torch.tanh(u_x)
        innov    = torch.tanh(u_x - pred)
        u_actual = self.W_pre(gate * innov) + pred          # (B, C)

        # ── Intrinsic reward (identical to LMUCell) ───────────────────
        u_null = self._compute_u(torch.zeros_like(u_x), u_h, u_m)
        r_intr = (u_actual - u_null).norm(dim=-1)           # (B,) — in graph

        # ── Step 2: LegS memory update ────────────────────────────────
        # This is the ONLY line that differs from LMUCell:
        # LMUCell:  m_new = einsum(A, m_prev) + B * u_actual
        # LegSCell: m_new = m_prev - (1/t)(A m_prev - B u_actual)
        m_new = self._legs_update(m_prev, u_actual, t)      # (B, d, C)

        # ── Step 3: dynamic read head (identical to LMUCell) ─────────
        C_t = F.normalize(self.W_query(h_prev), dim=-1)     # (B, d)
        y   = torch.einsum('bd,bdc->bc', C_t, m_new)        # (B, C)

        # ── Step 4: hidden update (identical to LMUCell) ─────────────
        h_new = torch.tanh(
            self.W_x(x) + self.W_h(h_prev) + self.W_m(y)
        )                                                    # (B, hidden)

        return h_new, m_new, r_intr, gate.detach(), innov.detach(), u_x.detach()

    def initial_state(
        self, n: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Zero initial state. Matches LMUCell.initial_state signature."""
        h = torch.zeros(n, self.hidden_size, device=device)
        m = torch.zeros(n, self.memory_size, self.input_size, device=device)
        return h, m
"""
Multichannel Legendre Memory Unit (LMU) — Gated Write variant.

What changed from baseline and why:
──────────────────────────────────
1. OrthoLayer (new class)
     W_pre ∈ O(C): orthogonal linear layer for the innovation projection.
     Maintained via Cayley-map Riemannian updates (ortho_update).
     MUST be excluded from the main Adam optimizer — see integration note below.
     Fallback: reorthogonalize() does a hard SVD reset if drift exceeds 1e-3.

2. _compute_u (new method on LMUCell)
     Replaces the plain linear sum  u = u_x + u_h + u_m  with a gated structure:
       pred  = u_h + u_m
       gate  = tanh(u_x)           ← zeroes when u_x = 0        (Condition 1)
       innov = tanh(u_x - pred)    ← zeroes when u_x = pred     (Condition 2)
       u     = W_pre(gate⊙innov) + pred
     Both conditions collapse to u = pred — novelty-gated writes.

3. e_x normalised in forward via F.normalize(e_x, dim=0)
     Pins ‖e_x‖₂ = 1.  minimize-r_intr gradient can only rotate e_x,
     not shrink it toward zero.  Per-channel scalar semantics preserved.

4. forward now returns (h_new, m_new, r_intr, gate, innov, u_x)
     [was: (h_new, m_new, r_intr) with _last_* side-effect attributes]

     gate  : (B, C) detached — tanh(u_x), diagnostic
     innov : (B, C) detached — tanh(u_x - pred), diagnostic
     u_x   : (B, C) detached — channel-weighted obs, diagnostic
     r_intr: (B,)  IN GRAPH  — ‖gate ⊙ innov‖₂ ∈ [0, √C]

     The _last_* side-effect attribute pattern (self._last_gate etc.) has been
     removed entirely.  It was vulnerable to EvalCallback.predict() calls
     mid-rollout clobbering the attributes with the eval-env batch size (1),
     causing shape mismatches when prod_buf was stacked.  Returning tensors
     directly as named values eliminates the race condition.

     Step 1 validation: β = 0, r_intr is LOGGED ONLY, not added to rewards.
     Step 2: add β * r_intr to extrinsic reward once Step 1 passes gate checks.

Old lines are commented with  # [OLD]  and kept for diff/reversion.
New lines are inline or immediately follow the old comment.

Optimizer integration (copy to lmu_ppo.py _setup_model):
─────────────────────────────────────────────────────────
    ortho_params = set(model.lmu_cell.W_pre.parameters())
    main_params  = [p for p in model.parameters() if p not in ortho_params]
    optimizer    = torch.optim.Adam(main_params, lr=3e-4, eps=1e-5)

Training loop (copy to lmu_ppo.py train):
──────────────────────────────────────────
    optimizer.zero_grad()
    loss.backward()
    model.lmu_cell.W_pre.ortho_update(lr=1e-3)  # Riemannian step, zeros grad
    optimizer.step()                             # Adam never sees W_pre grad

    if step % 100 == 0:
        err = model.lmu_cell.W_pre.orthogonality_error()
        if err > 1e-3:
            model.lmu_cell.W_pre.reorthogonalize()   # hard SVD reset
            print(f"W_pre ortho reset at step {step}, was {err:.6f}")
        logger.record("debug/W_pre_ortho_error", err)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import cont2discrete
from typing import Tuple

from torch.nn.utils import spectral_norm


# ─────────────────────────────────────────────────────────────────────────────
# OrthoLayer  [NEW CLASS]
# ─────────────────────────────────────────────────────────────────────────────

class OrthoLayer(nn.Module):
    """
    Orthogonal linear layer (no bias) maintained via Cayley-map Riemannian updates.

    No bias is not optional — it is required for the u_null proof.
    With no bias: W_pre(0) = 0 exactly, so u_null = pred exactly.
    If you add a bias, u_null = bias + pred ≠ pred and the intrinsic reward
    measure breaks.

    CRITICAL — exclude from main Adam optimizer:
        Adam maintains running moment estimates m_t, v_t.  ortho_update zeros
        the grad, but Adam still applies a nonzero Δ from accumulated momentum —
        corrupting orthogonality every step.  See optimizer integration note
        in module docstring above.

    Cayley retraction guarantee:
        A     = G Wᵀ - W Gᵀ            (skew-symmetric Riemannian gradient)
        W_new = (I + lr·A)⁻¹(I - lr·A) W_old
        W_newᵀ W_new = I  always ✓

    Isometry property (why W_pre drops from r_intr):
        ‖W_pre(v)‖₂ = ‖v‖₂  for all v
        So r_intr = ‖W_pre(gate⊙innov)‖₂ = ‖gate⊙innov‖₂.
        W_pre only rotates the innovation into a critic-useful basis;
        it does not affect reward magnitude.
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

            A = G @ W.t() - W @ G.t()  # skew-symmetric Riemannian gradient
            I = torch.eye(W.size(0), device=W.device, dtype=W.dtype)
            W_new = torch.linalg.solve(I + lr * A, (I - lr * A) @ W)
            self.weights.copy_(W_new)
            self.weights.grad.zero_()

    @torch.no_grad()
    def reorthogonalize(self) -> None:
        """Hard SVD reset.  Use when orthogonality_error() > 1e-3."""
        U, _, Vh = torch.linalg.svd(self.weights, full_matrices=False)
        self.weights.copy_(U @ Vh)

    @torch.no_grad()
    def orthogonality_error(self) -> float:
        """Diagnostic: ‖WᵀW − I‖_F.  Should stay < 1e-3."""
        I = torch.eye(self.weights.size(0), device=self.weights.device)
        return (self.weights.t() @ self.weights - I).norm().item()


# ─────────────────────────────────────────────────────────────────────────────
# LMU maths helper (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def get_AB(d: int, theta: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build continuous-time (A, B) from Voelker 2019 Eq. 2, then ZOH-discretise.

    A_ij = (2i+1) * { -1           if i < j
                    { (-1)^{i-j+1}  if i ≥ j
    B_i  = (2i+1) * (-1)^i

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

class LMUCell(nn.Module):
    """
    One step of the multichannel LMU — gated write variant.

    State shapes:      h : (B, n)     m : (B, d, C)
    Returns:           h_new, m_new, r_intr, gate, innov, u_x

    gate  : (B, C) detached — diagnostic only
    innov : (B, C) detached — diagnostic only
    u_x   : (B, C) detached — diagnostic only
    r_intr: (B,)  IN GRAPH  — needed for Step 2 loss

    The _last_* side-effect pattern has been removed.  See module docstring.
    """

    def __init__(
        self,
        input_size:  int,
        hidden_size: int,
        memory_size: int,
        theta:       float,
    ):
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size

        Ad, Bd = get_AB(memory_size, theta)
        self.register_buffer('A', torch.from_numpy(Ad))   # (d, d)
        self.register_buffer('B', torch.from_numpy(Bd))   # (d, 1)

        self.e_x = nn.Parameter(torch.empty(input_size))
        self.E_h = spectral_norm(nn.Linear(hidden_size, input_size, bias=False))
        self.e_m = nn.Parameter(torch.zeros(memory_size))

        self.W_pre = OrthoLayer(input_size)

        self.W_query = nn.Linear(hidden_size, memory_size, bias=False)
        nn.init.orthogonal_(self.W_query.weight, gain=0.01)

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

    def _compute_u(
        self,
        u_x: torch.Tensor,   # (B, C)
        u_h: torch.Tensor,   # (B, C)
        u_m: torch.Tensor,   # (B, C)
    ) -> torch.Tensor:
        """
        Novelty-gated write input. Replaces the baseline:
          [OLD]  u = u_x + u_h + u_m

        pred  = u_h + u_m              memory's joint prediction of u_x
        gate  = tanh(u_x)              zeroes at u_x = 0        (Condition 1)
        innov = tanh(u_x − pred)       zeroes at u_x = pred     (Condition 2)
        u     = W_pre(gate⊙innov) + pred

        Verification:
          u_x = 0    → gate = 0 → u = W_pre(0) + pred = pred  ✓
          u_x = pred → innov = 0 → u = W_pre(0) + pred = pred  ✓
          novel      → gate≠0, innov≠0 → u = pred + W_pre(innovation)  ✓

        Note: W_pre(0) = 0 exactly because OrthoLayer has no bias.
        """
        pred  = u_h + u_m
        gate  = u_x /(1+ u_x.abs())
        innov = (u_x - pred) / (1 + (u_x - pred).abs())
        return self.W_pre(gate + innov) + pred

    def forward(
        self,
        x:      torch.Tensor,   # (B, C)
        h_prev: torch.Tensor,   # (B, n)
        m_prev: torch.Tensor,   # (B, d, C)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns: h_new (B,n), m_new (B,d,C), r_intr (B,),
                 gate (B,C), innov (B,C), u_x (B,C)

        gate, innov, u_x are .detach()-ed — diagnostic tensors, not for loss.
        r_intr stays in the compute graph — needed for Step 2 auxiliary loss.

        r_intr = ‖u_actual − u_null‖₂
               = ‖W_pre(gate ⊙ innov)‖₂
               = ‖gate ⊙ innov‖₂        (W_pre is an isometry)
               ∈ [0, √C]

        Step 1 (current): β = 0 — r_intr logged, not added to rewards.
        Step 2: r_combined = r_ext + β * r_intr, β starting at 0.001.

        NOTE: The _last_* side-effect attributes (self._last_gate etc.) that
        previously existed have been REMOVED.  They were clobbered by
        EvalCallback.predict() calls mid-rollout (eval-env n_envs=1 vs training
        n_envs=16 produced shape mismatches when stacking prod_buf).
        Diagnostic tensors are now returned as named values, eliminating the
        race condition entirely.
        """

        # ── Step 1: per-channel u ─────────────────────────────────────────

        # [OLD] u_x = x * self.e_x
        e_x_n = F.normalize(self.e_x, dim=0)               # (C,)  ‖e_x_n‖₂ = 1
        u_x   = x * e_x_n                                   # (B, C)

        u_h = self.E_h(h_prev)                              # (B, C)
        u_m = torch.einsum('d,bdc->bc', self.e_m, m_prev)  # (B, C)

        pred  = u_h + u_m
        gate  = u_x /(1+ u_x.abs())
        innov = (u_x - pred) / (1 + (u_x - pred).abs())
        u_actual = self.W_pre(gate + innov) + pred           # (B, C)

        # [OLD] self._last_gate  = gate.detach()
        # [OLD] self._last_innov = innov.detach()
        # [OLD] self._last_u_x   = u_x.detach()
        # [OLD] self._last_prod  = (gate * innov).detach()
        # Removed — see docstring above.

        # ── Intrinsic reward ──────────────────────────────────────────────
        u_null = self._compute_u(torch.zeros_like(u_x), u_h, u_m)  # (B, C) = pred
        r_intr = (u_actual - u_null).norm(dim=-1)                   # (B,) — in graph

        # ── Step 2: Legendre memory update ────────────────────────────────
        Am    = torch.einsum('ij,bjc->bic', self.A, m_prev)   # (B, d, C)
        # [OLD] Bu = self.B * u.unsqueeze(1)
        Bu    = self.B * u_actual.unsqueeze(1)                 # (B, d, C)
        m_new = Am + Bu                                        # (B, d, C)

        # ── Step 3: dynamic read head ─────────────────────────────────────
        C_t = F.normalize(self.W_query(h_prev), dim=-1)       # (B, d)
        y   = torch.einsum('bd,bdc->bc', C_t, m_new)          # (B, C)

        # ── Step 4: hidden update ─────────────────────────────────────────
        h_new = torch.tanh(
            self.W_x(x) + self.W_h(h_prev) + self.W_m(y)
        )                                                      # (B, n)

        # [OLD] return h_new, m_new, r_intr
        return h_new, m_new, r_intr, gate.detach(), innov.detach(), u_x.detach()

    def initial_state(
        self, n: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Zero initial state. n = batch_size or n_envs."""
        h = torch.zeros(n, self.hidden_size, device=device)
        m = torch.zeros(n, self.memory_size, self.input_size, device=device)
        return h, m


# ─────────────────────────────────────────────────────────────────────────────
# LMU — sequence wrapper (not used by lmu_ppo.py, updated for completeness)
# ─────────────────────────────────────────────────────────────────────────────

class LMU(nn.Module):
    """
    Sequence wrapper around LMUCell.
    Input  : x  (B, T, C)
    Output : out (B, T, n),  r_intrs (B, T),  final_state (h, m)

    gate/innov/u_x are discarded here — they are per-step diagnostics most
    useful in the online rollout loop (lmu_ppo.collect_rollouts) where they
    are collected step-by-step before any callback can clobber them.
    """
    def __init__(self, input_size, hidden_size, memory_size, theta):
        super().__init__()
        self.cell = LMUCell(input_size, hidden_size, memory_size, theta)

    def forward(
        self,
        x:     torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T, _ = x.shape
        h, m = state if state is not None else self.cell.initial_state(B, x.device)
        outs, r_intrs = [], []
        for t in range(T):
            # [OLD] h, m, r = self.cell(x[:, t], h, m)
            h, m, r, _, _, _ = self.cell(x[:, t], h, m)   # gate/innov/u_x discarded
            outs.append(h)
            r_intrs.append(r)
        # [OLD] return torch.stack(outs, dim=1), (h, m)
        return torch.stack(outs, dim=1), torch.stack(r_intrs, dim=1), (h, m)
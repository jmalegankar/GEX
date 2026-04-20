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

4. forward now returns (h_new, m_new, r_intr)   [was: (h_new, m_new)]
     r_intr = ‖gate ⊙ innov‖₂ ∈ [0, √C]  (W_pre isometry; drops from magnitude).
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
import torch.nn.functional as F                  # [NEW] needed for normalize
from scipy.signal import cont2discrete           # unchanged — do NOT alias
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
        # Identity init: valid starting point on O(C).
        # At init, W_pre(v) = v, so the gated write is just innov + pred.
        # Learning rotates away from identity as the critic finds a better basis.
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
            G = self.weights.grad       # (C, C) Euclidean gradient
            W = self.weights

            # Riemannian gradient: project G onto tangent space of O(C) at W.
            # A = G Wᵀ - W Gᵀ  is skew-symmetric (Aᵀ = −A).
            A = G @ W.t() - W @ G.t()  # (C, C)

            I = torch.eye(W.size(0), device=W.device, dtype=W.dtype)

            # Cayley retraction.
            # torch.linalg.solve(X, B) = X⁻¹B — numerically stabler than .inv()
            W_new = torch.linalg.solve(I + lr * A, (I - lr * A) @ W)
            self.weights.copy_(W_new)
            self.weights.grad.zero_()   # Adam must not see this gradient

    @torch.no_grad()
    def reorthogonalize(self) -> None:
        """
        Hard SVD reset.  Use as a fallback when orthogonality_error() > 1e-3.
        The Cayley map can accumulate floating-point drift over many steps;
        this resets exactly to the nearest orthogonal matrix.
        """
        U, _, Vh = torch.linalg.svd(self.weights, full_matrices=False)
        self.weights.copy_(U @ Vh)

    @torch.no_grad()
    def orthogonality_error(self) -> float:
        """
        Diagnostic: ‖WᵀW − I‖_F.  Should stay < 1e-3.
        If it exceeds 1e-3, call reorthogonalize() and reduce the Cayley lr.
        """
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
    Returns:           h_new, m_new, r_intr
                       r_intr : (B,)  — Step 1: log only (β=0, not added to rewards)

    See module docstring for optimizer and training loop integration.
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

        # ── Fixed memory matrices (frozen) ────────────────────────────────
        Ad, Bd = get_AB(memory_size, theta)
        self.register_buffer('A', torch.from_numpy(Ad))   # (d, d)
        self.register_buffer('B', torch.from_numpy(Bd))   # (d, 1)

        # ── Encoding parameters ───────────────────────────────────────────
        self.e_x = nn.Parameter(torch.empty(input_size))
        # [NEW] e_x is still (C,) but is normalised to unit sphere in forward
        #       via F.normalize(self.e_x, dim=0).
        #       This prevents the minimize-r_intr gradient from collapsing e_x→0
        #       (it can only rotate the per-channel weighting, not shrink it).
        #       Per-channel scalar semantics are preserved — some channels can
        #       still dominate others; the constraint is ‖e_x‖₂ = 1, not e_x = const.

        self.E_h = spectral_norm(nn.Linear(hidden_size, input_size, bias=False))
        self.e_m = nn.Parameter(torch.zeros(memory_size))

        # ── Gated write  [NEW] ────────────────────────────────────────────
        # W_pre ∈ O(C): projects innovation into a critic-useful basis.
        # No bias — required for u_null = pred proof (W_pre(0) = 0 exactly).
        # MUST be excluded from main Adam optimizer (see module docstring).
        self.W_pre = OrthoLayer(input_size)

        # ── Memory readout (unchanged) ────────────────────────────────────
        # [OLD] self.C_proj = nn.Parameter(torch.empty(memory_size))
        self.W_query = nn.Linear(hidden_size, memory_size, bias=False)
        nn.init.orthogonal_(self.W_query.weight, gain=0.01)

        # ── Hidden state kernels (unchanged) ─────────────────────────────
        self.W_x = nn.Linear(input_size,  hidden_size, bias=True)
        self.W_h = spectral_norm(nn.Linear(hidden_size, hidden_size, bias=False))
        self.W_m = nn.Linear(input_size,  hidden_size, bias=False)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.uniform_(self.e_x, -1.0, 1.0)
        nn.init.xavier_normal_(self.E_h.weight)
        # e_m stays zero (Voelker 2019 §3)
        # W_pre stays identity (OrthoLayer.__init__)
        for layer in (self.W_x, self.W_h, self.W_m):
            nn.init.xavier_normal_(layer.weight)
        nn.init.zeros_(self.W_x.bias)

    # ── Gated write  [NEW METHOD] ─────────────────────────────────────────────

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
        This is not an approximation — it is a hard requirement.
        """
        pred  = u_h + u_m
        gate  = torch.tanh(u_x)
        innov = torch.tanh(u_x - pred)
        return self.W_pre(gate * innov) + pred

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        x:      torch.Tensor,   # (B, C)
        h_prev: torch.Tensor,   # (B, n)
        m_prev: torch.Tensor,   # (B, d, C)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns: h_new (B, n), m_new (B, d, C), r_intr (B,)

        r_intr = ‖u_actual − u_null‖₂
               = ‖W_pre(gate ⊙ innov)‖₂
               = ‖gate ⊙ innov‖₂        (W_pre is an isometry: ‖Wv‖₂ = ‖v‖₂)
               ∈ [0, √C]               (both tanh factors bounded in (−1, 1)^C)

        Step 1 (current): β = 0 — r_intr logged, not added to rewards.
        Step 2: r_combined = r_ext + β * r_intr, β starting at 0.001.
                Also add minimize-r_intr auxiliary loss with weight η.
        """

        # ── Step 1: per-channel u ─────────────────────────────────────────

        # [OLD] u_x = x * self.e_x
        e_x_n = F.normalize(self.e_x, dim=0)               # (C,)  ‖e_x_n‖₂ = 1
        u_x   = x * e_x_n                                   # (B, C)

        u_h = self.E_h(h_prev)                              # (B, C)
        u_m = torch.einsum('d,bdc->bc', self.e_m, m_prev)  # (B, C)

        # [OLD] u = u_x + u_h + u_m
        u_actual = self._compute_u(u_x, u_h, u_m)          # (B, C)

        # ── Intrinsic reward  [NEW] ───────────────────────────────────────
        # u_null: what _compute_u returns when u_x = 0.
        # Provably equals pred (gate_null = tanh(0) = 0 → W_pre(0) = 0).
        # Kept in the live graph (no stop_grad) so that gradients flow through
        # u_h and u_m, training E_h and e_m to predict u_x (world model signal).
        # At β=0 (Step 1), no gradient flows through r_intr anyway — the graph
        # connection is inert until r_intr appears in the loss at Step 2.
        u_null = self._compute_u(torch.zeros_like(u_x), u_h, u_m)  # (B, C) = pred
        r_intr = (u_actual - u_null).norm(dim=-1)                   # (B,)

        # ── Step 2: Legendre memory update ────────────────────────────────
        Am    = torch.einsum('ij,bjc->bic', self.A, m_prev)   # (B, d, C)
        # [OLD] Bu = self.B * u.unsqueeze(1)
        Bu    = self.B * u_actual.unsqueeze(1)                 # (B, d, C)
        m_new = Am + Bu                                        # (B, d, C)

        # ── Step 3: dynamic read head (unchanged) ─────────────────────────
        C_t = F.normalize(self.W_query(h_prev), dim=-1)       # (B, d)
        y   = torch.einsum('bd,bdc->bc', C_t, m_new)          # (B, C)

        # ── Step 4: hidden update (unchanged) ────────────────────────────
        h_new = torch.tanh(
            self.W_x(x) + self.W_h(h_prev) + self.W_m(y)
        )                                                      # (B, n)

        # [OLD] return h_new, m_new
        return h_new, m_new, r_intr

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
    # [OLD] Output : out (B, T, n),  final_state (h, m)
    Output : out (B, T, n),  r_intrs (B, T),  final_state (h, m)
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
            # [OLD] h, m = self.cell(x[:, t], h, m)
            h, m, r = self.cell(x[:, t], h, m)
            outs.append(h)
            r_intrs.append(r)
        # [OLD] return torch.stack(outs, dim=1), (h, m)
        return torch.stack(outs, dim=1), torch.stack(r_intrs, dim=1), (h, m)
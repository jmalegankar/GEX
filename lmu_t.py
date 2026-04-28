"""
Multichannel LMU — Gated Write variant (HiPPO-LegT backbone).

Gate types (controlled by gate_type arg to LMUCell):
─────────────────────────────────────────────────────
  'softsign_sum'   (default, recommended)
      gate  = softsign(u_x)          = u_x / (1 + |u_x|)
      innov = softsign(u_x − pred)   = (u_x−pred) / (1 + |u_x−pred|)
      write = W_pre(gate + innov) + pred

  'tanh_product'   (original, both null conditions exact)
      gate  = tanh(u_x)
      innov = tanh(u_x − pred)
      write = W_pre(gate ⊙ innov) + pred

  'none'   (no gating — plain additive write)
      write = u_x + u_h + u_m   (no W_pre, no gate, no innov)
      r_intr = ||u_x||₂          (raw encoder output magnitude)

Read head (NEW — controls the y readout that feeds into the hidden update):
─────────────────────────────────────────────────────────────────────────
  'dynamic' (default)
      C_t = normalize(W_query(h_prev))
      y   = C_t · m_new                              (content-addressable)
      W_query is allocated and learned.

  'first_coef'  (vanilla-LMU ablation)
      y = m_new[:, 0, :]                             (lowest-order Legendre coef)
      No W_query parameter is allocated. The lowest-order coefficient
      approximates the local mean of the window — closest single-vector
      analogue of what POPGym's vanilla LMU exposes through W_m.

      The cell's read_state(h, m) also adapts: 'dynamic' returns
      cat([h, layer_norm(m_pooled)]) of size hidden+input_size; 'first_coef'
      returns h alone of size hidden_size. The actor/critic head is sized
      from cell.head_input_size, which dispatches accordingly.

Unified r_intr formula (works for all gate types):
    r_intr = ||u_actual − pred||₂

m_norm anti-collapse residual:
    Same as before — controlled by residual_scale.

OrthoLayer (W_pre) notes:
    - 'none' mode does not call W_pre (still allocated for optimizer-exclusion uniformity).
    - 'tanh_product' and 'softsign_sum' both require W_pre excluded from Adam.
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

    No bias: W_pre(0) = 0 exactly, so r_intr = ||u_actual − pred|| equals
    ||W_pre(innovation_vec)||₂ = ||innovation_vec||₂ (isometry).

    CRITICAL: exclude from main Adam optimizer.
    """

    def __init__(self, size: int):
        super().__init__()
        self.weights = nn.Parameter(torch.eye(size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weights

    def ortho_update(self, lr: float) -> None:
        with torch.no_grad():
            if self.weights.grad is None:
                return
            G, W = self.weights.grad, self.weights
            A = G @ W.t() - W @ G.t()
            I = torch.eye(W.size(0), device=W.device, dtype=W.dtype)
            W_new = torch.linalg.solve(I + lr * A, (I - lr * A) @ W)
            if not (torch.isnan(W_new).any() or torch.isinf(W_new).any()):
                self.weights.copy_(W_new)
            self.weights.grad.zero_()

    @torch.no_grad()
    def reorthogonalize(self) -> None:
        if torch.isnan(self.weights).any() or torch.isinf(self.weights).any():
            nn.init.eye_(self.weights)
            return
        U, _, Vh = torch.linalg.svd(self.weights, full_matrices=False)
        self.weights.copy_(U @ Vh)

    @torch.no_grad()
    def orthogonality_error(self) -> float:
        I = torch.eye(self.weights.size(0), device=self.weights.device)
        return (self.weights.t() @ self.weights - I).norm().item()


# ─────────────────────────────────────────────────────────────────────────────
# HiPPO-LegT matrices (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def get_AB(d: int, theta: float) -> Tuple[np.ndarray, np.ndarray]:
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
ReadHead = Literal['dynamic', 'first_coef']


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
        read_head:       'dynamic' | 'first_coef' [NEW]

    Returns (from forward):
        h_new   (B, n)    — new hidden state
        m_new   (B, d, C) — new Legendre memory
        r_intr  (B,)      — prediction error, IN GRAPH
        gate    (B, C)    — detached diagnostic
        innov   (B, C)    — detached diagnostic
        u_x     (B, C)    — detached diagnostic

    For gate_type='none': gate and innov are zeros_like(u_x) (no meaning).
    For read_head='first_coef': W_query is None; y = m_new[:, 0, :].
    """

    def __init__(
        self,
        input_size:     int,
        hidden_size:    int,
        memory_size:    int,
        theta:          float,
        gate_type:      GateType = 'softsign_sum',
        residual_scale: float    = 0.05,
        read_head:      ReadHead = 'dynamic',
    ):
        super().__init__()
        self.input_size     = input_size
        self.hidden_size    = hidden_size
        self.memory_size    = memory_size
        self.gate_type      = gate_type
        self.residual_scale = residual_scale
        assert read_head in ('dynamic', 'first_coef'), (
            f"read_head must be 'dynamic' or 'first_coef'; got {read_head!r}"
        )
        self.read_head      = read_head

        # ── LegT memory matrices ──────────────────────────────────────────
        Ad, Bd = get_AB(memory_size, theta)
        self.register_buffer('A', torch.from_numpy(Ad))   # (d, d)
        self.register_buffer('B', torch.from_numpy(Bd))   # (d, 1)

        # ── Encoding parameters ───────────────────────────────────────────
        self.e_x    = nn.Parameter(torch.empty(input_size))
        self.E_h    = spectral_norm(nn.Linear(hidden_size, input_size, bias=False))
        self.e_m    = nn.Parameter(torch.zeros(memory_size))

        # ── W_pre: still allocated for all gate types (uniformity) ────────
        # Optimizer exclusion logic in policies.py uses hasattr(cell, 'W_pre').
        # For gate_type='none', W_pre.forward is never called.
        self.W_pre = OrthoLayer(input_size)

        # ── Dynamic read head: ALLOCATED ONLY WHEN USED [NEW] ─────────────
        # 'first_coef' read is parameter-free (just a slice of m_new), so
        # we don't allocate W_query at all — saves an unused optimizer slot.
        if read_head == 'dynamic':
            self.W_query = nn.Linear(hidden_size, memory_size, bias=False)
            nn.init.orthogonal_(self.W_query.weight, gain=0.01)
        else:
            self.W_query = None

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

    # ── Head interface ────────────────────────────────────────────────────────

    @property
    def head_input_size(self) -> int:
        """
        Size of read_state(h, m)'s output — the actor/critic head input.

        'dynamic'    → hidden_size + input_size  (cat of h with mean-pooled m)
        'first_coef' → hidden_size               (just h)
        """
        if self.read_head == 'dynamic':
            return self.hidden_size + self.input_size
        return self.hidden_size

    def read_state(self, h: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """
        Return the actor/critic head input for the current state.

        For 'dynamic': cat([h, layer_norm(m.mean(dim=1))]).
        For 'first_coef': h alone — m's contribution is already absorbed into
                          h via W_m during the cell's forward step.
        """
        if self.read_head == 'dynamic':
            m_pooled = F.layer_norm(m.mean(dim=1), [self.input_size])
            return torch.cat([h, m_pooled], dim=-1)
        return h

    # ── Gate implementations ──────────────────────────────────────────────────

    @staticmethod
    def _softsign(x: torch.Tensor) -> torch.Tensor:
        return x / (1.0 + x.abs())

    def _compute_write(
        self,
        u_x:  torch.Tensor,
        u_h:  torch.Tensor,
        u_m:  torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pred = u_h + u_m

        if self.gate_type == 'softsign_sum':
            gate  = self._softsign(u_x)
            innov = self._softsign(u_x - pred)
            innovation_vec = gate + innov
            u_actual = self.W_pre(innovation_vec) + pred

        elif self.gate_type == 'tanh_product':
            gate  = torch.tanh(u_x)
            innov = torch.tanh(u_x - pred)
            innovation_vec = gate * innov
            u_actual = self.W_pre(innovation_vec) + pred

        elif self.gate_type == 'none':
            u_actual = u_x + pred
            gate  = torch.zeros_like(u_x)
            innov = torch.zeros_like(u_x)

        else:
            raise ValueError(
                f"Unknown gate_type '{self.gate_type}'. "
                f"Choose from: softsign_sum, tanh_product, none"
            )

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
        """

        # ── Step 1: encode input ──────────────────────────────────────────
        e_x_n = F.normalize(self.e_x, dim=0)
        u_x   = x * e_x_n
        u_h   = self.E_h(h_prev)
        u_m   = torch.einsum('d,bdc->bc', self.e_m, m_prev)

        # ── Step 2: gated write ───────────────────────────────────────────
        u_actual, pred, gate, innov = self._compute_write(u_x, u_h, u_m)

        # ── Step 3: intrinsic reward ──────────────────────────────────────
        r_intr = (u_actual - pred).norm(dim=-1)

        # ── Step 4: Legendre memory update ────────────────────────────────
        Am    = torch.einsum('ij,bjc->bic', self.A, m_prev)
        Bu    = self.B * u_actual.unsqueeze(1)
        m_new = Am + Bu

        # ── Step 5: read head [DISPATCHES ON read_head] ───────────────────
        if self.read_head == 'dynamic':
            C_t = F.normalize(self.W_query(h_prev), dim=-1)
            y   = torch.einsum('bd,bdc->bc', C_t, m_new)
        else:
            # 'first_coef' — lowest-order Legendre coefficient. For LegT, the
            # first basis function is constant over the window, so m[:, 0, :]
            # is approximately the windowed mean of u_actual. Parameter-free,
            # closer to POPGym's vanilla-LMU readout.
            y = m_new[:, 0, :]

        # ── Step 6: hidden update ─────────────────────────────────────────
        h_new = torch.tanh(
            self.W_x(x) + self.W_h(h_prev) + self.W_m(y)
        )

        return h_new, m_new, r_intr, gate.detach(), innov.detach(), u_x.detach()

    def initial_state(
        self, n: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(n, self.hidden_size, device=device)
        m = torch.zeros(n, self.memory_size, self.input_size, device=device)
        return h, m


# ─────────────────────────────────────────────────────────────────────────────
# LMU — sequence wrapper (unchanged behavior, threads read_head)
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
        read_head:      ReadHead = 'dynamic',
    ):
        super().__init__()
        self.cell = LMUCell(
            input_size, hidden_size, memory_size, theta,
            gate_type=gate_type,
            residual_scale=residual_scale,
            read_head=read_head,
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
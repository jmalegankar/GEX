"""
Multichannel Legendre Memory Unit (LMU) — PyTorch implementation.

Core idea (S4-style):
    Run C independent scalar LMUs in parallel, one per input feature,
    sharing the same frozen (Ā, B̄) matrices.

    Old (scalar bottleneck):
        u : (B, 1)     — all C features funnel through one number
        m : (B, d)     — d Legendre coeffs for the whole vector

    New (multichannel):
        u : (B, C)     — one scalar per feature channel
        m : (B, d, C)  — d Legendre coeffs × C independent channels

Equations per channel c  (Voelker 2019, Eq. 4 + 6 + 7):
    u_c  = eₓ[c]·x_c  +  Eₕ[:,c]·h  +  eₘ·m[:,c]    scalar encoding
    m_c' = Ā m_c + B̄ u_c                               linear memory update
    y    = Cₚᵣₒⱼ·m'                                    per-channel readout
    h'   = tanh(Wₓ(x) + Wₕ(h) + Wₘ(y))               shared nonlinear hidden

Vectorised shapes:
    u    = x⊙eₓ  +  hEₕ  +  einsum('d,bdc→bc', eₘ, m)   (B, C)
    m'   = einsum('ij,bjc→bic', Ā, m)  +  B̄·u.unsqueeze(1)  (B, d, C)
    y    = einsum('d,bdc→bc', Cₚᵣₒⱼ, m')                    (B, C)
    h'   = tanh(Wₓ(x) + Wₕ(h) + Wₘ(y))                     (B, n)

Broadcasting proof for memory update:
    B̄         : (d, 1)  → padded by PyTorch to (1, d, 1)
    u.unsqueeze(1) : (B, 1, C)
    product        : (B, d, C)  element [b,i,c] = B̄[i] · u[b,c]  ✓
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import cont2discrete
from typing import Tuple


def get_AB(d: int, theta: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build continuous-time (A, B) from Voelker 2019 Eq. 2, then ZOH-discretise.

    A_ij = (2i+1) * { -1           if i < j
                    { (-1)^{i-j+1}  if i ≥ j

    B_i  = (2i+1) * (-1)^i

    ZOH via scipy.signal.cont2discrete gives exact Ā, B̄ for dt=1.
    Returns float32 arrays of shape (d, d) and (d, 1).
    """
    Q = np.arange(d, dtype=float)
    R = (2 * Q + 1)[:, None]          # (d, 1)
    j, i = np.meshgrid(Q, Q)          # i=row, j=col  (both (d,d))

    A = R * np.where(i < j, -1.0, (-1.0) ** (i - j + 1))
    A /= theta
    B = R * ((-1.0) ** Q)[:, None]    # (d, 1)
    B /= theta

    # ZOH: Ā = expm(A·dt),  B̄ = Ā (A⁻¹ B - A⁻¹ B e^{-Adt})  (computed by scipy)
    C_dummy = np.zeros((1, d))
    D_dummy = np.zeros((1,))
    Ad, Bd, _, _, _ = cont2discrete((A, B, C_dummy, D_dummy), dt=1.0, method='zoh')
    return Ad.astype(np.float32), Bd.astype(np.float32)


class LMUCell(nn.Module):
    """
    One step of the multichannel LMU.

    Args:
        input_size  (C): encoder output dim — one independent channel per feature
        hidden_size (n): nonlinear hidden state units
        memory_size (d): Legendre polynomial degree (more = finer temporal resolution)
        theta          : memory window length in time-steps
                         Set to ~max episode length.  Error ∝ θω/d.

    State shapes:
        h : (B, n)      shared nonlinear hidden state
        m : (B, d, C)   d Legendre coefficients × C independent channels

    Hyperparameter guidance:
        theta  — MemoryS7: ~100,  MemoryS13: ~200  (cover full episode)
        d      — 32–64 (32 is a reasonable start; increase if memory is shallow)
        n      — 64–128 (controls nonlinear capacity, independent of d)
    """

    def __init__(
        self,
        input_size:  int,
        hidden_size: int,
        memory_size: int,
        theta:       float,
    ):
        super().__init__()
        self.input_size  = input_size   # C
        self.hidden_size = hidden_size  # n
        self.memory_size = memory_size  # d

        # ── Fixed memory matrices (frozen, not trained) ───────────────────
        # Ā: (d, d)   B̄: (d, 1)
        # These encode the Legendre projection; training them destroys the
        # theoretical guarantees (Voelker 2019 §3).
        Ad, Bd = get_AB(memory_size, theta)
        self.register_buffer('A', torch.from_numpy(Ad))   # (d, d)
        self.register_buffer('B', torch.from_numpy(Bd))   # (d, 1)

        # ── Encoding: (x, h, m) → u per channel ──────────────────────────

        # eₓ ∈ ℝ^C — one scalar weight per input channel
        # u_from_x[b, c] = eₓ[c] * x[b, c]
        self.e_x = nn.Parameter(torch.empty(input_size))

        # Eₕ ∈ ℝ^{n×C} — hidden state → per-channel scalar
        # u_from_h[b, c] = Σ_i Eₕ[c, i] * h[b, i]  (Linear(n→C, no bias))
        self.E_h = nn.Linear(hidden_size, input_size, bias=False)

        # eₘ ∈ ℝ^d — memory readout for encoding, shared across channels
        # u_from_m[b, c] = Σ_i eₘ[i] * m[b, i, c]
        # MUST be initialised to 0 (prevents unstable memory feedback at init,
        # per Voelker 2019 §3: "memory's feedback encoders initialised to eₘ=0")
        self.e_m = nn.Parameter(torch.zeros(memory_size))

        # ── Memory readout for hidden update ─────────────────────────────
        # Cₚᵣₒⱼ ∈ ℝ^d — contracts d-dim memory → scalar per channel
        # Equivalent to the C output matrix in SSM notation: y = C m
        # y[b, c] = Σ_i Cₚᵣₒⱼ[i] * m_new[b, i, c]
        self.C_proj = nn.Parameter(torch.empty(memory_size))

        # ── Hidden state kernels ─────────────────────────────────────────
        # Wₓ: C → n  (input → hidden)
        # Wₕ: n → n  (hidden recurrence, no bias to avoid double-counting)
        # Wₘ: C → n  (memory readout y → hidden)
        self.W_x = nn.Linear(input_size,  hidden_size, bias=True)
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_m = nn.Linear(input_size,  hidden_size, bias=False)

        self._reset_parameters()

    def _reset_parameters(self):
        # eₓ: LeCun uniform (fan_in=1 per element → U[-1, 1])
        nn.init.uniform_(self.e_x, -1.0, 1.0)
        # Eₕ: Xavier normal (per Voelker 2019 §3)
        nn.init.xavier_normal_(self.E_h.weight)
        # e_m already zeros from __init__; do not reinit here
        # Cₚᵣₒⱼ: uniform ± 1/√d  (small to not dominate at init)
        nn.init.uniform_(self.C_proj,
                         -1.0 / self.memory_size ** 0.5,
                          1.0 / self.memory_size ** 0.5)
        # Hidden kernels: Xavier normal (per paper §3)
        for layer in (self.W_x, self.W_h, self.W_m):
            nn.init.xavier_normal_(layer.weight)
        nn.init.zeros_(self.W_x.bias)

    def forward(
        self,
        x:      torch.Tensor,   # (B, C)
        h_prev: torch.Tensor,   # (B, n)
        m_prev: torch.Tensor,   # (B, d, C)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            h_new : (B, n)
            m_new : (B, d, C)
        """

        # ── Step 1: compute u ∈ (B, C) ───────────────────────────────────
        # Each channel c gets an independent scalar:
        #   u_c = eₓ[c]·x_c  +  Eₕ[:,c]·h  +  eₘ·m[:,c]

        u_x = x * self.e_x                                    # (B, C) element-wise
        u_h = self.E_h(h_prev)                                # (B, C) via Linear(n→C)
        u_m = torch.einsum('d,bdc->bc', self.e_m, m_prev)    # (B, C) dot over d-dim
        u   = u_x + u_h + u_m                                 # (B, C)

        # ── Step 2: Legendre memory update ───────────────────────────────
        # m'[b,i,c] = Σ_j Ā[i,j] m[b,j,c]  +  B̄[i] · u[b,c]
        #
        # einsum('ij,bjc->bic'): matrix-multiply Ā over the d dimension,
        #   independently for every batch element b and channel c.
        # B̄: (d, 1), u.unsqueeze(1): (B, 1, C)
        # PyTorch pads B̄ to (1, d, 1) → broadcast to (B, d, C)  ✓
        Am    = torch.einsum('ij,bjc->bic', self.A, m_prev)   # (B, d, C)
        Bu    = self.B * u.unsqueeze(1)                        # (B, d, C)
        m_new = Am + Bu                                        # (B, d, C)

        # ── Step 3: memory readout → y ∈ (B, C) ─────────────────────────
        # y[b,c] = Σ_i Cₚᵣₒⱼ[i] · m_new[b,i,c]
        # (SSM C-matrix: maps d-dim Legendre state → scalar per channel)
        y = torch.einsum('d,bdc->bc', self.C_proj, m_new)     # (B, C)

        # ── Step 4: nonlinear hidden update ──────────────────────────────
        # h' = tanh(Wₓ x + Wₕ h + Wₘ y)
        h_new = torch.tanh(
            self.W_x(x) + self.W_h(h_prev) + self.W_m(y)
        )                                                      # (B, n)

        return h_new, m_new

    def initial_state(
        self, n: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Zero initial state.  n = batch_size or n_envs."""
        h = torch.zeros(n, self.hidden_size, device=device)
        m = torch.zeros(n, self.memory_size, self.input_size, device=device)
        return h, m


class LMU(nn.Module):
    """
    Sequence wrapper around LMUCell.
    Input  : x  (B, T, C)
    Output : out (B, T, n),  final_state (h, m)
    """
    def __init__(self, input_size, hidden_size, memory_size, theta):
        super().__init__()
        self.cell = LMUCell(input_size, hidden_size, memory_size, theta)

    def forward(
        self,
        x:     torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T, _ = x.shape
        h, m = state if state is not None else self.cell.initial_state(B, x.device)
        outs = []
        for t in range(T):
            h, m = self.cell(x[:, t], h, m)
            outs.append(h)
        return torch.stack(outs, dim=1), (h, m)


# if __name__ == '__main__':
#     torch.manual_seed(0)
#     B, C, n, d = 4, 64, 128, 32
#     theta = 100.0  # MemoryS7; use 200 for MemoryS13

#     cell = LMUCell(C, n, d, theta)

#     # Sanity: spectral radius of Ā must be < 1 for stability
#     rho = torch.linalg.eigvals(cell.A).abs().max().item()
#     assert rho < 1.0, f"Unstable Ā: spectral radius {rho:.4f}"
#     print(f"Ā spectral radius: {rho:.6f}  ✓ stable")

#     # A, B must be non-trainable
#     assert not cell.A.requires_grad and not cell.B.requires_grad

#     h, m = cell.initial_state(B, torch.device('cpu'))
#     x = torch.randn(B, C)
#     h_new, m_new = cell(x, h, m)
#     assert h_new.shape == (B, n),    f"h shape: {h_new.shape}"
#     assert m_new.shape == (B, d, C), f"m shape: {m_new.shape}"
#     print(f"h: {tuple(h_new.shape)}  m: {tuple(m_new.shape)}  ✓")

#     # Sequence wrapper
#     model = LMU(C, n, d, theta)
#     seq = torch.randn(B, 50, C)
#     out, (h_f, m_f) = model(seq)
#     assert out.shape == (B, 50, n)
#     print(f"LMU sequence output: {tuple(out.shape)}  ✓")
#     print("All checks passed.")
"""
LMU cell for HSWVIME-PPO.

Faithful to Voelker et al. (NeurIPS 2019):
  - Scalar encoding signal u_t = e_x x_t + e_h h_{t-1} + e_m m_{t-1}
  - Single (d×d) A and (d×1) B matrices — NOT block-diagonal
  - Separate hidden (n) and memory (d) dimensions

State stored in the rollout buffer: flat (B, hidden_dim + order)
  h : (B, hidden_dim)   — nonlinear hidden state, read by feature extractor
  m : (B, order)        — Legendre memory state, must survive across steps
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import cont2discrete
from typing import Tuple


def _get_AB(order: int, theta: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct continuous-time A ∈ ℝ^{d×d}, B ∈ ℝ^{d×1} (Padé delay line),
    then discretise via ZOH with dt=1.
    """
    Q = np.arange(order, dtype=float)
    R = (2 * Q + 1)[:, None]                             # (d, 1)
    j, i = np.meshgrid(Q, Q)
    A = R * np.where(i < j, -1.0, (-1.0) ** (i - j + 1))
    A /= theta
    B = R * ((-1.0) ** Q)[:, None]
    B /= theta
    C = np.zeros((1, order))
    D = np.zeros((1,))
    A_d, B_d, *_ = cont2discrete((A, B, C, D), dt=1.0, method="zoh")
    return A_d.astype(np.float32), B_d.astype(np.float32)


class LMUCell(nn.Module):
    """
    One step of the LMU (Voelker et al. 2019).

    u_t = e_x x_t + e_h h_{t-1} + e_m m_{t-1}   (scalar per sample)
    m_t = Ā m_{t-1} + B̄ u_t                       (linear memory update)
    h_t = tanh(W_x x_t + W_h h_{t-1} + W_m m_t)  (nonlinear hidden state)

    Buffer interface: state is a flat (B, hidden_dim + order) tensor.
    """

    def __init__(self, input_dim: int, hidden_dim: int, order: int = 4, theta: float = 50.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.order = order
        self.flat_state_dim = hidden_dim + order   # h concat m

        # Fixed (non-trainable) memory matrices — single (d,d) and (d,1)
        A_d, B_d = _get_AB(order, theta)
        self.register_buffer("A", torch.from_numpy(A_d))   # (order, order)
        self.register_buffer("B", torch.from_numpy(B_d))   # (order, 1)

        # Scalar encoding: projects inputs into the signal written to memory
        self.e_x = nn.Linear(input_dim,  1, bias=False)
        self.e_h = nn.Linear(hidden_dim, 1, bias=False)
        self.e_m = nn.Linear(order,      1, bias=False)

        # Hidden state kernels
        self.W_x = nn.Linear(input_dim,  hidden_dim, bias=True)
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_m = nn.Linear(order,      hidden_dim, bias=False)

        self._reset_parameters()

    def _reset_parameters(self):
        # e_m = 0: no memory feedback at init (prevents instability)
        nn.init.zeros_(self.e_m.weight)
        # Xavier normal for hidden kernels (per paper §3)
        for layer in (self.W_x, self.W_h, self.W_m):
            nn.init.xavier_normal_(layer.weight)
        # LeCun uniform for encoding vectors
        for layer in (self.e_x, self.e_h):
            fan_in = layer.weight.shape[1]
            nn.init.uniform_(layer.weight, -1.0 / fan_in ** 0.5, 1.0 / fan_in ** 0.5)

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        x:     (B, input_dim)
        state: (B, flat_state_dim)  — flat [h, m]
        ->     (B, flat_state_dim)
        """
        h = state[:, :self.hidden_dim]   # (B, hidden_dim)
        m = state[:, self.hidden_dim:]   # (B, order)

        # Scalar signal written to memory
        u = self.e_x(x) + self.e_h(h) + self.e_m(m)   # (B, 1)

        # Linear memory update: m_t = Ā m_{t-1} + B̄ u_t
        m_new = m @ self.A.T + u * self.B.T             # (B, order)

        # Nonlinear hidden update
        h_new = torch.tanh(self.W_x(x) + self.W_h(h) + self.W_m(m_new))  # (B, hidden_dim)

        return torch.cat([h_new, m_new], dim=-1)        # (B, flat_state_dim)

    def unpack_h(self, state: torch.Tensor) -> torch.Tensor:
        """Extract just h from the flat state. (B, flat) -> (B, hidden_dim)"""
        return state[:, :self.hidden_dim]

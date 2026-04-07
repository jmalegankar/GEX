"""
Legendre Memory Unit (LMU) — PyTorch implementation
Based on: Voelker et al., NeurIPS 2019

"""

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import cont2discrete


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_AB(d: int, theta: float = 1.0):
    """
    Construct the continuous-time A ∈ ℝ^{d×d}, B ∈ ℝ^{d×1} matrices
    from the Padé approximant of an ideal delay line (paper eq. 2),
    then discretise via zero-order hold (ZOH) with dt=1.

    Scalar u_t is intentional: the Legendre basis decomposition depends
    on the linear dynamics of A, B being preserved exactly. Expanding to
    vector u_t via repeated B columns collapses to a scalar sum and adds
    no memory capacity — use larger d or multiple cells instead.

    Args:
        d     : memory dimension
        theta : window length (time-steps); set to episode horizon or tuned

    Returns:
        A_d, B_d : discretised numpy arrays
    """
    Q = np.arange(d, dtype=float)
    R = (2 * Q + 1)[:, None]                              # (d, 1)
    j, i = np.meshgrid(Q, Q)

    # Continuous A
    A = R * np.where(i < j, -1.0, (-1.0) ** (i - j + 1))
    A /= theta

    # Continuous B (scalar input — preserved from paper)
    B = R * ((-1.0) ** Q)[:, None]
    B /= theta

    # ZOH discretisation via scipy (only runs once at init)
    C = np.zeros((1, d))
    D = np.zeros((1,))
    A_d, B_d, _, _, _ = cont2discrete((A, B, C, D), dt=1.0, method="zoh")
    return A_d.astype(np.float32), B_d.astype(np.float32)


# ---------------------------------------------------------------------------
# LMU Cell  (single time-step, for use in RL rollout loops)
# ---------------------------------------------------------------------------

class LMUCell(nn.Module):
    """
    One step of the LMU:

        u_t = e_x x_t + e_h h_{t-1} + e_m m_{t-1}   (scalar per sample)
        m_t = Ā m_{t-1} + B̄ u_t                      (linear memory update)
        h_t = tanh(W_x x_t + W_h h_{t-1} + W_m m_t)  (nonlinear hidden state)

    State tuple: (h, m)
      h : (batch, hidden_size)
      m : (batch, memory_size)
    """

    def __init__(
        self,
        input_size:  int,
        hidden_size: int,
        memory_size: int,
        theta:       float = 1.0,
    ):
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size   # n  — nonlinear units
        self.memory_size = memory_size   # d  — Legendre coefficients

        # Fixed (non-trainable) memory matrices
        A, B = get_AB(memory_size, theta)
        self.register_buffer("A", torch.from_numpy(A))   # (d, d)
        self.register_buffer("B", torch.from_numpy(B))   # (d, 1)

        # Encoding vectors — project inputs into the scalar u written to memory
        self.e_x = nn.Linear(input_size,  1, bias=False)
        self.e_h = nn.Linear(hidden_size, 1, bias=False)
        self.e_m = nn.Linear(memory_size, 1, bias=False)

        # Hidden state kernels
        self.W_x = nn.Linear(input_size,  hidden_size, bias=True)
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_m = nn.Linear(memory_size, hidden_size, bias=False)

        self._reset_parameters()

    def _reset_parameters(self):
        # e_m = 0: prevents memory feedback at init
        nn.init.zeros_(self.e_m.weight)
        # Xavier normal for hidden kernels (per paper §3)
        for layer in (self.W_x, self.W_h, self.W_m):
            nn.init.xavier_normal_(layer.weight)
        # LeCun uniform for encoding vectors (fan_in = weight columns)
        for layer in (self.e_x, self.e_h):
            fan_in = layer.weight.shape[1]
            nn.init.uniform_(
                layer.weight,
                -1.0 / fan_in ** 0.5,
                 1.0 / fan_in ** 0.5,
            )

    def forward(
        self,
        x:      torch.Tensor,   # (batch, input_size)
        h_prev: torch.Tensor,   # (batch, hidden_size)
        m_prev: torch.Tensor,   # (batch, memory_size)
    ):
        # Scalar signal written to memory per sample
        u = self.e_x(x) + self.e_h(h_prev) + self.e_m(m_prev)  # (batch, 1)

        # Linear memory update  m_t = Ā m_{t-1} + B̄ u_t
        m = m_prev @ self.A.T + u * self.B.T                    # (batch, d)

        # Nonlinear hidden update
        h = torch.tanh(self.W_x(x) + self.W_h(h_prev) + self.W_m(m))  # (batch, n)

        return h, m

    def initial_state(self, batch_size: int, device: torch.device):
        """Return zeroed (h, m) state tuple."""
        h = torch.zeros(batch_size, self.hidden_size, device=device)
        m = torch.zeros(batch_size, self.memory_size, device=device)
        return h, m


# ---------------------------------------------------------------------------
# LMU  (sequence wrapper — useful for offline / supervised use)
# ---------------------------------------------------------------------------

class LMU(nn.Module):
    """
    Runs LMUCell over a full sequence.

    Input  : x  (batch, seq_len, input_size)
    Output : out (batch, seq_len, hidden_size),  final state (h, m)
    """

    def __init__(
        self,
        input_size:  int,
        hidden_size: int,
        memory_size: int,
        theta:       float = 1.0,
    ):
        super().__init__()
        self.cell = LMUCell(input_size, hidden_size, memory_size, theta)

    @property
    def hidden_size(self):
        return self.cell.hidden_size

    @property
    def memory_size(self):
        return self.cell.memory_size

    def forward(self, x: torch.Tensor, state=None):
        B, T, _ = x.shape
        device   = x.device

        if state is None:
            h, m = self.cell.initial_state(B, device)
        else:
            h, m = state

        outputs = []
        for t in range(T):
            h, m = self.cell(x[:, t], h, m)
            outputs.append(h)

        return torch.stack(outputs, dim=1), (h, m)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, T, input_size = 4, 64, 16
    hidden_size      = 64
    memory_size      = 32
    theta            = float(T)   # window = full sequence length

    model = LMU(input_size, hidden_size, memory_size, theta=theta).to(device)
    x     = torch.randn(B, T, input_size, device=device)

    out, (h_final, m_final) = model(x)

    print(f"Input  : {tuple(x.shape)}")
    print(f"Output : {tuple(out.shape)}")
    print(f"h_final: {tuple(h_final.shape)}")
    print(f"m_final: {tuple(m_final.shape)}")
    print(f"A norm : {model.cell.A.norm():.4f}  (should be < 1 for stable memory)")

    # Check A, B are frozen
    assert not model.cell.A.requires_grad, "A must not be trainable"
    assert not model.cell.B.requires_grad, "B must not be trainable"
    print("All checks passed.")
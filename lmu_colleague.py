import numpy as np
import torch
import torch.nn as nn
from scipy.signal import cont2discrete
from typing import Tuple


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


class LMUCell(nn.Module):
    def __init__(self, input_size, hidden_size, memory_size, num_channels, theta):
        super().__init__()
        self.input_size   = input_size
        self.hidden_size  = hidden_size
        self.memory_size  = memory_size
        self.num_channels = num_channels
        self.theta        = theta

        A, B = get_AB(memory_size, theta)
        self.register_buffer('A', torch.from_numpy(A))  # (memory_size, memory_size)
        self.register_buffer('B', torch.from_numpy(B))  # (memory_size, 1)

        self.input_timestep_extractor = nn.Sequential(
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.ReLU(),
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Conv1d(num_channels, 1, kernel_size=hidden_size),
            nn.Sigmoid(),
            nn.Flatten(),
        )
        # Zero-init final Conv1d bias — e_m=0 analog (Voelker §3)
        nn.init.zeros_(self.input_timestep_extractor[4].bias)

        self.multi_timestep_extractor = nn.Sequential(
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.ReLU(),
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Conv1d(num_channels, hidden_size, kernel_size=hidden_size),
            nn.Sigmoid(),
            nn.Flatten(),
        )
        nn.init.zeros_(self.multi_timestep_extractor[4].bias)

        self.ut_processor = nn.Sequential(
            nn.Linear(num_channels + input_size, 2 * num_channels),
            nn.ReLU(),
            nn.Linear(2 * num_channels, num_channels),
        )

        self.W_x = nn.Linear(input_size,  hidden_size)
        self.W_h = nn.Linear(hidden_size, hidden_size)
        self.W_m = nn.Linear(num_channels, num_channels)

    @torch.jit.export
    def recon_data(
        self,
        memory:    torch.Tensor,   # (B, memory_size, num_channels)
        timesteps: torch.Tensor,   # (B, num_timesteps)
    ) -> torch.Tensor:
        r = timesteps if timesteps.dtype == torch.float32 \
            else (timesteps / self.theta).frac()

        x = 2.0 * r - 1.0
        batch, num_timesteps = x.shape

        # FIX 1: P_0 = 1 for ALL timestep positions, not just index 0
        Parr = [torch.ones(batch, num_timesteps, device=memory.device,
                           dtype=memory.dtype)]
        if self.memory_size > 1:
            Parr.append(x)
        for i in range(1, self.memory_size - 1):
            Parr.append(((2 * i + 1) * x * Parr[-1] - i * Parr[-2]) / (i + 1))

        P = torch.stack(Parr, dim=-1)   # (B, num_timesteps, memory_size)
        return torch.einsum('bti,bic->btc', P, memory)  # (B, num_timesteps, num_channels)

    @torch.jit.export
    def forward(
        self,
        x: torch.Tensor,   # (B, input_size)
        h: torch.Tensor,   # (B, hidden_size, num_channels)
        m: torch.Tensor,   # (B, memory_size, num_channels)
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # ── Write path ────────────────────────────────────────────────────────
        init_timesteps = self.input_timestep_extractor(
            h.transpose(1, 2)          # (B, num_channels, hidden_size)
        )                              # (B, 1)

        recon = self.recon_data(m, init_timesteps).view(-1, self.num_channels)
        # (B, num_channels) — reconstructed signal at one learned lag

        u_t = self.ut_processor(
            torch.cat([recon, x], dim=-1)  # (B, num_channels + input_size)
        )                                  # (B, num_channels)

        # FIX 2: B einsum — B is (memory_size, 1), u_t is (B, num_channels)
        # correct broadcast: result[b,i,c] = B[i] * u_t[b,c]
        m_new = (torch.einsum('ij,bjc->bic', self.A, m) +
                 torch.einsum('i,bc->bic', self.B.squeeze(-1), u_t))
        # (B, memory_size, num_channels)

        # ── Read path ─────────────────────────────────────────────────────────
        multi_timesteps = self.multi_timestep_extractor(
            h.transpose(1, 2)          # (B, num_channels, hidden_size)
        )                              # (B, hidden_size)

        recon = self.recon_data(m_new, multi_timesteps)
        # (B, hidden_size, num_channels) — history at hidden_size different lags

        h_new  = self.W_m(recon)                                   # (B, hidden_size, num_channels)
        h_new += self.W_x(x).unsqueeze(2)                         # (B, hidden_size, 1) → broadcast
        h_new += self.W_h(h.transpose(1, 2)).transpose(1, 2)      # (B, hidden_size, num_channels)
        h_new  = torch.tanh(h_new)                                 # (B, hidden_size, num_channels)

        return h_new, m_new

    def initial_state(
        self, n: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(n, self.hidden_size,  self.num_channels, device=device)
        m = torch.zeros(n, self.memory_size,  self.num_channels, device=device)
        return h, m
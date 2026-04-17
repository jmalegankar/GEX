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

    ZOH via scipy.signal.cont2discrete gives exact \hat{A}, \hat{B} for dt=1.
    Returns float32 arrays of shape (d, d) and (d, 1).
    """
    Q = np.arange(d, dtype=float)
    R = (2 * Q + 1)[:, None]          # (d, 1)
    j, i = np.meshgrid(Q, Q)          # i=row, j=col  (both (d,d))

    A = R * np.where(i < j, -1.0, (-1.0) ** (i - j + 1))
    A /= theta
    B = R * ((-1.0) ** Q)[:, None]    # (d, 1)
    B /= theta

    # ZOH: \hat{A} = expm(A dt),  \hat{B} = \hat{A} (A^{-1} B - A^{-1} B e^{-Adt})  (computed by scipy)
    C_dummy = np.zeros((1, d))
    D_dummy = np.zeros((1,))
    Ad, Bd, _, _, _ = cont2discrete((A, B, C_dummy, D_dummy), dt=1.0, method='zoh')
    return Ad.astype(np.float32), Bd.astype(np.float32)

class LMUCell(nn.Module):
    """
    One step of the multichannel LMU.

    Args:
        input_size: Dimensionality of input vector x_t
        hidden_size: Dimensionality of output vector h_t
        memory_size: Dimensionality of memory vector m_t (i.e. LMU order)
        num_channels: Number of parallel LMU channels (i.e. number of independent LMUs)
        theta: Time constant of the LMU memory (see Voelker 2019)
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        memory_size: int,
        num_channels: int,
        theta: float
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.num_channels = num_channels
        self.theta = theta

        # Get (A, B) matrices for the LMU memory update, shared across channels.
        A, B = get_AB(memory_size, theta)
        self.register_buffer('A', torch.from_numpy(A))  # (memory_size, memory_size)
        self.register_buffer('B', torch.from_numpy(B).view(-1))  # (memory_size)

        self.input_timestep_extractor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Conv1d(num_channels, 1, kernel_size=hidden_size),
            nn.Sigmoid(),
            nn.Flatten()
        )

        self.multi_timestep_extractor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Conv1d(num_channels, self.hidden_size, kernel_size=hidden_size),
            nn.Sigmoid(),
            nn.Flatten()
        )

        self.ut_processor = nn.Sequential(
            nn.Linear(2*num_channels, 2*num_channels),
            nn.ReLU(),
            nn.Linear(2*num_channels, num_channels)
        )

        self.W_x = nn.Linear(input_size, num_channels)
        self.W_h = nn.Linear(hidden_size, hidden_size)
        self.W_m = nn.Linear(hidden_size, hidden_size)

        self.e_m = nn.Parameter(torch.zeros(memory_size))
        self.e_h = nn.Parameter(torch.zeros(hidden_size))
        self.e_x = nn.Linear(input_size, num_channels)

        self._reset_parameters()
    
    def _reset_parameters(self):
        # Initialize weights using Xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        nn.init.uniform_(self.e_h, -0.1, 0.1)
        nn.init.uniform_(self.e_m, -0.1, 0.1)
        nn.init.uniform_(self.e_x.weight, -0.1, 0.1)
        if self.e_x.bias is not None:
            nn.init.zeros_(self.e_x.bias)
    
    @torch.jit.export
    def recon_data(
        self,
        memory: torch.Tensor,  # (batch_size, memory_size, num_channels)
        timesteps: torch.Tensor # (batch_size, num_timesteps)
    ) -> torch.Tensor:
        """
        Reconstruct input data from LMU memory using the A matrix.

        Args:
            memory: LMU memory state, shape (batch_size, memory_size, num_channels)
            timesteps: Time steps corresponding to each memory state, shape (batch_size, num_timesteps)
        Returns:
            recon: Reconstructed input data, shape (batch_size, num_timesteps, num_channels)
        """
        if timesteps.dtype != torch.float32:
            r = (timesteps / self.theta).frac()
        else:
            r = timesteps
        
        x = 2.0 * r - 1.0
        batch, num_timesteps = x.shape
        Parr = []
        P = torch.zeros(batch, num_timesteps, device=memory.device, dtype=memory.dtype)
        P[:, :] = 1.0
        Parr.append(P)
        if self.memory_size > 1:
            Parr.append(x)
        for i in range(1, self.memory_size - 1):
            Parr.append(((2 * i + 1) * x * Parr[-1] - i * Parr[-2]) / (i + 1))
        
        P = torch.stack(Parr, dim=-1) # (batch, num_timesteps, memory_size)

        return torch.einsum('bti,bic->btc', P, memory) # (batch, num_timesteps, num_channels)
    
    @torch.jit.export
    def forward(
        self,
        x: torch.Tensor,  # (batch_size, input_size)
        h: torch.Tensor,  # (batch_size, hidden_size, num_channels)
        m: torch.Tensor   # (batch_size, memory_size, num_channels)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute one step of the LMU.

        Args:
            x: Input at current time step, shape (batch_size, input_size)
            h: Hidden state from previous time step, shape (batch_size, hidden_size, num_channels)
            m: Memory state from previous time step, shape (batch_size, memory_size, num_channels)

        Returns:
            h_new: Updated hidden state, shape (batch_size, hidden_size, num_channels)
            m_new: Updated memory state, shape (batch_size, memory_size, num_channels)
        """
        init_timesteps = self.input_timestep_extractor(
            h.transpose(1, 2)  # (batch_size, num_channels, hidden_size)
        ) # (batch_size, 1)

        recon = self.recon_data(m, init_timesteps).view(-1, self.num_channels) # (batch_size, num_channels)

        u_t = torch.einsum('i,bic->bc', self.e_m, m) + torch.einsum('i,bic->bc', self.e_h, h) + self.e_x(x) # (batch_size, num_channels)
        u_t = self.ut_processor(
            torch.cat([recon, u_t], dim=-1) # (batch_size, 2*num_channels)
        ) # (batch_size, num_channels)

        m_new = torch.einsum('ij,bjc->bic', self.A, m) + torch.einsum('d,bc->bdc', self.B, u_t) # (batch_size, memory_size, num_channels)

        multi_timesteps = self.multi_timestep_extractor(
            h.transpose(1, 2)  # (batch_size, num_channels, hidden_size)
        ) # (batch_size, hidden_size)

        recon = self.recon_data(m_new, multi_timesteps) # (batch_size, hidden_size, num_channels)

        h_new = self.W_m(recon.transpose(1, 2)).transpose(1, 2) # (batch_size, hidden_size, num_channels)
        h_new += self.W_x(x).unsqueeze(1) # (batch_size, hidden_size, num_channels)
        h_new += self.W_h(h.transpose(1, 2)).transpose(1, 2) # (batch_size, hidden_size, num_channels)
        h_new = torch.tanh(h_new) # (batch_size, hidden_size, num_channels)

        return h_new, m_new


if __name__ == "__main__":
    # Test the LMUCell with dummy data
    batch_size = 4
    input_size = 10
    hidden_size = 32
    memory_size = 16
    num_channels = 10
    theta = 100.0

    lmu_cell = LMUCell(input_size, hidden_size, memory_size, num_channels, theta)

    x = torch.randn(batch_size, input_size)
    h = torch.randn(batch_size, hidden_size, num_channels)
    m = torch.randn(batch_size, memory_size, num_channels)

    h_new, m_new = lmu_cell(x, h, m)
    print("h_new shape:", h_new.shape)  # Expected: (batch_size, hidden_size, num_channels)
    print("m_new shape:", m_new.shape)  # Expected: (batch_size, memory_size, num_channels)

    ### LEARN PREDICTING MULTIPLE SIN WAVE WITH DIFFERENT FREQUENCIES AND PHASES
    # Train the LMUCell to predict the next step given initial steps of a multi-sin wave

    class PredictNextSinWave(nn.Module):
        def __init__(self, lmu_cell: LMUCell):
            super().__init__()
            self.lmu_cell = torch.jit.script(lmu_cell)
            self.output_layer = nn.Sequential(
                nn.Linear(hidden_size + memory_size, 2*input_size),
                nn.ReLU(),
                nn.Conv1d(num_channels, input_size, kernel_size=2*input_size),
                nn.Flatten(),
                nn.Tanh()
            )

        def forward(self, x, h, m):
            h_new, m_new = self.lmu_cell(x, h, m)
            concat = torch.cat((h_new, m_new), dim=1) # (batch_size, hidden_size + memory_size, num_channels)
            output = self.output_layer(concat.transpose(1, 2)) # (batch_size, input_size)
            return output, h_new, m_new

    predict_model = PredictNextSinWave(lmu_cell)
    optim = torch.optim.Adam(predict_model.parameters(), lr=3e-4)
    num_epochs = 1000
    num_timesteps = 100
    torch.manual_seed(0)
    for epoch in range(num_epochs):
        # Generate random multi-sin wave data
        t = torch.linspace(0, 10, num_timesteps)
        freqs = torch.rand(input_size) * 4 + 0.5  # Random frequencies between 0.5 and 4.5 Hz
        phases = torch.rand(input_size) * 2 * np.pi  # Random phases between 0 and 2pi
        data = torch.stack([torch.sin(freqs[i] * t + phases[i]) for i in range(input_size)], dim=1) # (num_timesteps, num_channels)

        # Initialize hidden and memory states
        h = torch.zeros(1, hidden_size, num_channels)
        m = torch.zeros(1, memory_size, num_channels)

        # Train on each time step
        total_loss = 0.0
        for t in range(num_timesteps//2):
            optim.zero_grad()
            output, h, m = predict_model(data[t].unsqueeze(0), h, m)
            loss = nn.MSELoss()(output.squeeze(0), data[t+1])  # Predict next time step
            loss.backward()
            optim.step()
            total_loss += loss.item()
            h, m = h.detach(), m.detach()  # Detach to prevent backprop through time
            output = output.detach()  # Detach to prevent backprop through time
        for t in range(num_timesteps//2, num_timesteps - 1):
            optim.zero_grad()
            output, h, m = predict_model(output, h, m)
            loss = nn.MSELoss()(output.squeeze(0), data[t+1])  # Predict next time step
            loss.backward()
            optim.step()
            h, m = h.detach(), m.detach()  # Detach to prevent backprop through time
            output = output.detach()  # Detach to prevent backprop through time
            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss / (num_timesteps - 1)}")
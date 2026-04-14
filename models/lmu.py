import torch as th
import torch.nn as nn
import math

from typing import Tuple

class LMUCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        memory_size: int,
        num_units: int,
        theta: int,
    ):
        super(LMUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.num_units = num_units
        self.theta = theta

        # Precompute the A and B matrices for the Legendre Memory Unit
        A = th.zeros(memory_size, memory_size, dtype=th.float32)
        for i in range(memory_size):
            for j in range(memory_size):
                if i < j:
                    A[i, j] = -2*i - 1
                else:
                    A[i, j] = (-1 ** ((i-j+1)%2)) * (2*i + 1)
        B = th.zeros(memory_size, dtype=th.int32)
        for i in range(memory_size):
            B[i] = (2*i + 1)*(-1 ** (i%2))
        self.register_buffer('A', A.view(1, memory_size, memory_size))
        self.register_buffer('B', B.view(memory_size, 1))

        # Precompute the permutation matrix for the memory update
        P = th.zeros(memory_size, memory_size, dtype=th.float32)
        for i in range(memory_size):
            neg = (-1)**(i%2)
            for j in range(i+1):
                P[i, j] = neg*math.comb(i, j) * math.comb(i+j, j) * (0.5**j)
        
        self.register_buffer('P', P.view(1, 1, memory_size, memory_size))

        # Attention to calculate u_t
        self.decode_mh = nn.MultiheadAttention(
            embed_dim=num_units,
            num_heads=1,
            batch_first=True,
        )
        
        self.proj_input = nn.Linear(input_size, num_units)
        self.encode_input = nn.MultiheadAttention(
            embed_dim=num_units,
            num_heads=1,
            batch_first=True,
        )

        self.enc_input_hidden = nn.Conv1d(
            in_channels=1,
            out_channels=num_units,
            kernel_size=input_size,
        )

        self.hidden_update = nn.MultiheadAttention(
            embed_dim=num_units,
            num_heads=1,
            batch_first=True,
        )
    
    @th.jit.export
    def forward(
        self,
        input: th.Tensor, # (batch_size, input_size)
        hidden: th.Tensor, # (batch_size, hidden_size, num_units)
        memory: th.Tensor, # (batch_size, memory_size, num_units)
    ) -> Tuple[th.Tensor, th.Tensor]:
        decoded = hidden + self.decode_mh(
            hidden, memory, memory, need_weights=False
        )[0] # (batch_size, memory_size, num_units)

        input_proj = self.proj_input(input) # (batch_size, num_units)
        u_t, _ = self.encode_input(
            input_proj.unsqueeze(1), decoded, decoded, need_weights=False
        ) # (batch_size, 1, num_units), None

        # Update the memory using the LMU equations
        u_t = u_t.squeeze(1) # (batch_size, num_units)
        new_memory = th.matmul(self.A, memory) + self.B * u_t.unsqueeze(1) # (batch_size, memory_size, num_units)
        kv = th.cat(
            (hidden, new_memory, input_proj.unsqueeze(1)),
            dim=1
        ) # (batch_size, hidden_size + memory_size + 1, num_units)
        new_hidden = hidden + self.hidden_update(
            hidden, kv, kv, need_weights=False
        )[0] # (batch_size, hidden_size, num_units)

        return new_hidden, new_memory
    
    @th.jit.export
    def recon(
        self,
        memory: th.Tensor, # (batch_size, memory_size, num_units)
        timesteps: th.Tensor, # (batch_size, num_timesteps)
    ) -> th.Tensor: # (batch_size, num_timesteps, num_units)
        # Reconstruct the input from the memory using the precomputed permutation matrix
        timesteps = -2 * timesteps / self.theta # (batch_size, num_timesteps)
        timesteps = timesteps.unsqueeze(2).repeat(1, 1, self.memory_size) # (batch_size, num_timesteps, memory_size)
        # Create exponentials for the reconstruction
        timesteps = timesteps ** th.arange(self.memory_size, device=memory.device).view(1, 1, -1) # (batch_size, num_timesteps, memory_size)
        # Get permutation matrix for timesteps
        P = (self.P * timesteps.unsqueeze(2)).sum(dim=-1) # (batch_size, num_timesteps, memory_size)
        # Perform the reconstruction
        recon = (P.unsqueeze(-1) * memory.unsqueeze(1)).sum(dim=2) # (batch_size, num_timesteps, num_units)
        return recon



### TODO: MOVE THIS TO WYNER WHEN CLEANING UP
from typing import List, Optional

@th.jit.script
class WynerOutput:
    def __init__(
        self,
        w: th.Tensor,
        logvar: th.Tensor,
        recon: th.Tensor,
        recon_next: Optional[th.Tensor] = None,
        prior_mu: Optional[th.Tensor] = None,
        prior_logvar: Optional[th.Tensor] = None,
    ):
        self.recon = recon
        self.recon_next = recon_next
        self.w = w
        self.logvar = logvar
        self.prior_mu = prior_mu
        self.prior_logvar = prior_logvar


@th.jit.script
class WynerLoss:
    def __init__(
        self,
        kl_loss: th.Tensor,
        recon_loss: Optional[th.Tensor] = None,
        recon_next_loss: Optional[th.Tensor] = None,
    ):
        self.recon_loss = recon_loss
        self.recon_next_loss = recon_next_loss
        self.kl_loss = kl_loss

@th.jit.interface
class WynerInterface:
    def encode(self, w: th.Tensor, mu: th.Tensor, skips: Optional[List[th.Tensor]], timestep: Optional[th.Tensor] = None) -> Tuple[th.Tensor, th.Tensor]:
        pass

    def decode(self, z: th.Tensor, timestep: Optional[th.Tensor] = None) -> th.Tensor:
        pass

    def forward(self, w: th.Tensor, mu: th.Tensor, skips: Optional[List[th.Tensor]], timestep: Optional[th.Tensor] = None) -> WynerOutput:
        pass

    def loss(self, output: WynerOutput, recon_target: Optional[th.Tensor], recon_next_target: Optional[th.Tensor]) -> WynerLoss:
        pass


class WynerVAE(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        memory_size: int,
        num_units: int,
        theta: int,
        decode_hidden: int,
        decode_output: int,
    ):
        super(WynerVAE, self).__init__()
        self.lmu_cell = LMUCell(input_size, hidden_size, memory_size, num_units, theta)
        self.log_var = nn.MultiheadAttention(
            embed_dim=num_units,
            num_heads=1,
            batch_first=True,
        )

        self.decode_hidden = decode_hidden
        self.decode_output = decode_output

        self.decoder = nn.MultiheadAttention(
            embed_dim=num_units,
            num_heads=1,
            batch_first=True,
        )

        self.decode_fc = nn.Sequential(
            nn.Linear(2*num_units, decode_hidden),
            nn.ReLU(),
            nn.Linear(decode_hidden, decode_output),
        )

    
    def encode(self, w: th.Tensor, mu: th.Tensor, skips: Optional[List[th.Tensor]], timestep: Optional[th.Tensor] = None) -> Tuple[th.Tensor, th.Tensor]:
        h, m = w[..., :self.lmu_cell.hidden_size, :], w[..., self.lmu_cell.hidden_size:, :]
        new_h, new_m = self.lmu_cell(input=mu, hidden=h, memory=m)
        z_mu = th.cat((new_h, new_m), dim=-2) # (batch_size, hidden_size + memory_size, num_units)
        z_logvar = self.log_var(z_mu, z_mu, z_mu, need_weights=False)[0] # (batch_size, hidden_size + memory_size, num_units)
        return z_mu, z_logvar
    
    def decode(self, z: th.Tensor, timestep: Optional[th.Tensor] = None) -> th.Tensor:
        h, m = z[..., :self.lmu_cell.hidden_size, :], z[..., self.lmu_cell.hidden_size:, :]
        recon = self.lmu_cell.recon(m, timestep) # (batch_size, num_timesteps, num_units)
        kv = th.cat((h,m), dim=-2) # (batch_size, hidden_size + memory_size, num_units)
        attn = self.decoder(recon, kv, kv, need_weights=False)[0] # (batch_size, num_timesteps, num_units)
        recon = th.cat((recon, attn), dim=-1) # (batch_size, num_timesteps, 2*num_units)
        recon = self.decode_fc(recon) # (batch_size, num_timesteps, decode_output)
        return recon
    
    def forward(self, w: th.Tensor, mu: th.Tensor, skips: Optional[List[th.Tensor]], timestep: Optional[th.Tensor] = None) -> WynerOutput:
        z_mu, z_logvar = self.encode(w, mu, skips, timestep)
        # prior
        prior_mu, prior_logvar = self.encode(
            w, th.zeros_like(mu), skips, timestep
        )
        std = th.exp(0.5 * z_logvar)
        eps = th.randn_like(std)
        z = z_mu + eps * std
        if timestep is not None:
            recon = self.decode(z, timestep)
            recon_next = self.decode(z, timestep+1)
        else:
            recon = None
            recon_next = None
        return WynerOutput(
            w=z_mu,
            logvar=z_logvar,
            recon=recon,
            recon_next=recon_next,
            prior_mu=prior_mu,
            prior_logvar=prior_logvar,
        )
    
    def loss(self, output: WynerOutput, recon_target: Optional[th.Tensor], recon_next_target: Optional[th.Tensor]) -> WynerLoss:
        kl_loss = -0.5 * th.sum(1 + output.logvar - output.w.pow(2) - output.logvar.exp())
        recon_loss = None
        recon_next_loss = None
        if output.recon is not None and recon_target is not None:
            recon_loss = nn.functional.mse_loss(output.recon, recon_target)
        if output.recon_next is not None and recon_next_target is not None:
            recon_next_loss = nn.functional.mse_loss(output.recon_next, recon_next_target)
        return WynerLoss(
            kl_loss=kl_loss,
            recon_loss=recon_loss,
            recon_next_loss=recon_next_loss,
        )



if __name__ == "__main__":
    batch_size = 3
    input_size = 7
    hidden_size = 11
    memory_size = 13
    num_units = 17
    theta = 93

    cell = LMUCell(input_size, hidden_size, memory_size, num_units, theta)
    cell = th.jit.script(cell)
    input = th.randn(batch_size, input_size)
    hidden = th.randn(batch_size, hidden_size, num_units)
    memory = th.randn(batch_size, memory_size, num_units)

    new_hidden, new_memory = cell(input, hidden, memory)
    print("New Hidden Shape:", new_hidden.shape) # (batch_size, hidden_size, num_units)
    print("New Memory Shape:", new_memory.shape) # (batch_size, memory_size, num_units)

    timesteps = th.arange(10).unsqueeze(0).repeat(batch_size, 1) # (batch_size, num_timesteps)
    recon = cell.recon(new_memory, timesteps)
    print("Reconstructed Input Shape:", recon.shape) # (batch_size, num_timesteps, num_units)

    vae = WynerVAE(
        input_size=input_size,
        hidden_size=hidden_size,
        memory_size=memory_size,
        num_units=num_units,
        theta=theta,
        decode_hidden=19,
        decode_output=23,
    )

    # Test the VAE forward pass
    w = th.randn(batch_size, hidden_size + memory_size, num_units)
    mu = th.randn(batch_size, input_size)
    output = vae(w, mu, skips=None, timestep=timesteps)
    print("WynerVAE Output Recon Shape:", output.recon.shape) # (batch_size, num_timesteps, decode_output)
    print("WynerVAE Output Recon Next Shape:", output.recon_next.shape) # (batch_size, num_timesteps, decode_output)

    # Calculate total number of parameters in the VAE
    total_params = sum(p.numel() for p in vae.parameters())
    print("Total number of parameters in WynerVAE:", total_params)
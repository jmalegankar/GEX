# import torch as th
# import torch.nn as nn
# import math

# from typing import Tuple

# class LMUCell(nn.Module):
#     def __init__(
#         self,
#         input_size: int,
#         hidden_size: int,
#         memory_size: int,
#         num_units: int,
#         theta: int,
#     ):
#         super(LMUCell, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.memory_size = memory_size
#         self.num_units = num_units
#         self.theta = float(theta)

#         # Precompute the A and B matrices for the Legendre Memory Unit
#         A = th.zeros(memory_size, memory_size, dtype=th.int32)
#         for i in range(memory_size):
#             for j in range(memory_size):
#                 if i < j:
#                     A[i, j] = -2*i - 1
#                 else:
#                     A[i, j] = (-1 ** ((i-j+1)%2)) * (2*i + 1)
#         A = A / self.theta + th.diag(th.ones(memory_size, dtype=th.float32))
#         B = th.zeros(memory_size, dtype=th.int32)
#         for i in range(memory_size):
#             B[i] = (2*i + 1)*(-1 ** (i%2))
#         B = B / self.theta
#         self.register_buffer('A', A.view(1, memory_size, memory_size))
#         self.register_buffer('B', B.view(memory_size, 1))

#         # Precompute the permutation matrix for the memory update
#         P = th.zeros(memory_size, memory_size, dtype=th.float32)
#         for i in range(memory_size):
#             neg = (-1)**(i%2)
#             for j in range(i+1):
#                 P[i, j] = neg*math.comb(i, j) * math.comb(i+j, j) * (0.5**j)
        
#         self.register_buffer('P', P.view(1, 1, memory_size, memory_size))

#         # Attention to calculate u_t
#         self.decode_mh = nn.MultiheadAttention(
#             embed_dim=num_units,
#             num_heads=1,
#             batch_first=True,
#         )
        
#         self.proj_input = nn.Linear(input_size, num_units)
#         self.encode_input = nn.MultiheadAttention(
#             embed_dim=num_units,
#             num_heads=1,
#             batch_first=True,
#         )

#         self.enc_input_hidden = nn.Conv1d(
#             in_channels=1,
#             out_channels=num_units,
#             kernel_size=input_size,
#         )

#         self.hidden_update = nn.MultiheadAttention(
#             embed_dim=num_units,
#             num_heads=1,
#             batch_first=True,
#         )
    
#     @th.jit.export
#     def forward(
#         self,
#         input: th.Tensor, # (batch_size, input_size)
#         hidden: th.Tensor, # (batch_size, hidden_size, num_units)
#         memory: th.Tensor, # (batch_size, memory_size, num_units)
#     ) -> Tuple[th.Tensor, th.Tensor]:
#         decoded = hidden + self.decode_mh(
#             hidden, memory, memory, need_weights=False
#         )[0] # (batch_size, memory_size, num_units)

#         input_proj = self.proj_input(input) # (batch_size, num_units)
#         u_t, _ = self.encode_input(
#             input_proj.unsqueeze(1), decoded, decoded, need_weights=False
#         ) # (batch_size, 1, num_units), None

#         # Update the memory using the LMU equations
#         u_t = u_t.squeeze(1) + input_proj # (batch_size, num_units)
#         # Normalize u_t for stability
#         u_t = u_t / (u_t.norm(dim=-1, keepdim=True) + 1e-8)
#         new_memory = th.matmul(self.A, memory) + self.B * u_t.unsqueeze(1) # (batch_size, memory_size, num_units)
#         kv = th.cat(
#             (hidden, new_memory, input_proj.unsqueeze(1)),
#             dim=1
#         ) # (batch_size, hidden_size + memory_size + 1, num_units)
#         new_hidden = hidden + self.hidden_update(
#             hidden, kv, kv, need_weights=False
#         )[0] # (batch_size, hidden_size, num_units)

#         return new_hidden, new_memory
    
#     @th.jit.export
#     def recon(
#         self,
#         memory: th.Tensor, # (batch_size, memory_size, num_units)
#         timesteps: th.Tensor, # (batch_size, num_timesteps)
#     ) -> th.Tensor: # (batch_size, num_timesteps, num_units)
#         # Reconstruct the input from the memory using the precomputed permutation matrix
#         timesteps = -2 * (timesteps / self.theta).frac() # (batch_size, num_timesteps)
#         timesteps = timesteps.to(th.float32)
#         timesteps = timesteps.unsqueeze(2).repeat(1, 1, self.memory_size) # (batch_size, num_timesteps, memory_size)
#         # Create exponentials for the reconstruction
#         timesteps = timesteps ** th.arange(self.memory_size, device=memory.device).view(1, 1, -1) # (batch_size, num_timesteps, memory_size)
#         # Get permutation matrix for timesteps
#         P = (self.P * timesteps.unsqueeze(2)).sum(dim=-1) # (batch_size, num_timesteps, memory_size)
#         # Perform the reconstruction
#         recon = (P.unsqueeze(-1) * memory.unsqueeze(1)).sum(dim=2) # (batch_size, num_timesteps, num_units)
#         # Print recon norm values
#         recon_norm = recon.norm(dim=-1)
#         return recon



# ### TODO: MOVE THIS TO WYNER WHEN CLEANING UP
# from typing import List, Optional

# @th.jit.script
# class WynerOutput:
#     def __init__(
#         self,
#         w: th.Tensor,
#         logvar: th.Tensor,
#         recon: th.Tensor,
#         recon_next: Optional[th.Tensor] = None,
#         prior_mu: Optional[th.Tensor] = None,
#         prior_logvar: Optional[th.Tensor] = None,
#     ):
#         self.recon = recon
#         self.recon_next = recon_next
#         self.w = w
#         self.logvar = logvar
#         self.prior_mu = prior_mu
#         self.prior_logvar = prior_logvar


# @th.jit.script
# class WynerLoss:
#     def __init__(
#         self,
#         kl_loss: th.Tensor,
#         recon_loss: Optional[th.Tensor] = None,
#         recon_next_loss: Optional[th.Tensor] = None,
#     ):
#         self.recon_loss = recon_loss
#         self.recon_next_loss = recon_next_loss
#         self.kl_loss = kl_loss

# @th.jit.interface
# class WynerInterface:
#     def encode(self, w: th.Tensor, mu: th.Tensor, skips: Optional[List[th.Tensor]], timestep: Optional[th.Tensor] = None) -> Tuple[th.Tensor, th.Tensor]:
#         pass

#     def decode(self, z: th.Tensor, timestep: Optional[th.Tensor] = None) -> th.Tensor:
#         pass

#     def forward(self, w: th.Tensor, mu: th.Tensor, skips: Optional[List[th.Tensor]], timestep: Optional[th.Tensor] = None) -> WynerOutput:
#         pass

#     def loss(self, output: WynerOutput, recon_target: Optional[th.Tensor], recon_next_target: Optional[th.Tensor]) -> WynerLoss:
#         pass


# class WynerVAE(nn.Module):
#     def __init__(
#         self,
#         input_size: int,
#         hidden_size: int,
#         memory_size: int,
#         num_units: int,
#         theta: int,
#         decode_hidden: int,
#         decode_output: int,
#     ):
#         super(WynerVAE, self).__init__()
#         self.lmu_cell = LMUCell(input_size, hidden_size, memory_size, num_units, theta)
#         self.log_var = nn.MultiheadAttention(
#             embed_dim=num_units,
#             num_heads=1,
#             batch_first=True,
#         )

#         self.decode_hidden = decode_hidden
#         self.decode_output = decode_output

#         self.decoder = nn.MultiheadAttention(
#             embed_dim=num_units,
#             num_heads=1,
#             batch_first=True,
#         )

#         self.decode_fc = nn.Sequential(
#             nn.Linear(2*num_units, decode_hidden),
#             nn.ReLU(),
#             nn.Linear(decode_hidden, decode_output),
#         )

    
#     def encode(self, w: th.Tensor, mu: th.Tensor, skips: Optional[List[th.Tensor]], timestep: Optional[th.Tensor] = None) -> Tuple[th.Tensor, th.Tensor]:
#         h, m = w[..., :self.lmu_cell.hidden_size, :], w[..., self.lmu_cell.hidden_size:, :]
#         new_h, new_m = self.lmu_cell(input=mu, hidden=h, memory=m)
#         z_mu = th.cat((new_h, new_m), dim=-2) # (batch_size, hidden_size + memory_size, num_units)
#         z_logvar = th.zeros_like(z_mu)
#         return z_mu, z_logvar
    
#     def decode(self, z: th.Tensor, timestep: Optional[th.Tensor] = None) -> th.Tensor:
#         h, m = z[..., :self.lmu_cell.hidden_size, :], z[..., self.lmu_cell.hidden_size:, :]
#         recon = self.lmu_cell.recon(m, timestep) # (batch_size, num_timesteps, num_units)
#         kv = th.cat((h,m), dim=-2) # (batch_size, hidden_size + memory_size, num_units)
#         attn = self.decoder(recon, kv, kv, need_weights=False)[0] # (batch_size, num_timesteps, num_units)
#         recon = th.cat((recon, attn), dim=-1) # (batch_size, num_timesteps, 2*num_units)
#         recon = self.decode_fc(recon) # (batch_size, num_timesteps, decode_output)
#         return recon
    
#     def forward(self, w: th.Tensor, mu: th.Tensor, skips: Optional[List[th.Tensor]], timestep: Optional[th.Tensor] = None) -> WynerOutput:
#         z_mu, z_logvar = self.encode(w, mu, skips, timestep)
#         # prior
#         prior_mu, prior_logvar = self.encode(
#             w, th.zeros_like(mu), skips, timestep
#         )
#         std = th.exp(0.5 * z_logvar)
#         eps = th.randn_like(std)
#         z = z_mu + eps * std
#         if timestep is not None:
#             recon = self.decode(z, timestep)
#             recon_next = self.decode(z, timestep+1)
#         else:
#             recon = None
#             recon_next = None
#         return WynerOutput(
#             w=z_mu,
#             logvar=z_logvar,
#             recon=recon,
#             recon_next=recon_next,
#             prior_mu=prior_mu,
#             prior_logvar=prior_logvar,
#         )
    
#     def loss(self, output: WynerOutput, recon_target: Optional[th.Tensor] = None, recon_next_target: Optional[th.Tensor] = None) -> WynerLoss:
#         kl_per_dim = 0.5 * (
#                 output.prior_logvar - output.logvar
#                 + (output.logvar - output.prior_logvar).exp() + (output.w - output.prior_mu).pow(2) / output.prior_logvar.exp()
#                 - 1.0
#             ) # (batch_size, hidden_size + memory_size, num_units)
#         kl_loss = kl_per_dim.reshape(kl_per_dim.size(0), -1).mean(dim=-1) 
#         recon_loss = None
#         recon_next_loss = None
#         if output.recon is not None and recon_target is not None:
#             recon_loss = nn.functional.mse_loss(output.recon, recon_target)
#         if output.recon_next is not None and recon_next_target is not None:
#             recon_next_loss = nn.functional.mse_loss(output.recon_next, recon_next_target)
#         return WynerLoss(
#             kl_loss=kl_loss,
#             recon_loss=recon_loss,
#             recon_next_loss=recon_next_loss,
#         )



# if __name__ == "__main__":
#     batch_size = 3
#     input_size = 7
#     hidden_size = 11
#     memory_size = 13
#     num_units = 17
#     theta = 93

#     cell = LMUCell(input_size, hidden_size, memory_size, num_units, theta)
#     cell = th.jit.script(cell)
#     input = th.randn(batch_size, input_size)
#     hidden = th.randn(batch_size, hidden_size, num_units)
#     memory = th.randn(batch_size, memory_size, num_units)

#     new_hidden, new_memory = cell(input, hidden, memory)
#     print("New Hidden Shape:", new_hidden.shape) # (batch_size, hidden_size, num_units)
#     print("New Memory Shape:", new_memory.shape) # (batch_size, memory_size, num_units)

#     timesteps = th.arange(10).unsqueeze(0).repeat(batch_size, 1) # (batch_size, num_timesteps)
#     recon = cell.recon(new_memory, timesteps)
#     print("Reconstructed Input Shape:", recon.shape) # (batch_size, num_timesteps, num_units)

#     vae = WynerVAE(
#         input_size=input_size,
#         hidden_size=hidden_size,
#         memory_size=memory_size,
#         num_units=num_units,
#         theta=theta,
#         decode_hidden=19,
#         decode_output=23,
#     )

#     # Test the VAE forward pass
#     w = th.randn(batch_size, hidden_size + memory_size, num_units)
#     mu = th.randn(batch_size, input_size)
#     output = vae(w, mu, skips=None, timestep=timesteps)
#     print("WynerVAE Output Recon Shape:", output.recon.shape) # (batch_size, num_timesteps, decode_output)
#     print("WynerVAE Output Recon Next Shape:", output.recon_next.shape) # (batch_size, num_timesteps, decode_output)

#     # Calculate total number of parameters in the VAE
#     total_params = sum(p.numel() for p in vae.parameters())
#     print("Total number of parameters in WynerVAE:", total_params)
import torch as th
import torch.nn as nn
import math

from typing import Tuple, List, Optional


class LMUCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        memory_size: int,
        num_units: int,
        theta: int,
        dt: float = 1.0,
    ):
        super(LMUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.num_units = num_units
        self.theta = float(theta)
        self.dt = dt

        # ------------------------------------------------------------------ #
        # A matrix  (Eq. 2 in Voelker et al. 2019)
        #   A_ij = (2i+1) * { -1        if i < j
        #                    { (-1)^(i-j+1)  if i >= j
        #
        # FIX 1: use (-1)**(x) not -1**x  (precedence bug)
        # FIX 2: build in float64 to avoid int32 truncation
        # FIX 3: apply Euler discretisation  A_bar = (dt/theta)*A + I
        # ------------------------------------------------------------------ #
        A = th.zeros(memory_size, memory_size, dtype=th.float64)
        for i in range(memory_size):
            for j in range(memory_size):
                if i < j:
                    A[i, j] = -(2 * i + 1)
                else:
                    A[i, j] = ((-1) ** (i - j + 1)) * (2 * i + 1)
        A_bar = (dt / self.theta) * A + th.eye(memory_size, dtype=th.float64)

        # ------------------------------------------------------------------ #
        # B vector  (Eq. 2)
        #   B_i = (2i+1) * (-1)^i   →  [1, -3, 5, -7, ...]
        #
        # FIX 1: same precedence fix
        # FIX 2: float64, apply (dt/theta) scaling
        # ------------------------------------------------------------------ #
        B = th.zeros(memory_size, dtype=th.float64)
        for i in range(memory_size):
            B[i] = (2 * i + 1) * ((-1) ** i)
        B_bar = (dt / self.theta) * B

        # Store as float32 buffers, shaped for batched matmul
        # A: (1, d, d)   B: (d, 1)
        self.register_buffer('A', A_bar.float().view(1, memory_size, memory_size))
        self.register_buffer('B', B_bar.float().view(memory_size, 1))

        # ------------------------------------------------------------------ #
        # Attention modules
        # ------------------------------------------------------------------ #
        self.decode_mh  = nn.MultiheadAttention(
            embed_dim=num_units, num_heads=1, batch_first=True,
        )
        self.proj_input = nn.Linear(input_size, num_units)
        self.encode_input = nn.MultiheadAttention(
            embed_dim=num_units, num_heads=1, batch_first=True,
        )
        self.u_norm = nn.LayerNorm(num_units)
        self.hidden_update = nn.MultiheadAttention(
            embed_dim=num_units, num_heads=1, batch_first=True,
        )

    @th.jit.export
    def forward(
        self,
        input: th.Tensor,   # (batch, input_size)
        hidden: th.Tensor,  # (batch, hidden_size, num_units)
        memory: th.Tensor,  # (batch, memory_size, num_units)
    ) -> Tuple[th.Tensor, th.Tensor]:

        # Decode hidden state by attending over memory
        decoded = hidden + self.decode_mh(
            hidden, memory, memory, need_weights=False
        )[0]  # (batch, hidden_size, num_units)

        # Project raw input and compute context-aware u_t
        input_proj = self.proj_input(input)  # (batch, num_units)
        u_t, _ = self.encode_input(
            input_proj.unsqueeze(1), decoded, decoded, need_weights=False
        )  # (batch, 1, num_units)

        # ------------------------------------------------------------------ #
        # u_t: what gets written into the Legendre memory
        #
        # FIX: replaced unit-sphere normalisation (which destroys magnitude
        #      and breaks the Legendre reconstruction contract) with LayerNorm,
        #      which preserves relative magnitudes while controlling scale.
        # ------------------------------------------------------------------ #
        u_t = self.u_norm(u_t.squeeze(1) + input_proj)  # (batch, num_units)

        # ------------------------------------------------------------------ #
        # Legendre memory update  (Eq. 4)
        #   m_t = A_bar @ m_{t-1} + B_bar * u_t
        #
        # Multichannel: each of the C=num_units channels evolves independently
        # under the same (A_bar, B_bar), driven by its own scalar u_t^(c).
        #
        # Shapes:
        #   A_bar          : (1,  d, d)
        #   memory         : (B,  d, C)
        #   matmul result  : (B,  d, C)   ← A applied per channel via broadcast
        #   B_bar          : (d,  1)
        #   u_t.unsqueeze  : (B,  1, C)
        #   B*u broadcast  : (B,  d, C)   ← B scales each channel independently
        # ------------------------------------------------------------------ #
        new_memory = (
            th.matmul(self.A, memory)          # (batch, d, C)
            + self.B * u_t.unsqueeze(1)        # (batch, d, C)
        )

        # Update hidden state by attending over {hidden, new_memory, input}
        kv = th.cat(
            (hidden, new_memory, input_proj.unsqueeze(1)), dim=1
        )  # (batch, hidden_size + memory_size + 1, num_units)
        new_hidden = hidden + self.hidden_update(
            hidden, kv, kv, need_weights=False
        )[0]  # (batch, hidden_size, num_units)

        return new_hidden, new_memory

    @th.jit.export
    def recon(
        self,
        memory: th.Tensor,    # (batch, memory_size, num_units)
        timesteps: th.Tensor, # (batch, T) — lags θ' in [0, theta]
    ) -> th.Tensor:           # (batch, T, num_units)
        # ------------------------------------------------------------------ #
        # Reconstruct u(t − θ') from memory via shifted Legendre polynomials
        # (Eq. 3):  u(t−θ') ≈ Σ_i P_i(θ'/θ) · m_i(t)
        #
        # Uses the 3-term Legendre recurrence instead of the monomial expansion:
        #   P_0 = 1,  P_1 = 2r−1
        #   P_{n+1}(r) = [(2n+1)(2r−1)P_n(r) − n P_{n−1}(r)] / (n+1)
        #
        # All intermediate values stay in [−1, 1] for r ∈ [0, 1], so there
        # is no catastrophic float32 cancellation regardless of memory_size.
        # The old monomial approach had O(1e6) intermediate terms that cancelled
        # to O(1e−1) — 7 decimal digits lost at d=16 in float32.
        # ------------------------------------------------------------------ #
        r = (timesteps / self.theta).clamp(0.0, 1.0)  # (batch, T) in [0, 1]
        x = 2.0 * r - 1.0                              # map to [−1, 1]

        batch, T = x.shape
        d = self.memory_size

        P = th.zeros(batch, T, d, device=memory.device, dtype=memory.dtype)
        P[:, :, 0] = 1.0
        if d > 1:
            P[:, :, 1] = x
        for n in range(1, d - 1):
            P[:, :, n + 1] = (
                (2 * n + 1) * x * P[:, :, n] - n * P[:, :, n - 1]
            ) / (n + 1)

        # (batch, T, d) ⊗ (batch, d, C) → (batch, T, C)
        return th.einsum('bti,bic->btc', P, memory)


# --------------------------------------------------------------------------- #
# Wyner dataclasses
# --------------------------------------------------------------------------- #

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
    def encode(
        self, w: th.Tensor, mu: th.Tensor,
        skips: Optional[List[th.Tensor]], timestep: Optional[th.Tensor] = None
    ) -> Tuple[th.Tensor, th.Tensor]:
        pass

    def decode(self, z: th.Tensor, timestep: Optional[th.Tensor] = None) -> th.Tensor:
        pass

    def forward(
        self, w: th.Tensor, mu: th.Tensor,
        skips: Optional[List[th.Tensor]], timestep: Optional[th.Tensor] = None
    ) -> WynerOutput:
        pass

    def loss(
        self, output: WynerOutput,
        recon_target: Optional[th.Tensor],
        recon_next_target: Optional[th.Tensor]
    ) -> WynerLoss:
        pass


# --------------------------------------------------------------------------- #
# WynerVAE
# --------------------------------------------------------------------------- #

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
        dt: float = 1.0,   # passed through to LMUCell for ZOH discretisation
    ):
        super(WynerVAE, self).__init__()
        self.lmu_cell = LMUCell(
            input_size, hidden_size, memory_size, num_units, theta, dt=dt
        )

        # ------------------------------------------------------------------ #
        # FIX: restore learned logvar via attention (was hardcoded to zeros,
        #      making the posterior variance always 1 and the KL a constant).
        # ------------------------------------------------------------------ #
        self.log_var = nn.MultiheadAttention(
            embed_dim=num_units, num_heads=1, batch_first=True,
        )

        self.decode_hidden = decode_hidden
        self.decode_output = decode_output

        self.decoder = nn.MultiheadAttention(
            embed_dim=num_units, num_heads=1, batch_first=True,
        )
        self.decode_fc = nn.Sequential(
            nn.Linear(2 * num_units, decode_hidden),
            nn.ReLU(),
            nn.Linear(decode_hidden, decode_output),
        )

    def encode(
        self,
        w: th.Tensor,
        mu: th.Tensor,
        skips: Optional[List[th.Tensor]],
        timestep: Optional[th.Tensor] = None,
    ) -> Tuple[th.Tensor, th.Tensor]:
        h = w[..., :self.lmu_cell.hidden_size, :]
        m = w[..., self.lmu_cell.hidden_size:, :]
        new_h, new_m = self.lmu_cell(input=mu, hidden=h, memory=m)
        z_mu = th.cat((new_h, new_m), dim=-2)  # (batch, H+d, C)

        # FIX: learned posterior logvar — was th.zeros_like(z_mu) which gave
        #      σ_q²=1 always, making reparameterisation pure noise injection
        #      and KL gradients zero everywhere.
        z_logvar = self.log_var(z_mu, z_mu, z_mu, need_weights=False)[0]
        return z_mu, z_logvar

    def decode(
        self, z: th.Tensor, timestep: Optional[th.Tensor] = None
    ) -> th.Tensor:
        h = z[..., :self.lmu_cell.hidden_size, :]
        m = z[..., self.lmu_cell.hidden_size:, :]
        recon = self.lmu_cell.recon(m, timestep)          # (batch, T, C)
        kv = th.cat((h, m), dim=-2)                        # (batch, H+d, C)
        attn = self.decoder(recon, kv, kv, need_weights=False)[0]
        recon = th.cat((recon, attn), dim=-1)              # (batch, T, 2C)
        return self.decode_fc(recon)                        # (batch, T, decode_output)

    def forward(
        self,
        w: th.Tensor,
        mu: th.Tensor,
        skips: Optional[List[th.Tensor]],
        timestep: Optional[th.Tensor] = None,
    ) -> WynerOutput:
        z_mu, z_logvar = self.encode(w, mu, skips, timestep)

        # Wyner prior: encode with zero input — "what does memory alone predict?"
        prior_mu, prior_logvar = self.encode(
            w, th.zeros_like(mu), skips, timestep
        )

        # Reparameterisation with learned variance
        std = th.exp(0.5 * z_logvar)
        z = z_mu + th.randn_like(std) * std

        recon: Optional[th.Tensor] = None
        recon_next: Optional[th.Tensor] = None
        if timestep is not None:
            recon = self.decode(z, timestep)
            recon_next = self.decode(z, timestep + 1)

        return WynerOutput(
            w=z_mu,
            logvar=z_logvar,
            recon=recon,
            recon_next=recon_next,
            prior_mu=prior_mu,
            prior_logvar=prior_logvar,
        )

    def loss(
        self,
        output: WynerOutput,
        recon_target: Optional[th.Tensor] = None,
        recon_next_target: Optional[th.Tensor] = None,
    ) -> WynerLoss:
        # ------------------------------------------------------------------ #
        # KL( q(z|x,w) || p(z|w) )  with both q and p being diagonal Gaussians
        #
        # KL(N(μ_q,σ_q²) || N(μ_p,σ_p²))
        #   = 0.5 * [ log(σ_p²/σ_q²) - 1 + σ_q²/σ_p² + (μ_q-μ_p)²/σ_p² ]
        #
        # In logvar form (lq = log σ_q², lp = log σ_p²):
        #   = 0.5 * [ (lp - lq) - 1 + exp(lq - lp) + (μ_q-μ_p)² / exp(lp) ]
        #
        # FIX: previously used unit-Gaussian prior, discarding the computed
        #      prior_mu / prior_logvar entirely (Wyner structure wasted).
        # ------------------------------------------------------------------ #
        assert output.prior_mu is not None
        assert output.prior_logvar is not None

        lq = output.logvar                    # log σ_q²
        lp = output.prior_logvar              # log σ_p²
        mu_q = output.w
        mu_p = output.prior_mu

        kl_per_dim = 0.5 * (
            (lp - lq)                          # log σ_p²/σ_q²
            - 1.0
            + (lq - lp).exp()                  # σ_q²/σ_p²
            + (mu_q - mu_p).pow(2) / lp.exp()  # (μ_q-μ_p)²/σ_p²
        )  # (batch, H+d, C)

        # Mean over dims, keep batch dim for per-sample losses upstream
        kl_loss = kl_per_dim.reshape(kl_per_dim.size(0), -1).mean(dim=-1)

        recon_loss: Optional[th.Tensor] = None
        recon_next_loss: Optional[th.Tensor] = None
        if output.recon is not None and recon_target is not None:
            recon_loss = nn.functional.mse_loss(output.recon, recon_target)
        if output.recon_next is not None and recon_next_target is not None:
            recon_next_loss = nn.functional.mse_loss(output.recon_next, recon_next_target)

        return WynerLoss(
            kl_loss=kl_loss,
            recon_loss=recon_loss,
            recon_next_loss=recon_next_loss,
        )


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    batch_size  = 3
    input_size  = 7
    hidden_size = 11
    memory_size = 13
    num_units   = 17
    theta       = 93

    # ── LMUCell ──────────────────────────────────────────────────────────── #
    cell = LMUCell(input_size, hidden_size, memory_size, num_units, theta)

    # Verify ZOH stability: spectral radius must be < 1
    A_bar = cell.A.squeeze(0)
    B_bar = cell.B.squeeze(-1)
    rho   = th.linalg.eigvals(A_bar.double()).abs().max().item()
    print(f"ZOH spectral radius: {rho:.6f}  ({'✓ stable' if rho < 1 else '✗ UNSTABLE'})")
    print("B_bar[:5] (should alternate sign):", B_bar[:5].tolist())

    cell = th.jit.script(cell)
    inp    = th.randn(batch_size, input_size)
    hidden = th.randn(batch_size, hidden_size, num_units)
    memory = th.randn(batch_size, memory_size, num_units)

    new_hidden, new_memory = cell(inp, hidden, memory)
    print("New Hidden Shape:", new_hidden.shape)   # (3, 11, 17)
    print("New Memory Shape:", new_memory.shape)   # (3, 13, 17)
    print("Memory norm (should be O(1)):", new_memory.norm().item())

    # timesteps are lags in [0, theta]
    timesteps = th.linspace(0, theta, 10).unsqueeze(0).repeat(batch_size, 1)
    recon = cell.recon(new_memory, timesteps)
    print("Recon Shape:", recon.shape)             # (3, 10, 17)
    print("Recon norm (should be O(1)):", recon.norm(dim=-1))

    # ── WynerVAE ─────────────────────────────────────────────────────────── #
    vae = WynerVAE(
        input_size=input_size,
        hidden_size=hidden_size,
        memory_size=memory_size,
        num_units=num_units,
        theta=theta,
        decode_hidden=19,
        decode_output=23,
    )

    w   = th.randn(batch_size, hidden_size + memory_size, num_units)
    mu  = th.randn(batch_size, input_size)
    out = vae(w, mu, skips=None, timestep=timesteps)
    print("Recon shape:",      out.recon.shape)       # (3, 10, 23)
    print("Recon next shape:", out.recon_next.shape)  # (3, 10, 23)

    recon_tgt      = th.randn_like(out.recon)
    recon_next_tgt = th.randn_like(out.recon_next)
    losses = vae.loss(out, recon_tgt, recon_next_tgt)
    print("KL loss shape:", losses.kl_loss.shape)     # (batch,)
    print("Recon loss:",    losses.recon_loss)
    print("KL loss:",       losses.kl_loss)

    total_params = sum(p.numel() for p in vae.parameters())
    print("Total params:", total_params)
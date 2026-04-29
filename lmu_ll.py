import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import cont2discrete
from typing import Literal, Tuple

from torch.nn.utils import spectral_norm

class OrthoLayer(nn.Module):
    """
    Orthogonal linear layer (no bias) maintained via Cayley-map Riemannian updates.

    No bias is required: W_pre(0) = 0 exactly, which makes r_intr = ||u_actual - pred||
    equal to ||W_pre(innovation)||₂ = ||innovation||₂ (isometry). A bias would add
    a constant offset and break this equality.

    CRITICAL: exclude from main Adam optimizer. See module docstring.
    """

    def __init__(self, size: int):
        super().__init__()
        self.weights = nn.Parameter(th.eye(size))

    def forward(self, x: th.Tensor) -> th.Tensor:
        return x @ self.weights   # (B, C) @ (C, C) → (B, C)

    def ortho_update(self, lr: float) -> None:
        with th.no_grad():
            if self.weights.grad is None:
                return
            G, W = self.weights.grad, self.weights
            A = G @ W.t() - W @ G.t()
            I = th.eye(W.size(0), device=W.device, dtype=W.dtype)
            W_new = th.linalg.solve(I + lr * A, (I - lr * A) @ W)
            if not (th.isnan(W_new).any() or th.isinf(W_new).any()):
                self.weights.copy_(W_new)
            self.weights.grad.zero_()

    @th.no_grad()
    def reorthogonalize(self) -> None:
        if th.isnan(self.weights).any() or th.isinf(self.weights).any():
            nn.init.eye_(self.weights)
            return
        U, _, Vh = th.linalg.svd(self.weights, full_matrices=False)
        self.weights.copy_(U @ Vh)

    @th.no_grad()
    def orthogonality_error(self) -> float:
        I = th.eye(self.weights.size(0), device=self.weights.device)
        return (self.weights.t() @ self.weights - I).norm().item()



def get_AB(d: int, theta: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build HiPPO-LegT (A, B) from Voelker 2019 Eq. 2, then ZOH-discretize.
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


class LMUCell(nn.Module):
    """
    One step of the multichannel LMU - HiPPO-LegT backbone
    This will be used recursively inside LMUMemory to process sequences.

    Args:
        input_size:      p - input size
        hidden_size:     n - hidden state dimension
        memory_size:     d - Legendre polynomial order
        theta:           sliding window length (LegT)
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        memory_size: int,
        theta: int,
        batch_size: int = 1,  # for preallocating memory buffers
    ):
        super().__init__()
        self.input_size     = input_size
        self.hidden_size    = hidden_size
        self.memory_size    = memory_size
        A, B = get_AB(self.memory_size, float(theta))
        self.register_buffer('A', th.from_numpy(A))  # (d, d)
        self.register_buffer('B', th.from_numpy(B))  # (d, 1)

        # Channel to handle input
        self.e_x = nn.Parameter(th.empty(input_size))
        # Channel to handle hidden state
        self.e_h = spectral_norm(nn.Linear(hidden_size, input_size, bias=False))
        # Channel to handle memory state
        self.e_m = nn.Parameter(th.zeros(memory_size))

        # Ortho weight for gating innovation
        self.W_pre = OrthoLayer(self.hidden_size)

        # Dynamic read
        self.W_query = nn.Linear(hidden_size, memory_size, bias=False)
        nn.init.orthogonal_(self.W_query.weight, gain=0.01)

        self.W_x = nn.Linear(hidden_size,  hidden_size, bias=True)
        self.W_h = spectral_norm(nn.Linear(hidden_size, hidden_size, bias=False))
        self.W_m = nn.Linear(hidden_size,  hidden_size, bias=False)

        self.step_count = 0
        self.batch_size = batch_size

        self.register_buffer('m', th.zeros(self.batch_size, self.memory_size, self.input_size))
        self.register_buffer('h', th.zeros(self.batch_size, self.hidden_size))

        self.flush_point = self

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_normal_(self.e_x.weight)
        nn.init.xavier_normal_(self.e_h.weight)
        # e_m stays zero: silent write at episode start
        for layer in (self.W_x, self.W_h, self.W_m):
            nn.init.xavier_normal_(layer.weight)
        nn.init.zeros_(self.W_x.bias)
        self.step_count = 0
        self.m.zero_()
        self.h.zero_()
    
    def ortho_params(self):
        return self.W_pre.parameters()
    
    def set_flush_point(self, flush_point: LMUCell) -> None:
        """
        Use this to flush to another LMUCell
        If not set, will flush to self
        """
        assert self.batch_size % flush_point.batch_size == 0, "Flush point batch size must divide cell batch size"
        self.flush_point = flush_point
    
    def _compute_write(
        self,
        u_x: th.Tensor,  # (B, p)
        u_h: th.Tensor,  # (B, p)
        u_m: th.Tensor,  # (B, p)
    ) -> th.Tensor:  # (B, p)
        pred = u_h + u_m  # (B, p)
        gate = u_x / (1+u_x.abs())  # (B, p)  # softsign nonlinearity
        innov = (u_x - pred) / (1 + (u_x-pred).abs()) # (B, p)  # softsign innovation
        return pred + self.W_pre(gate + innov)  # (B, p)
    
    @th.no_grad()
    def flush_memory(
        self,
    ) -> None:
        self.step_count = 0
        m = self.m.view(self.flush_point.batch_size, -1, self.input_size)  # (flush_B, k, p)
        for i in range(m.size(1)):
            self.flush_point.forward_eval(
                m[:, i, :]
            )
    
    def flush_forward(
        self,
        m_oth: th.Tensor,  # (B', d, p)
        h_self: th.Tensor, # (B, n)
        m_self: th.Tensor, # (B, d, p)
    ) -> Tuple[th.Tensor, th.Tensor]: # (B, B' x d / B, n), (B, d, p)
        """
        Primarily to estimate gradients from flushes on each step
        Flushes are rare and thus there will be very few updates

        Estimation process:
        Assume that a flush happend at this step

        Compute the final memory state as:

        m_new = A^k m_self + sum_{i=1}^k A^{k-i} B u_i

        where gradients only flow through u_i = f(m_oth[:, i, :], h_self)
        and not through m_self or h_self (detach)

        Stack all intermediate h states for an LMU cell that might
        be in series with the flush point

        """
        m_oth = m_oth.view(self.batch_size, -1, self.input_size)  # (B, k, p)
        m_new = m_self.clone()
        h_new = h_self.clone()
        m_out = m_self.clone()
        usum = th.zeros_like(m_self)
        h_out = th.zeros(self.batch_size, m_oth.size(1), self.hidden_size, device=h_self.device)
        for i in range(m_oth.size(1)):
            # Standard LMU write with m_oth[:, i, :] as input instead of x
            u_x = m_oth[:, i, :] * F.normalize(self.e_x, dim=0)  # (B, p)
            u_h = self.e_h(h_new) # (B, p)
            u_m = (self.e_m.view(1, -1, 1) * m_new.detach()).sum(dim=1)  # (B, p)
            u_actual = self._compute_write(u_x, u_h, u_m)  # (B, p)

            Bu = self.B * u_actual.unsqueeze(1)  # (B, d, p)

            m_out = th.einsum('ij,bjc->bic', self.A, m_out) # (B, d, p)
            usum = th.einsum('ij,bjc->bic', self.A, usum) + Bu  # (B, d, p)

            Am = th.einsum('ij,bjc->bic', self.A, m_new.detach())     # (B, d, p)
            m_new = Am + Bu.detach()  # (B, d, p) -- detach to prevent gradients from accumulating across flush steps

            h_new = h_new.detach()  # prevent gradients from accumulating across flush steps
            C_t = F.normalize(self.W_query(h_new), dim=-1) # (B, d)  # cosine similarity read
            y = th.einsum('bd,bdc->bc', C_t, m_new) # (B, p)
            h_new = self.W_x(m_oth[:, i, :]) + self.W_h(h_new) + self.W_m(y)  # (B, n)
            h_out[:, i, :].copy_(h_new)
        m_out = m_out + usum  # (B, d, p)  # add the contributions from all flush steps
        return h_out, m_out
    
    def update(
        self,
        x: th.Tensor,       # (B, n)
        h_prev: th.Tensor,  # (B, n)
        m_prev: th.Tensor,  # (B, d, p)        
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        u_x = x * F.normalize(self.e_x, dim=0)  # (B, p)
        u_h = self.e_h(h_prev) # (B, p)
        u_m = (self.e_m.view(1, -1, 1) * m_prev).sum(dim=1)  # (B, p)

        u_actual = self._compute_write(u_x, u_h, u_m)  # (B, p)
        u_null = self._compute_write(th.zeros_like(u_x), u_h, u_m)  # (B, p)

        r_intr = u_actual - u_null  # (B, p)

        Am = th.einsum('ij,bjc->bic', self.A, m_prev)     # (B, d, p)
        Bu = self.B * u_actual.unsqueeze(1)  # (B, d, p)

        m_new = Am + Bu  # (B, d, p)
        C_t = F.normalize(self.W_query(h_prev), dim=-1) # (B, d)  # cosine similarity read
        y = th.einsum('bd,bdc->bc', C_t, m_new) # (B, p)

        h_new = self.W_x(x) + self.W_h(h_prev) + self.W_m(y)  # (B, n)
        return h_new, m_new, r_intr
        
                
    @th.no_grad()
    def forward_eval(
        self,
        x: th.Tensor, # (B, n)
    ) -> th.Tensor: # (B, p)
        if self.step_count == self.theta:
            self.flush_memory()
        self.step_count += 1
        h, m, r_intr = self.update(x, self.h, self.m)
        self.h.copy_(h)
        self.m.copy_(m)
        return r_intr

class LMUMemory(nn.Module):
    """

    """
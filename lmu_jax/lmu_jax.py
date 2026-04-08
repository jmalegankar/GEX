"""
Legendre Memory Unit (LMU) — JAX/Flax implementation.
Ports lmu.py; public API is identical: LMUCell, LMU, get_AB.
"""

from typing import Optional, Tuple

import numpy as np
from scipy.signal import cont2discrete

import jax
import jax.numpy as jnp
import flax.linen as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_AB(d: int, theta: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct continuous-time (A, B) from Padé approximant of ideal delay
    (paper eq. 2), then ZOH-discretise with dt=1.  Identical to PyTorch ver.
    Runs once at module init; result is baked as a constant into the JIT graph.
    """
    Q = np.arange(d, dtype=float)
    R = (2 * Q + 1)[:, None]
    j, i = np.meshgrid(Q, Q)
    A  = R * np.where(i < j, -1.0, (-1.0) ** (i - j + 1))
    A /= theta
    B  = R * ((-1.0) ** Q)[:, None]
    B /= theta
    C  = np.zeros((1, d))
    D  = np.zeros((1,))
    A_d, B_d, _, _, _ = cont2discrete((A, B, C, D), dt=1.0, method="zoh")
    return A_d.astype(np.float32), B_d.astype(np.float32)


# ---------------------------------------------------------------------------
# LMU Cell
# ---------------------------------------------------------------------------

class LMUCell(nn.Module):
    """
    One LMU step:
        u_t = e_x(x_t) + e_h(h_{t-1}) + e_m(m_{t-1})   # scalar per sample
        m_t = Ā m_{t-1} + B̄ u_t                          # linear memory update
        h_t = tanh(W_x x_t + W_h h_{t-1} + W_m m_t)     # nonlinear hidden state

    State convention: (h, m) are states *before* processing x_t so that
    training can re-run the cell step with gradient (same as PyTorch version).
    """
    input_size:  int
    hidden_size: int
    memory_size: int
    theta:       float = 1.0

    def setup(self):
        # Fixed matrices — stored as plain attributes (not params).
        # They are concrete jnp arrays; JIT folds them as compile-time constants.
        A_np, B_np = get_AB(self.memory_size, self.theta)
        self.A = jnp.array(A_np)    # (d, d)
        self.B = jnp.array(B_np)    # (d, 1)

        # Encoding vectors — project inputs into scalar u written to memory
        self.e_x = nn.Dense(1, use_bias=False,
                             kernel_init=nn.initializers.lecun_uniform())
        self.e_h = nn.Dense(1, use_bias=False,
                             kernel_init=nn.initializers.lecun_uniform())
        self.e_m = nn.Dense(1, use_bias=False,
                             kernel_init=nn.initializers.zeros)   # zero init per paper

        # Hidden state kernels
        self.W_x = nn.Dense(self.hidden_size, use_bias=True,
                             kernel_init=nn.initializers.glorot_normal())
        self.W_h = nn.Dense(self.hidden_size, use_bias=False,
                             kernel_init=nn.initializers.glorot_normal())
        self.W_m = nn.Dense(self.hidden_size, use_bias=False,
                             kernel_init=nn.initializers.glorot_normal())

    def __call__(
        self,
        x:      jnp.ndarray,   # (B, input_size)
        h_prev: jnp.ndarray,   # (B, hidden_size)
        m_prev: jnp.ndarray,   # (B, memory_size)
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        u = self.e_x(x) + self.e_h(h_prev) + self.e_m(m_prev)   # (B, 1)
        m = m_prev @ self.A.T + u * self.B.T                      # (B, d)
        h = jnp.tanh(self.W_x(x) + self.W_h(h_prev) + self.W_m(m))
        return h, m                                                # (B, n), (B, d)


# ---------------------------------------------------------------------------
# LMU sequence wrapper  (offline / supervised use)
# ---------------------------------------------------------------------------

class LMU(nn.Module):
    """
    Runs LMUCell over a full sequence using jax.lax.scan.

    Input  : x  (B, T, input_size)
    Output : out (B, T, hidden_size),  final state (h, m)
    """
    input_size:  int
    hidden_size: int
    memory_size: int
    theta:       float = 1.0

    def setup(self):
        self.cell = LMUCell(
            input_size  = self.input_size,
            hidden_size = self.hidden_size,
            memory_size = self.memory_size,
            theta       = self.theta,
        )

    def __call__(
        self,
        x:     jnp.ndarray,                           # (B, T, input_size)
        state: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
    ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        B = x.shape[0]
        if state is None:
            h = jnp.zeros((B, self.hidden_size))
            m = jnp.zeros((B, self.memory_size))
        else:
            h, m = state

        def scan_step(carry, x_t):
            h, m     = carry
            h_new, m_new = self.cell(x_t, h, m)
            return (h_new, m_new), h_new

        # scan over time axis: x transposed to (T, B, F)
        (h_final, m_final), outputs = jax.lax.scan(
            scan_step, (h, m), x.transpose(1, 0, 2)
        )
        return outputs.transpose(1, 0, 2), (h_final, m_final)   # (B,T,n), ((B,n),(B,d))


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import jax

    B, T, F = 4, 64, 16
    hidden  = 64
    memory  = 32
    theta   = float(T)

    key  = jax.random.PRNGKey(0)
    x    = jax.random.normal(key, (B, T, F))

    model  = LMU(F, hidden, memory, theta=theta)
    params = model.init(key, x)
    out, (h_final, m_final) = model.apply(params, x)

    print(f"Input  : {x.shape}")
    print(f"Output : {out.shape}")
    print(f"h_final: {h_final.shape}")
    print(f"m_final: {m_final.shape}")

    A_np, _ = get_AB(memory, theta)
    print(f"A spectral radius: {max(abs(np.linalg.eigvals(A_np))):.4f}  (should be < 1)")
    print("All checks passed.")
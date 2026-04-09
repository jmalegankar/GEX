"""
Minimal sanity-check: does the JAX LMUCell retain a signal injected at t=0
through T steps, and does the SB3 (PyTorch) LMUCell produce the same m_T?

Run from the repo root:
    python scripts/probe_lmu_retention_jax.py

Expected output (both cells identical, signal retained):
  JAX   m_T norm: ~0.5–3.0   (not zero)
  Torch m_T norm: matches JAX to < 1e-4
  Cosine similarity JAX vs Torch: > 0.999
  Signal retained:  diff in m_T between signal vs noise episode > 0.1
"""

import numpy as np
import jax
import jax.numpy as jnp
import torch

# --- local imports -----------------------------------------------------------
import sys; sys.path.insert(0, ".")
from lmu_jax.lmu_jax import LMUCell as JaxLMUCell, get_AB
from lmu import LMUCell as TorchLMUCell          # original pytorch lmu.py


# ---------------------------------------------------------------------------
# Config — must match train_jax.py defaults
# ---------------------------------------------------------------------------
INPUT_SIZE  = 64        # encoder_dim
HIDDEN_SIZE = 64
MEMORY_SIZE = 32
THETA       = 50.0      # MemoryS7
T           = 50        # episode length


# ---------------------------------------------------------------------------
# 1.  Check A/B matrices are numerically identical
# ---------------------------------------------------------------------------
A_np, B_np = get_AB(MEMORY_SIZE, THETA)
A_torch = torch.from_numpy(A_np)
B_torch = torch.from_numpy(B_np)

print(f"A spectral radius : {max(abs(np.linalg.eigvals(A_np))):.6f}  (must be < 1)")
print(f"A norm            : {np.linalg.norm(A_np):.6f}")
print(f"B norm            : {np.linalg.norm(B_np):.6f}")


# ---------------------------------------------------------------------------
# 2.  Forward-pass both cells with IDENTICAL zero weights, compare m_T
# ---------------------------------------------------------------------------

key = jax.random.PRNGKey(42)
batch = 4

# -- JAX cell init --
jax_cell = JaxLMUCell(INPUT_SIZE, HIDDEN_SIZE, MEMORY_SIZE, THETA)
dummy_x  = jnp.zeros((batch, INPUT_SIZE))
dummy_h  = jnp.zeros((batch, HIDDEN_SIZE))
dummy_m  = jnp.zeros((batch, MEMORY_SIZE))
jax_params = jax_cell.init(key, dummy_x, dummy_h, dummy_m)

# -- PyTorch cell init -- (re-use same random weights via numpy)
torch_cell = TorchLMUCell(INPUT_SIZE, HIDDEN_SIZE, MEMORY_SIZE, THETA)

# Copy JAX Dense weights → PyTorch Linear weights
def copy_dense_to_linear(jax_kernel, pt_layer):
    """JAX Dense kernel is (in, out); PyTorch Linear weight is (out, in)."""
    w = np.array(jax_kernel)
    pt_layer.weight.data = torch.from_numpy(w.T)
    if pt_layer.bias is not None:
        pt_layer.bias.data.zero_()

copy_dense_to_linear(jax_params["params"]["e_x"]["kernel"], torch_cell.e_x)
copy_dense_to_linear(jax_params["params"]["e_h"]["kernel"], torch_cell.e_h)
copy_dense_to_linear(jax_params["params"]["e_m"]["kernel"], torch_cell.e_m)
copy_dense_to_linear(jax_params["params"]["W_x"]["kernel"], torch_cell.W_x)
copy_dense_to_linear(jax_params["params"]["W_h"]["kernel"], torch_cell.W_h)
copy_dense_to_linear(jax_params["params"]["W_m"]["kernel"], torch_cell.W_m)
# W_x bias
torch_cell.W_x.bias.data = torch.from_numpy(
    np.array(jax_params["params"]["W_x"]["bias"])
)

print("\n--- Running T={} steps with identical weights ---".format(T))

# Input sequence: signal at t=0, noise afterwards
key, sk = jax.random.split(key)
signal_x   = jax.random.normal(sk, (INPUT_SIZE,)) * 2.0   # strong signal
noise_seq  = jax.random.normal(key, (T, INPUT_SIZE))

def run_jax(x_seq):
    h = jnp.zeros((1, HIDDEN_SIZE))
    m = jnp.zeros((1, MEMORY_SIZE))
    for t in range(T):
        x_t  = x_seq[t:t+1]
        h, m = jax_cell.apply(jax_params, x_t, h, m)
    return m

def run_torch(x_seq):
    h = torch.zeros(1, HIDDEN_SIZE)
    m = torch.zeros(1, MEMORY_SIZE)
    with torch.no_grad():
        for t in range(T):
            x_t  = torch.from_numpy(np.array(x_seq[t:t+1]))
            h, m = torch_cell(x_t, h, m)
    return m

# Episode with signal at t=0
signal_seq    = noise_seq.at[0].set(signal_x)
# Episode with different signal at t=0 (simulates ball vs triangle)
nosignal_seq  = noise_seq.at[0].set(-signal_x)

m_jax_sig   = run_jax(signal_seq)
m_jax_nosig = run_jax(nosignal_seq)
m_torch_sig = run_torch(signal_seq)

m_jax_sig_np   = np.array(m_jax_sig[0])
m_jax_nosig_np = np.array(m_jax_nosig[0])
m_torch_sig_np = np.array(m_torch_sig[0])

print(f"\nJAX   m_T norm  (signal ep)  : {np.linalg.norm(m_jax_sig_np):.4f}")
print(f"Torch m_T norm  (signal ep)  : {np.linalg.norm(m_torch_sig_np):.4f}")

cos_sim = (
    np.dot(m_jax_sig_np, m_torch_sig_np)
    / (np.linalg.norm(m_jax_sig_np) * np.linalg.norm(m_torch_sig_np) + 1e-8)
)
max_diff = np.max(np.abs(m_jax_sig_np - m_torch_sig_np))
print(f"Cosine sim JAX vs Torch      : {cos_sim:.6f}  (should be > 0.999)")
print(f"Max abs diff JAX vs Torch    : {max_diff:.2e}  (should be < 1e-4)")

signal_diff = np.linalg.norm(m_jax_sig_np - m_jax_nosig_np)
print(f"\nSignal retention (‖m_T_sig − m_T_nosig‖): {signal_diff:.4f}")
print(f"  -> {'PASS: signal distinguishable in m_T' if signal_diff > 0.1 else 'FAIL: m_T indistinguishable — memory not retaining signal'}")

if cos_sim < 0.999:
    print("\n*** WARNING: JAX and Torch LMU cells produce DIFFERENT outputs for")
    print("    identical weights and inputs. There is a numerical bug in the JAX port. ***")
else:
    print("\nJAX and Torch LMU cells are numerically identical. ✓")
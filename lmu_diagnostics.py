"""
LMUCell isolation diagnostic.

Key finding: Euler discretization is UNSTABLE for large d (memory_size).
The LMU continuous-time A matrix has eigenvalues with large imaginary parts
that grow with d. Euler stability requires |1 + (dt/θ)μ| < 1 for all
eigenvalues μ of A — this fails for d ≥ ~12 at θ=50, dt=1.

Fix: ZOH (zero-order hold) via matrix exponential, which is unconditionally
stable for any Hurwitz A, regardless of dt/θ.

Tests:
  1. Euler vs ZOH spectral radius across d values  (the smoking gun)
  2. Pure LMU: Euler vs ZOH memory norms + old vs new recon accuracy
  3. Full LMUCell with ZOH: u_t stability, memory norm, recon old vs new
"""

import torch as th
import torch.nn as nn
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────── #
#  Core: continuous A, B and both discretisations
# ─────────────────────────────────────────────────────────────────────────── #

def build_continuous_AB(d: int) -> Tuple[th.Tensor, th.Tensor]:
    """Continuous-time LMU state matrices (Eq. 2, Voelker 2019). float64."""
    A = th.zeros(d, d, dtype=th.float64)
    for i in range(d):
        for j in range(d):
            A[i, j] = -(2*i+1) if i < j else ((-1)**(i-j+1))*(2*i+1)
    B = th.tensor([(2*i+1)*((-1)**i) for i in range(d)], dtype=th.float64)
    return A, B


def discretize_euler(A: th.Tensor, B: th.Tensor,
                     dt: float, theta: float) -> Tuple[th.Tensor, th.Tensor]:
    """
    Euler: A_bar = I + (dt/θ)A,  B_bar = (dt/θ)B
    Conditionally stable — fails when |1 + (dt/θ)μ_k| > 1 for some eigenvalue μ_k of A.
    """
    s = dt / theta
    return th.eye(A.shape[0], dtype=th.float64) + s * A, s * B


def discretize_zoh(A: th.Tensor, B: th.Tensor,
                   dt: float, theta: float) -> Tuple[th.Tensor, th.Tensor]:
    """
    ZOH: A_bar = expm((dt/θ)A),  B_bar = A⁻¹(A_bar − I)B
    Unconditionally stable for Hurwitz A, any dt/θ.
    """
    s = dt / theta
    A_bar = th.linalg.matrix_exp(s * A)                                  # (d, d)
    rhs   = (A_bar - th.eye(A.shape[0], dtype=th.float64)) @ B.unsqueeze(-1)  # (d,1)
    B_bar = th.linalg.solve(A, rhs).squeeze(-1)                          # (d,)
    return A_bar, B_bar


# ─────────────────────────────────────────────────────────────────────────── #
#  Recon: old (monomial + .frac bug) vs new (Legendre recurrence)
# ─────────────────────────────────────────────────────────────────────────── #

def build_P_buffer(d: int) -> th.Tensor:
    P = th.zeros(d, d, dtype=th.float32)
    for i in range(d):
        neg = (-1)**(i % 2)
        for j in range(i+1):
            P[i, j] = neg * math.comb(i,j) * math.comb(i+j,j) * (0.5**j)
    return P


def recon_old(memory: th.Tensor, timesteps: th.Tensor,
              theta: float, P_buf: th.Tensor) -> th.Tensor:
    """Original — .frac() wraps lags > theta; float32 monomial catastrophic cancellation."""
    r = -2 * (timesteps / theta).frac()
    r = r.float().unsqueeze(2).repeat(1, 1, P_buf.shape[0])
    r = r ** th.arange(P_buf.shape[0], device=memory.device).view(1, 1, -1)
    P = (P_buf.view(1, 1, *P_buf.shape) * r.unsqueeze(2)).sum(dim=-1)
    return (P.unsqueeze(-1) * memory.unsqueeze(1)).sum(dim=2)


def recon_new(memory: th.Tensor, timesteps: th.Tensor, theta: float) -> th.Tensor:
    """3-term Legendre recurrence — all intermediates in [−1,1], no cancellation."""
    r = (timesteps / theta).clamp(0.0, 1.0)
    x = 2.0 * r - 1.0
    B, T, d = x.shape[0], x.shape[1], memory.shape[1]
    P = th.zeros(B, T, d, device=memory.device, dtype=memory.dtype)
    P[:, :, 0] = 1.0
    if d > 1:
        P[:, :, 1] = x
    for n in range(1, d - 1):
        P[:, :, n+1] = ((2*n+1)*x*P[:,:,n] - n*P[:,:,n-1]) / (n+1)
    return th.einsum('bti,bic->btc', P, memory)


# ─────────────────────────────────────────────────────────────────────────── #
#  LMUCell — ZOH by default
# ─────────────────────────────────────────────────────────────────────────── #

class LMUCell(nn.Module):
    def __init__(self, input_size, hidden_size, memory_size, num_units,
                 theta, dt=1.0, use_zoh=True):
        super().__init__()
        self.input_size = input_size; self.hidden_size = hidden_size
        self.memory_size = memory_size; self.num_units = num_units
        self.theta = float(theta)

        A, B = build_continuous_AB(memory_size)
        A_bar, B_bar = (discretize_zoh(A, B, dt, theta) if use_zoh
                        else discretize_euler(A, B, dt, theta))

        self.register_buffer('A',     A_bar.float().view(1, memory_size, memory_size))
        self.register_buffer('B',     B_bar.float().view(memory_size, 1))
        self.register_buffer('P_buf', build_P_buffer(memory_size))

        self.decode_mh    = nn.MultiheadAttention(num_units, 1, batch_first=True)
        self.proj_input   = nn.Linear(input_size, num_units)
        self.encode_input = nn.MultiheadAttention(num_units, 1, batch_first=True)
        self.u_norm       = nn.LayerNorm(num_units)
        self.hidden_update= nn.MultiheadAttention(num_units, 1, batch_first=True)

    def forward(self, input, hidden, memory) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        decoded    = hidden + self.decode_mh(hidden, memory, memory, need_weights=False)[0]
        input_proj = self.proj_input(input)
        u_raw, _   = self.encode_input(input_proj.unsqueeze(1), decoded, decoded,
                                        need_weights=False)
        u_t        = self.u_norm(u_raw.squeeze(1) + input_proj)
        new_memory = th.matmul(self.A, memory) + self.B * u_t.unsqueeze(1)
        kv         = th.cat((hidden, new_memory, input_proj.unsqueeze(1)), dim=1)
        new_hidden = hidden + self.hidden_update(hidden, kv, kv, need_weights=False)[0]
        return new_hidden, new_memory, u_t

    def recon(self, memory, timesteps):
        return recon_new(memory, timesteps, self.theta)

    def recon_old(self, memory, timesteps):
        return recon_old(memory, timesteps, self.theta, self.P_buf)


# ─────────────────────────────────────────────────────────────────────────── #
#  Test 1 — Spectral radius: Euler vs ZOH across d values
#
#  This is the smoking gun. Euler becomes unstable (ρ > 1) around d=12-14
#  for theta=50, dt=1. ZOH stays below 1 at every d.
# ─────────────────────────────────────────────────────────────────────────── #

def test_spectral_radius(theta=50, dt=1.0, d_values=None):
    if d_values is None:
        d_values = [4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 32]
    print("\n── Test 1: Euler vs ZOH spectral radius across d ──")
    print(f"  {'d':>4}  {'ρ_euler':>10}  {'ρ_zoh':>10}  {'euler_stable':>14}")

    rho_euler_list, rho_zoh_list = [], []
    for d in d_values:
        A, B = build_continuous_AB(d)
        A_e, _ = discretize_euler(A, B, dt, theta)
        A_z, _ = discretize_zoh(A, B, dt, theta)
        rho_e = th.linalg.eigvals(A_e).abs().max().item()
        rho_z = th.linalg.eigvals(A_z).abs().max().item()
        rho_euler_list.append(rho_e)
        rho_zoh_list.append(rho_z)
        stable = '✓' if rho_e < 1.0 else '✗ UNSTABLE'
        print(f"  {d:>4}  {rho_e:>10.6f}  {rho_z:>10.6f}  {stable:>14}")

    _plot_spectral(d_values, rho_euler_list, rho_zoh_list, "test1_spectral_radius.png")

    # For actual config (d=16): is Euler unstable?
    idx16 = d_values.index(16) if 16 in d_values else -1
    euler_unstable_at_16 = rho_euler_list[idx16] > 1.0 if idx16 >= 0 else None
    if euler_unstable_at_16 is not None:
        print(f"\n  At d=16 (actual config): Euler ρ={rho_euler_list[idx16]:.6f} "
              f"→ {'UNSTABLE ✗' if euler_unstable_at_16 else 'stable ✓'}")
    return all(r < 1.0 for r in rho_zoh_list)


# ─────────────────────────────────────────────────────────────────────────── #
#  Test 2 — Pure LMU: Euler vs ZOH × old vs new recon vs ground truth
# ─────────────────────────────────────────────────────────────────────────── #

def _run_pure_lmu(A_bar, B_bar, signal, T):
    """Roll out pure LMU recurrence for T steps. Returns final memory + norm history."""
    mem = th.zeros(1, A_bar.shape[0], 1, dtype=th.float32)
    A_bar_f = A_bar.float().unsqueeze(0)   # (1, d, d)
    B_bar_f = B_bar.float().view(-1, 1)    # (d, 1)
    norm_hist = []
    for t in range(T):
        u   = signal[t].view(1, 1, 1)
        mem = th.matmul(A_bar_f, mem) + B_bar_f * u
        norm_hist.append(mem.norm().item())
    return mem, norm_hist


def test_pure_lmu(memory_size=16, theta=50, T=300, dt=1.0):
    print(f"\n── Test 2: pure LMU — Euler vs ZOH, old vs new recon (d={memory_size}, θ={theta}) ──")
    A, B     = build_continuous_AB(memory_size)
    A_e, B_e = discretize_euler(A, B, dt, theta)
    A_z, B_z = discretize_zoh(A, B, dt, theta)
    P_buf    = build_P_buffer(memory_size)

    freq   = 2*math.pi / 30
    signal = th.sin(freq * th.arange(T).float())
    history= signal.tolist()

    # Spectral radii for this specific d
    rho_e = th.linalg.eigvals(A_e).abs().max().item()
    rho_z = th.linalg.eigvals(A_z).abs().max().item()
    print(f"  Euler spectral radius: {rho_e:.6f}  ({'✗ UNSTABLE' if rho_e>1 else '✓ stable'})")
    print(f"  ZOH   spectral radius: {rho_z:.6f}  ({'✓ stable' if rho_z<1 else '✗ UNSTABLE'})")

    mem_e, norm_e = _run_pure_lmu(A_e, B_e, signal, T)
    mem_z, norm_z = _run_pure_lmu(A_z, B_z, signal, T)

    print(f"\n  Memory norm after {T} steps:")
    print(f"    Euler : {norm_e[-1]:.4f}  (max: {max(norm_e):.4f})")
    print(f"    ZOH   : {norm_z[-1]:.4f}  (max: {max(norm_z):.4f})")

    # Evaluate recon at lags 0..theta-1 using BOTH memories and BOTH recon methods
    max_lag = theta - 1
    lags    = th.arange(max_lag + 1).float().unsqueeze(0)   # (1, L)
    truths  = th.tensor([history[T-1-k] for k in range(max_lag+1)])

    r_euler_new = recon_new(mem_e, lags, theta).squeeze()
    r_euler_old = recon_old(mem_e, lags, theta, P_buf).squeeze()
    r_zoh_new   = recon_new(mem_z, lags, theta).squeeze()
    r_zoh_old   = recon_old(mem_z, lags, theta, P_buf).squeeze()

    def stats(r):
        e = (r - truths).abs()
        return e.mean().item(), e.max().item()

    print(f"\n  Recon accuracy (mean/max |error| vs ground truth):")
    print(f"    Euler + old recon: {stats(r_euler_old)[0]:.4f} / {stats(r_euler_old)[1]:.4f}")
    print(f"    Euler + new recon: {stats(r_euler_new)[0]:.4f} / {stats(r_euler_new)[1]:.4f}")
    print(f"    ZOH   + old recon: {stats(r_zoh_old)[0]:.4f}  / {stats(r_zoh_old)[1]:.4f}")
    print(f"    ZOH   + new recon: {stats(r_zoh_new)[0]:.4f}  / {stats(r_zoh_new)[1]:.4f}  ← target")

    # Per-lag table for ZOH (where memory is valid)
    print(f"\n  ZOH per-lag  (new recon vs old recon vs truth) — lags 0..19:")
    print(f"  {'lag':>4}  {'truth':>8}  {'zoh+new':>10}  {'zoh+old':>12}  "
          f"{'err_new':>10}  {'err_old':>10}")
    for k in range(min(20, max_lag+1)):
        t  = truths[k].item()
        n  = r_zoh_new[k].item()
        o  = r_zoh_old[k].item()
        print(f"  {k:>4}  {t:>8.4f}  {n:>10.4f}  {o:>12.4f}  "
              f"{abs(n-t):>10.6f}  {abs(o-t):>10.6f}")

    _plot_test2(truths, norm_e, norm_z,
                r_euler_new, r_euler_old,
                r_zoh_new, r_zoh_old,
                max_lag, "test2_pure_lmu.png")

    # Pass: ZOH memory stable AND ZOH+new recon error < 0.2
    return norm_z[-1] < 10.0 and stats(r_zoh_new)[0] < 0.2


# ─────────────────────────────────────────────────────────────────────────── #
#  Test 3 — Full LMUCell (ZOH) stability + old vs new recon
# ─────────────────────────────────────────────────────────────────────────── #

def test_full_cell(input_size=8, hidden_size=4, memory_size=16,
                   num_units=8, theta=50, T=200, batch=4):
    print(f"\n── Test 3: full LMUCell (ZOH) — u_t stability + recon old vs new ──")
    cell = LMUCell(input_size, hidden_size, memory_size, num_units, theta, use_zoh=True)
    cell.eval()

    freq    = 2*math.pi / 30
    phases  = th.linspace(0, math.pi, input_size)
    signals = th.stack([th.sin(freq*th.arange(T).float() + p) for p in phases], dim=1)

    h   = th.zeros(batch, hidden_size, num_units)
    mem = th.zeros(batch, memory_size, num_units)

    u_norms, m_norms           = [], []
    err_new_hist, err_old_hist = [], []
    lag0 = th.zeros(batch, 1)

    with th.no_grad():
        for t in range(T):
            x_t         = signals[t].unsqueeze(0).expand(batch, -1)
            h, mem, u_t = cell(x_t, h, mem)
            u_norms.append(u_t.norm(dim=-1).mean().item())
            m_norms.append(mem.norm().item())
            r0_new = cell.recon(mem, lag0).squeeze(1)
            r0_old = cell.recon_old(mem, lag0).squeeze(1)
            err_new_hist.append((r0_new - u_t).norm(dim=-1).mean().item())
            err_old_hist.append((r0_old - u_t).norm(dim=-1).mean().item())

    print(f"  u_t norm  — mean:{_m(u_norms):.4f}  max:{max(u_norms):.4f}  min:{min(u_norms):.4f}")
    print(f"  mem norm  — mean:{_m(m_norms):.4f}  max:{max(m_norms):.4f}  final:{m_norms[-1]:.4f}")
    print(f"  recon@lag=0 err NEW — mean:{_m(err_new_hist):.5f}  max:{max(err_new_hist):.5f}")
    print(f"  recon@lag=0 err OLD — mean:{_m(err_old_hist):.5f}  max:{max(err_old_hist):.5f}")

    # Lag sweep
    lags     = th.linspace(0, theta-1, 20).unsqueeze(0).expand(batch, -1)
    rn_sweep = cell.recon(mem, lags).norm(dim=-1).mean(0)
    ro_sweep = cell.recon_old(mem, lags).norm(dim=-1).mean(0)
    print(f"\n  Lag-sweep recon norm (ZOH memory, should stay O(1)):")
    print(f"  {'lag':>6}  {'new':>10}  {'old':>12}")
    for i, lag_val in enumerate(th.linspace(0, theta-1, 20).tolist()):
        print(f"  {lag_val:>6.1f}  {rn_sweep[i].item():>10.4f}  {ro_sweep[i].item():>12.4f}")

    _plot_test3(u_norms, m_norms, err_new_hist, err_old_hist,
                rn_sweep.tolist(), ro_sweep.tolist(), theta, "test3_full_cell.png")

    stable = max(m_norms) < 1e3 and max(u_norms) < 1e3
    print(f"\n  Stability: {'✓ PASS' if stable else '✗ FAIL'}")
    return stable


def _m(lst): return sum(lst) / len(lst)


# ─────────────────────────────────────────────────────────────────────────── #
#  Plots
# ─────────────────────────────────────────────────────────────────────────── #

def _plot_spectral(d_vals, rho_e, rho_z, fname):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(d_vals, rho_e, 'o-', label='Euler ρ', color='tab:red')
    ax.plot(d_vals, rho_z, 's-', label='ZOH ρ',   color='tab:blue')
    ax.axhline(1.0, color='black', ls='--', lw=1.2, label='stability boundary')
    ax.fill_between(d_vals, 1.0, [max(rho_e+[1.1])]*len(d_vals),
                    alpha=0.08, color='red', label='unstable region')
    ax.set_xlabel('memory_size d'); ax.set_ylabel('spectral radius ρ')
    ax.set_title('Euler vs ZOH spectral radius (θ=50, dt=1)\n'
                 'Euler crosses ρ=1 at moderate d → memory explodes')
    ax.legend(); fig.tight_layout(); fig.savefig(fname, dpi=120)
    print(f"  Plot saved → {fname}")


def _plot_test2(truths, norm_e, norm_z,
                r_en, r_eo, r_zn, r_zo, max_lag, fname):
    lags = list(range(max_lag + 1))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Memory norm over time
    axes[0,0].semilogy(norm_e, label='Euler', color='tab:red')
    axes[0,0].semilogy(norm_z, label='ZOH',   color='tab:blue')
    axes[0,0].set_title('Memory norm over time (log)'); axes[0,0].legend()

    # Recon signal — ZOH (valid memory)
    axes[0,1].plot(lags, truths.tolist(), label='truth', lw=2)
    axes[0,1].plot(lags, r_zn.tolist(), label='ZOH+new', ls='--', lw=1.5)
    axes[0,1].plot(lags, [v if abs(v)<5 else float('nan') for v in r_zo.tolist()],
                   label='ZOH+old (clip|·|>5)', ls=':', lw=1.5)
    axes[0,1].set_title('ZOH memory recon vs truth'); axes[0,1].legend()

    # Recon signal — Euler (corrupted memory)
    axes[0,2].plot(lags, truths.tolist(), label='truth', lw=2)
    axes[0,2].plot(lags, [v if abs(v)<50 else float('nan') for v in r_en.tolist()],
                   label='Euler+new (clip|·|>50)', ls='--', lw=1.5, color='tab:red')
    axes[0,2].set_title('Euler memory recon (corrupted)'); axes[0,2].legend()

    # Absolute error — ZOH only
    err_zn = (r_zn - truths).abs().tolist()
    err_zo = (r_zo - truths).abs().tolist()
    axes[1,0].semilogy(lags, [max(v,1e-9) for v in err_zn], label='ZOH+new')
    axes[1,0].semilogy(lags, [max(v,1e-9) for v in err_zo], label='ZOH+old', ls='--')
    axes[1,0].set_title('ZOH recon |error| (log)'); axes[1,0].legend()

    # Absolute error — Euler vs ZOH (new recon only, to isolate discretisation)
    err_en = (r_en - truths).abs().tolist()
    axes[1,1].semilogy(lags, [max(v,1e-9) for v in err_en], label='Euler+new', color='tab:red')
    axes[1,1].semilogy(lags, [max(v,1e-9) for v in err_zn], label='ZOH+new',   color='tab:blue')
    axes[1,1].set_title('Euler vs ZOH |error| — new recon (log)'); axes[1,1].legend()

    # Old vs new recon error — ZOH memory (to isolate recon)
    axes[1,2].semilogy(lags, [max(v,1e-9) for v in err_zn], label='ZOH+new (recurrence)')
    axes[1,2].semilogy(lags, [max(v,1e-9) for v in err_zo], label='ZOH+old (monomial)',   ls='--')
    axes[1,2].set_title('New vs old recon — same (ZOH) memory'); axes[1,2].legend()

    for ax in axes.flat: ax.set_xlabel('lag / step')
    fig.tight_layout(); fig.savefig(fname, dpi=120)
    print(f"  Plot saved → {fname}")


def _plot_test3(u_norms, m_norms, err_new, err_old, rn_sw, ro_sw, theta, fname):
    steps = list(range(len(u_norms)))
    lags  = list(th.linspace(0, theta-1, 20).numpy())
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes[0,0].plot(steps, u_norms);  axes[0,0].set_title('u_t norm (ZOH cell)')
    axes[0,1].plot(steps, m_norms);  axes[0,1].set_title('memory norm (ZOH, should plateau)')
    axes[0,2].semilogy(steps, [max(v,1e-9) for v in err_new], label='new')
    axes[0,2].semilogy(steps, [max(v,1e-9) for v in err_old], label='old', ls='--')
    axes[0,2].set_title('recon@lag=0 error (log)'); axes[0,2].legend()
    axes[1,0].plot(steps, err_new, label='new')
    axes[1,0].plot(steps, err_old, label='old', ls='--')
    axes[1,0].set_title('recon@lag=0 error (linear)'); axes[1,0].legend()
    axes[1,1].plot(lags, rn_sw, label='new')
    axes[1,1].plot(lags, ro_sw, label='old', ls='--')
    axes[1,1].set_title('lag-sweep recon norm'); axes[1,1].legend()
    axes[1,2].semilogy(lags, [max(v,1e-9) for v in rn_sw], label='new')
    axes[1,2].semilogy(lags, [max(v,1e-9) for v in ro_sw], label='old', ls='--')
    axes[1,2].set_title('lag-sweep recon norm (log)'); axes[1,2].legend()
    for ax in axes.flat: ax.set_xlabel('step / lag')
    fig.tight_layout(); fig.savefig(fname, dpi=120)
    print(f"  Plot saved → {fname}")


# ─────────────────────────────────────────────────────────────────────────── #
if __name__ == "__main__":
    th.manual_seed(0)
    p1 = test_spectral_radius(theta=100, dt=1.0)
    p2 = test_pure_lmu(memory_size=32, theta=100, T=300)
    p3 = test_full_cell(memory_size=32, theta=100)

    print("\n══ Results ══")
    print(f"  Spectral radius (ZOH all-stable) : {'PASS' if p1 else 'FAIL'}")
    print(f"  Pure LMU recon (ZOH+new < 0.2)  : {'PASS' if p2 else 'FAIL'}")
    print(f"  Full cell stability              : {'PASS' if p3 else 'FAIL'}")
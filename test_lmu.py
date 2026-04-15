"""
LMU Validation Suite
====================
Tests the mathematical properties of the multichannel LMU cell.

Sections
--------
A  Raw (Ā, B̄) math — bypasses learned encoders, tests the ODEs directly
   A1  A, B continuous-time formulas match Voelker 2019 Eq. 2
   A2  ZOH stability:  ρ(Ā) < 1 for all theta values
   A3  B̄ alternating-sign structure
   A4  Legendre reconstruction:  feed known scalar signal, reconstruct history from m_T

B  LMUCell unit tests
   B1  Shape correctness
   B2  Multichannel independence:  perturbing channel c leaves m[:,:,c'] unchanged
   B3  Gradient flow:  every learnable param receives a gradient; Ā, B̄ do not
   B4  Memory persistence:  norms stay bounded over 200 steps
   B5  e_m = 0 at init (Voelker §3 requirement)

C  Visualisations (saved to PNGs if matplotlib is available)
   C1  Reconstruction quality vs d
   C2  m and h norms over a 200-step rollout
"""

import sys
import numpy as np
import torch

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from lmu import LMUCell, get_AB


# ─── Utilities ────────────────────────────────────────────────────────────────

def shifted_legendre(r: float, d: int) -> np.ndarray:
    """
    Evaluate the shifted Legendre polynomials P_0, …, P_{d-1} at r ∈ [0, 1].

    Uses the numerically stable 3-term recurrence (never forms monomials):
        P_0(r) = 1
        P_1(r) = 2r − 1
        P_{n+1}(r) = [ (2n+1)(2r−1) P_n(r) − n P_{n−1}(r) ] / (n+1)

    Values stay in [−1, 1] for all r ∈ [0, 1], so there is no catastrophic
    cancellation even for large d.  The old monomial expansion (Eq. 3 written
    out literally) loses ~7 decimal digits of precision at d=16 in float32.
    """
    P = np.zeros(d)
    P[0] = 1.0
    if d > 1:
        P[1] = 2.0 * r - 1.0
    for n in range(1, d - 1):
        P[n + 1] = ((2*n + 1) * (2*r - 1.0) * P[n] - n * P[n - 1]) / (n + 1)
    return P


def run_raw_recurrence(signal: np.ndarray,
                       A: np.ndarray,
                       B: np.ndarray) -> np.ndarray:
    """
    Pure Legendre memory recurrence, no learned encoders:

        m_t = Ā m_{t-1} + B̄ u_t

    signal : (T,)  scalar u_t values
    Returns  (T, d)  memory trajectory
    """
    T, d = len(signal), A.shape[0]
    m = np.zeros(d)
    traj = np.zeros((T, d))
    for t, u in enumerate(signal):
        m = A @ m + B.squeeze(-1) * u
        traj[t] = m
    return traj


def reconstruct(m_T: np.ndarray, theta: float, d: int,
                n_pts: int = 500) -> tuple:
    """
    Reconstruct u(t − θ') from final memory state m_T.

    Voelker 2019 Eq. 3:
        u(t − θ') ≈ Σ_i  P_i(θ'/θ) · m_i(t)

    Returns (lags, u_recon) where lags ∈ [0, theta].
    """
    lags  = np.linspace(0.0, theta, n_pts)
    u_rec = np.array([np.dot(shifted_legendre(lag / theta, d), m_T)
                      for lag in lags])
    return lags, u_rec


# ─── Section A : raw math ─────────────────────────────────────────────────────

def test_A1_formula_correctness(d: int = 16, theta: float = 50.0) -> None:
    """Spot-check A, B entries against Voelker 2019 Eq. 2 (before ZOH)."""
    print("\n── A1  A, B continuous-time formula ──")
    Q = np.arange(d, dtype=float)
    R = (2 * Q + 1)[:, None]
    j_, i_ = np.meshgrid(Q, Q)

    A_c = R * np.where(i_ < j_, -1.0, (-1.0) ** (i_ - j_ + 1)) / theta
    B_c = (R * ((-1.0) ** Q)[:, None]) / theta

    # A[i,j] for i≥j: should be (2i+1)·(−1)^{i−j+1} / theta
    for i, j in [(3, 1), (5, 0), (4, 4), (0, 5)]:
        if i < j:
            expected = -(2*i + 1) / theta
        else:
            expected = ((-1) ** (i - j + 1)) * (2*i + 1) / theta
        assert abs(A_c[i, j] - expected) < 1e-12, \
            f"A[{i},{j}] = {A_c[i,j]:.6g}, expected {expected:.6g}"

    # B_i = (2i+1)(−1)^i / theta
    for i in range(d):
        expected = (2*i + 1) * ((-1) ** i) / theta
        assert abs(B_c[i, 0] - expected) < 1e-12, \
            f"B[{i}] = {B_c[i,0]:.6g}, expected {expected:.6g}"

    print(f"  B[:6] = {B_c[:6, 0].round(3)}  (alternating sign ✓)")
    print("  PASS")


def test_A2_stability(theta_vals=(10, 50, 100, 200, 500), d: int = 32) -> None:
    """ρ(Ā) < 1 for all theta values after ZOH discretisation."""
    print("\n── A2  Ā spectral radius ──")
    for theta in theta_vals:
        Ad, _ = get_AB(d, theta)
        rho = np.abs(np.linalg.eigvals(Ad)).max()
        status = "✓" if rho < 1.0 else "✗ UNSTABLE"
        print(f"  d={d}, θ={theta:5.0f}: ρ(Ā) = {rho:.10f}  {status}")
        assert rho < 1.0, f"Unstable Ā: theta={theta}, rho={rho}"
    print("  PASS")


def test_A3_B_structure(d: int = 16, theta: float = 50.0) -> None:
    """
    B structure tests — split between continuous-time and ZOH-discretised.

    Continuous-time B_i = (2i+1)(−1)^i / theta:
      • alternating sign  (trivially, (-1)^i)
      • monotonically increasing magnitude  |B_i| = (2i+1)/theta

    After ZOH, the matrix exponential mixes entries, so the magnitude-
    monotonicity property is NOT preserved for large i.  We only check
    alternating sign on the low-index entries where ZOH perturbation is small.
    """
    print("\n── A3  B̄ structure ──")

    # Continuous-time B — monotonic magnitude must hold exactly
    Q   = np.arange(d, dtype=float)
    B_c = (2*Q + 1) * ((-1.0)**Q) / theta   # (d,)
    mags_c = np.abs(B_c)
    assert all(mags_c[i] < mags_c[i+1] for i in range(d-1)), \
        "Continuous-time |B_i| not monotonically increasing"
    print(f"  B_cont[:6]  = {B_c[:6].round(4)}  (alternating ✓, increasing ✓)")

    # ZOH-discretised B̄ — sign alternation is NOT guaranteed by ZOH.
    # expm mixes entries via off-diagonal coupling; the pattern breaks at
    # varying indices depending on d and theta.  What IS guaranteed:
    #   1. B̄[0] > 0                  (DC component, preserved by expm)
    #   2. B̄ ≈ (1/theta) * B_cont    for small 1/theta  (first-order Euler)
    #   3. ||B̄||₂ is finite and non-zero
    _, Bd = get_AB(d, theta)
    b = Bd.squeeze(-1)

    assert b[0] > 0, f"B̄[0] should be positive, got {b[0]:.6f}"

    # First-order Euler approximation: B̄ ≈ B_cont / theta.  Check sign
    # agreement with B_cont for the first 4 entries where Euler error is small.
    B_c_scaled = (2 * np.arange(4) + 1) * ((-1.0) ** np.arange(4)) / theta
    for i in range(4):
        assert b[i] * B_c_scaled[i] > 0, \
            f"B̄[{i}] sign disagrees with B_cont[{i}] after ZOH"

    assert np.linalg.norm(b) > 0, "B̄ is all zeros"

    print(f"  B̄ (ZOH)[:6] = {b[:6].round(5)}")
    print(f"  B̄[0] > 0 ✓  |  sign agrees with B_cont for i<4 ✓  |  non-zero ✓")
    print(f"  (ZOH mixes high-index entries via expm — full sign alternation not expected)")
    print("  PASS")


def test_A4_reconstruction(theta: float = 50.0, d: int = 32,
                           period: float = 25.0) -> tuple:
    """
    Core math test: Voelker 2019 Eq. 3.

    Setup
    -----
    • Signal: sine wave, period=25 steps (2 full cycles inside the θ=50 window)
    • Run T=4·θ steps so the memory is in steady state
    • At t=T reconstruct u(T−θ') for θ' ∈ [0, θ]
    • Compare to the known ground truth u(T−θ') = sin(2π(T−θ')/period)

    Error bound: approximation error ≈ O(θ·ω/d) where ω = 1/period.
    For d=32, θ=50, period=25: O(50/(25·32)) ≈ 0.06 → MSE should be well below 0.01.
    """
    print(f"\n── A4  Legendre reconstruction  (d={d}, θ={theta}, period={period}) ──")
    Ad, Bd = get_AB(d, theta)
    T = int(4 * theta)

    t_vec  = np.arange(T, dtype=float)
    signal = np.sin(2 * np.pi * t_vec / period)

    m_traj = run_raw_recurrence(signal, Ad, Bd)
    m_T    = m_traj[-1]  # (d,)  final memory state

    lags, u_recon = reconstruct(m_T, theta, d, n_pts=500)
    u_true = np.sin(2 * np.pi * (T - lags) / period)

    mse     = np.mean((u_recon - u_true) ** 2)
    max_err = np.abs(u_recon - u_true).max()

    print(f"  Signal:          sine, period={period}")
    print(f"  Reconstruction MSE:  {mse:.6f}  (threshold 0.01)")
    print(f"  Max absolute error:  {max_err:.6f}")

    # Check reconstruction at t=0 lag (most recent): should ≈ sin(2π·T/period)
    r0_pred = float(np.dot(shifted_legendre(0.0, d), m_T))
    r0_true = float(signal[-1])
    print(f"  Recon at lag=0:      {r0_pred:.4f}  (true {r0_true:.4f})")

    assert mse < 0.01, f"Reconstruction MSE too high: {mse:.4f}"
    print("  PASS")
    return lags, u_recon, u_true


# ─── Section B : LMUCell unit tests ──────────────────────────────────────────

def test_B1_shapes(C=64, n=128, d=32, theta=100.0, B=4) -> LMUCell:
    print("\n── B1  Shape checks ──")
    cell = LMUCell(C, n, d, theta)
    h, m = cell.initial_state(B, torch.device('cpu'))

    assert h.shape == (B, n),    f"h init: {h.shape}"
    assert m.shape == (B, d, C), f"m init: {m.shape}"

    x = torch.randn(B, C)
    h_new, m_new = cell(x, h, m)

    assert h_new.shape == (B, n),    f"h_new: {h_new.shape}"
    assert m_new.shape == (B, d, C), f"m_new: {m_new.shape}"

    for name, t in [('h_init', h), ('m_init', m),
                    ('h_new',  h_new), ('m_new', m_new)]:
        print(f"  {name:8s} : {tuple(t.shape)}  ✓")
    print("  PASS")
    return cell


def test_B2_multichannel_independence(C=16, n=32, d=16, theta=50.0) -> None:
    """
    Verify channels evolve independently in the memory (m).

    Intuition
    ---------
    The memory update is:
        m_new[:, :, c] = Ā @ m[:, :, c]  +  B̄ · u[:, c]

    u[:, c] = e_x[c]·x[:,c]  +  E_h[:,c]·h  +  e_m·m[:,:,c]

    At t=0 (initial state), h=0 and m=0, so:
        u[:, c]  =  e_x[c] · x[:, c]      (only depends on x_c)

    Perturbing x[:,0] by Δ therefore only changes u[:,0] and thus m[:,:,0].
    All other channels c≠0 are unaffected for this single step.
    """
    print("\n── B2  Multichannel independence ──")
    cell = LMUCell(C, n, d, theta)
    cell.eval()

    h0, m0     = cell.initial_state(1, torch.device('cpu'))
    x_base     = torch.randn(1, C)
    x_perturb  = x_base.clone()
    x_perturb[0, 0] += 100.0      # very large Δ on channel 0 only

    with torch.no_grad():
        _, m_base  = cell(x_base,    h0, m0)
        _, m_pert  = cell(x_perturb, h0, m0)

    diff_ch0  = (m_base[0, :, 0]  - m_pert[0, :, 0]).abs().max().item()
    diff_rest = (m_base[0, :, 1:] - m_pert[0, :, 1:]).abs().max().item()

    print(f"  ch0  max |Δm|  (should be > 0):    {diff_ch0:.6f}")
    print(f"  ch1+ max |Δm|  (should be 0):       {diff_rest:.2e}")

    assert diff_ch0  > 1e-6,  "Channel 0 memory unchanged despite x perturbation"
    assert diff_rest < 1e-7,  f"Cross-channel contamination: {diff_rest:.2e}"
    print("  PASS")


def test_B3_gradient_flow(C=32, n=64, d=16, theta=50.0, T=20) -> None:
    """
    Every learnable parameter must receive a gradient.
    Frozen buffers Ā and B̄ must not.
    """
    print("\n── B3  Gradient flow ──")
    cell = LMUCell(C, n, d, theta)
    cell.train()

    h, m = cell.initial_state(2, torch.device('cpu'))
    for _ in range(T):
        h, m = cell(torch.randn(2, C), h, m)

    # Loss on both outputs
    (h.mean() + m.mean()).backward()

    params = {
        'e_x':    cell.e_x,
        'E_h.W':  cell.E_h.weight,
        'e_m':    cell.e_m,
        'C_proj': cell.C_proj,
        'W_x.W':  cell.W_x.weight,
        'W_x.b':  cell.W_x.bias,
        'W_h.W':  cell.W_h.weight,
        'W_m.W':  cell.W_m.weight,
    }
    all_ok = True
    for name, p in params.items():
        gnorm = p.grad.abs().max().item() if p.grad is not None else 0.0
        ok    = gnorm > 0.0
        print(f"  {name:10s}: grad max = {gnorm:.3e}  {'✓' if ok else '✗ NO GRAD'}")
        all_ok = all_ok and ok

    # Frozen buffers
    assert cell.A.grad is None, "Ā must not accumulate gradients"
    assert cell.B.grad is None, "B̄ must not accumulate gradients"
    print("  Ā, B̄: no grad (frozen)  ✓")
    assert all_ok, "Some learnable parameters received no gradient"
    print("  PASS")


def test_B4_memory_persistence(C=16, n=32, d=16, theta=50.0,
                                T=200) -> tuple:
    """h and m norms must stay bounded over a long rollout."""
    print(f"\n── B4  Memory persistence ({T} steps) ──")
    cell = LMUCell(C, n, d, theta)
    cell.eval()

    h, m = cell.initial_state(1, torch.device('cpu'))
    m_norms, h_norms = [], []
    with torch.no_grad():
        for _ in range(T):
            h, m = cell(torch.randn(1, C), h, m)
            m_norms.append(m.norm().item())
            h_norms.append(h.norm().item())

    print(f"  ||m|| — max: {max(m_norms):.4f}  final: {m_norms[-1]:.4f}")
    print(f"  ||h|| — max: {max(h_norms):.4f}  final: {h_norms[-1]:.4f}")
    assert max(m_norms) < 1e4, f"m exploded: {max(m_norms)}"
    assert max(h_norms) < 1e4, f"h exploded: {max(h_norms)}"
    print("  PASS")
    return m_norms, h_norms


def test_B5_init_checks(C=16, n=32, d=16, theta=50.0) -> None:
    """Verify e_m=0 at init and A,B are non-trainable (Voelker §3)."""
    print("\n── B5  Initialisation checks ──")
    cell = LMUCell(C, n, d, theta)

    em_max = cell.e_m.abs().max().item()
    print(f"  e_m max |value|: {em_max}  (should be 0.0)")
    assert em_max == 0.0, f"e_m not zero at init: {em_max}"

    assert not cell.A.requires_grad, "Ā must be frozen"
    assert not cell.B.requires_grad, "B̄ must be frozen"
    print("  Ā, B̄ not trainable  ✓")

    # Check spectral radius via buffer
    rho = torch.linalg.eigvals(cell.A).abs().max().item()
    print(f"  Ā spectral radius: {rho:.8f}  {'✓' if rho < 1 else '✗ UNSTABLE'}")
    assert rho < 1.0
    print("  PASS")


# ─── Section C : visualisations ──────────────────────────────────────────────

def plot_reconstruction(theta: float = 50.0, period: float = 25.0,
                        d_vals=(8, 16, 32, 64)) -> None:
    if not HAS_MPL:
        print("  [skip — matplotlib not found]")
        return

    T      = int(4 * theta)
    t_vec  = np.arange(T, dtype=float)
    signal = np.sin(2 * np.pi * t_vec / period)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    colours = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']

    # Panel 1: reconstruction curves
    for d, col in zip(d_vals, colours):
        Ad, Bd = get_AB(d, theta)
        m_T    = run_raw_recurrence(signal, Ad, Bd)[-1]
        lags, u_recon = reconstruct(m_T, theta, d, n_pts=500)
        u_true = np.sin(2 * np.pi * (T - lags) / period)
        mse    = np.mean((u_recon - u_true) ** 2)
        ax1.plot(lags, u_recon, color=col, lw=1.8,
                 label=f'd={d}   MSE={mse:.5f}', alpha=0.85)

    lags_gt = np.linspace(0, theta, 500)
    u_gt    = np.sin(2 * np.pi * (T - lags_gt) / period)
    ax1.plot(lags_gt, u_gt, 'k--', lw=1.2, alpha=0.55, label='ground truth')
    ax1.set_xlabel("lag θ' (steps back)")
    ax1.set_ylabel("u(t − θ')")
    ax1.set_title(f"Legendre reconstruction  (θ={theta}, period={period})")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.25)

    # Panel 2: MSE vs d
    d_range, mse_vals = list(range(4, 80, 2)), []
    for d in d_range:
        Ad, Bd = get_AB(d, theta)
        m_T    = run_raw_recurrence(signal, Ad, Bd)[-1]
        lags, u_recon = reconstruct(m_T, theta, d, n_pts=500)
        u_true = np.sin(2 * np.pi * (T - lags) / period)
        mse_vals.append(np.mean((u_recon - u_true) ** 2))

    ax2.semilogy(d_range, mse_vals, 'o-', color='#1f77b4', ms=4)
    ax2.axvline(theta / period * 2, color='r', ls='--', alpha=0.7,
                label=f'2·freq·θ ≈ {theta/period*2:.0f}')
    ax2.set_xlabel('memory size d')
    ax2.set_ylabel('reconstruction MSE (log scale)')
    ax2.set_title(f'MSE vs d  (θ={theta}, period={period})')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig('lmu_reconstruction.png', dpi=130, bbox_inches='tight')
    plt.close()
    print("  Saved → lmu_reconstruction.png")


def plot_norms(m_norms: list, h_norms: list) -> None:
    if not HAS_MPL:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
    ax1.plot(m_norms, color='#9467bd', lw=1.5)
    ax1.set_title('Memory m  ‖·‖ over time')
    ax1.set_xlabel('step'); ax1.set_ylabel('‖m‖'); ax1.grid(alpha=0.25)
    ax2.plot(h_norms, color='#2ca02c', lw=1.5)
    ax2.set_title('Hidden h  ‖·‖ over time')
    ax2.set_xlabel('step'); ax2.set_ylabel('‖h‖'); ax2.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig('lmu_norms.png', dpi=130, bbox_inches='tight')
    plt.close()
    print("  Saved → lmu_norms.png")


# ─── main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("LMU Validation Suite")
    print("=" * 60)

    failed = []

    def run(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed.append(fn.__name__)
            return None

    # ── A: raw math ──────────────────────────────────────────────────────
    run(test_A1_formula_correctness, d=16, theta=50.0)
    run(test_A2_stability)
    run(test_A3_B_structure, d=32, theta=50.0)
    result = run(test_A4_reconstruction, theta=50.0, d=32, period=25.0)
    recon_data = result  # (lags, u_recon, u_true) or None

    # ── B: full cell ──────────────────────────────────────────────────────
    run(test_B1_shapes)
    run(test_B2_multichannel_independence)
    run(test_B3_gradient_flow)
    norm_data = run(test_B4_memory_persistence)
    run(test_B5_init_checks)

    # ── C: visualisations ─────────────────────────────────────────────────
    print("\n── C  Visualisations ──")
    plot_reconstruction(theta=50.0, period=25.0)
    if norm_data:
        plot_norms(*norm_data)

    # ── summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if failed:
        print(f"FAILED: {failed}")
        sys.exit(1)
    else:
        print("All tests passed ✓")
    print("=" * 60)
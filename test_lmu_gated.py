"""
Tests for lmu.py — Gated Write variant.

Run:
    pytest test_lmu.py -v
    pytest test_lmu.py -v -k "ortho"        # OrthoLayer only
    pytest test_lmu.py -v -k "condition"    # _compute_u boundary conditions only

Coverage:
    TestOrthoLayer          — orthogonality, isometry, Cayley step, SVD fallback,
                              grad-zeroing, no-grad early exit
    TestComputeU            — Condition 1 (u_x=0), Condition 2 (u_x=pred),
                              novel state, no-bias requirement
    TestRIntr               — bounds [0, √C], zero at conditions, shape, dtype
    TestExNorm              — e_x always unit-norm after F.normalize
    TestEpisodeBoundary     — pred=0 at reset guarantees full gate strength
    TestLMUCellForward      — output shapes, memory update, stability over many steps
    TestGradientFlow        — u_null in live graph, W_pre.grad populated and zeroed,
                              gradient does NOT flow through e_x via u_null
    TestOptimizerExclusion  — W_pre excluded from Adam, ortho_update changes only W_pre
    TestLMUSequenceWrapper  — shapes, r_intrs accumulation
"""

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from lmu_t import LMUCell, LMU, OrthoLayer


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

B = 4    # batch size
C = 16   # input_size / encoder_dim  (small for fast tests)
N = 32   # hidden_size
D = 8    # memory_size
T = 20   # sequence length

THETA = 50.0
ATOL  = 1e-5   # tolerance for exact-math assertions


@pytest.fixture
def cell():
    torch.manual_seed(0)
    return LMUCell(input_size=C, hidden_size=N, memory_size=D, theta=THETA)


@pytest.fixture
def ortho():
    return OrthoLayer(size=C)


@pytest.fixture
def zero_state(cell):
    return cell.initial_state(B, torch.device('cpu'))


# ─────────────────────────────────────────────────────────────────────────────
# TestOrthoLayer
# ─────────────────────────────────────────────────────────────────────────────

class TestOrthoLayer:

    def test_init_is_identity(self, ortho):
        """Initialised to I — valid starting point on O(C)."""
        assert torch.allclose(ortho.weights, torch.eye(C))

    def test_forward_shape(self, ortho):
        x = torch.randn(B, C)
        out = ortho(x)
        assert out.shape == (B, C)

    def test_orthogonality_error_at_init(self, ortho):
        """WᵀW − I = 0 at identity init."""
        assert ortho.orthogonality_error() < 1e-6

    def test_isometry_at_init(self, ortho):
        """‖W_pre(v)‖₂ = ‖v‖₂ — holds at identity, should hold always."""
        v = torch.randn(B, C)
        out = ortho(v)
        torch.testing.assert_close(
            out.norm(dim=-1), v.norm(dim=-1), atol=ATOL, rtol=0
        )

    def test_isometry_after_ortho_update(self, ortho):
        """Isometry must hold after a Riemannian gradient step."""
        v   = torch.randn(B, C)
        out = ortho(v)
        loss = out.sum()
        loss.backward()
        ortho.ortho_update(lr=1e-2)

        # Fresh v to avoid any stale computation graph
        v2   = torch.randn(B, C)
        out2 = ortho(v2)
        torch.testing.assert_close(
            out2.norm(dim=-1), v2.norm(dim=-1), atol=1e-4, rtol=0
        )

    def test_ortho_update_maintains_orthogonality(self, ortho):
        """‖WᵀW − I‖_F < 1e-5 after one Riemannian step."""
        v = torch.randn(B, C)
        ortho(v).sum().backward()
        ortho.ortho_update(lr=1e-2)
        assert ortho.orthogonality_error() < 1e-5

    def test_ortho_update_zeros_grad(self, ortho):
        """ortho_update must zero the gradient so Adam never sees it."""
        v = torch.randn(B, C)
        ortho(v).sum().backward()
        assert ortho.weights.grad is not None
        ortho.ortho_update(lr=1e-2)
        assert ortho.weights.grad is not None           # object exists
        assert ortho.weights.grad.abs().max() == 0.0   # but is zeroed

    def test_ortho_update_no_grad_is_noop(self, ortho):
        """ortho_update with no gradient should return without error."""
        assert ortho.weights.grad is None
        ortho.ortho_update(lr=1e-2)   # should not raise
        # weights unchanged (identity)
        assert torch.allclose(ortho.weights, torch.eye(C))

    def test_reorthogonalize_restores_from_corruption(self, ortho):
        """SVD fallback: hard-corrupt W then reorthogonalize → error < 1e-5.
        FIXED: 1e-6 → 1e-5.  SVD on a heavily corrupted float32 matrix
        (0.5*randn corruption, C=16) accumulates rounding across 16×16 ops.
        Observed residual ~3e-6 is within float32 expectations; 1e-6 was
        too tight.  1e-5 is the correct float32 bound here.
        """
        with torch.no_grad():
            ortho.weights += 0.5 * torch.randn(C, C)   # deliberate corruption
        assert ortho.orthogonality_error() > 0.1        # corruption confirmed
        ortho.reorthogonalize()
        assert ortho.orthogonality_error() < 1e-5

    def test_ortho_update_does_not_change_other_params(self):
        """Riemannian step on OrthoLayer only moves W_pre, not any other tensor."""
        ortho_a = OrthoLayer(C)
        ortho_b = OrthoLayer(C)   # untouched reference

        v = torch.randn(B, C)
        ortho_a(v).sum().backward()
        w_before = ortho_b.weights.clone()
        ortho_a.ortho_update(lr=1e-2)
        assert torch.allclose(ortho_b.weights, w_before)

    def test_cumulative_orthogonality_over_many_steps(self, ortho):
        """After 200 Cayley steps, error should stay < 1e-4."""
        for _ in range(200):
            v = torch.randn(B, C)
            ortho(v).sum().backward()
            ortho.ortho_update(lr=1e-3)
        assert ortho.orthogonality_error() < 1e-4

    def test_zero_input_produces_zero_output(self, ortho):
        """W_pre has no bias: W_pre(0) = 0 exactly. Critical for u_null proof."""
        z   = torch.zeros(B, C)
        out = ortho(z)
        assert out.abs().max().item() == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# TestComputeU  — boundary conditions
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeU:
    """
    Tests for _compute_u in isolation.  We call it directly with fabricated
    (u_x, u_h, u_m) tensors to verify the mathematical conditions independently
    of the rest of the forward pass.
    """

    def test_condition1_zero_ux_returns_pred(self, cell):
        """
        Condition 1: u_x = 0 → u = pred = u_h + u_m.
        Zero observation should write only the memory prediction.
        """
        u_h = torch.randn(B, C)
        u_m = torch.randn(B, C)
        u_x = torch.zeros(B, C)
        pred = u_h + u_m

        u = cell._compute_u(u_x, u_h, u_m)
        torch.testing.assert_close(u, pred, atol=ATOL, rtol=0,
            msg="Condition 1 failed: u_x=0 should give u=pred")

    def test_condition2_ux_equals_pred_returns_pred(self, cell):
        """
        Condition 2: u_x = pred → u = pred.
        Correctly anticipated observation should not update memory beyond prediction.
        """
        u_h = torch.randn(B, C)
        u_m = torch.randn(B, C)
        u_x = u_h + u_m   # exactly pred

        u = cell._compute_u(u_x, u_h, u_m)
        pred = u_h + u_m
        torch.testing.assert_close(u, pred, atol=ATOL, rtol=0,
            msg="Condition 2 failed: u_x=pred should give u=pred")

    def test_novel_state_departs_from_pred(self, cell):
        """Novel observation (u_x ≠ 0, u_x ≠ pred) should write u ≠ pred."""
        u_h = torch.randn(B, C)
        u_m = torch.randn(B, C)
        # u_x is random — almost certainly ≠ 0 and ≠ pred
        u_x = torch.randn(B, C)
        pred = u_h + u_m

        u = cell._compute_u(u_x, u_h, u_m)
        # They should differ for at least some elements
        assert not torch.allclose(u, pred, atol=1e-3), \
            "Novel u_x should produce u ≠ pred"

    def test_conditions_hold_with_zero_pred(self, cell):
        """
        Special case: u_h = u_m = 0 (episode start after reset).
        Condition 1: u_x=0 → u=0.
        Condition 2: u_x=0=pred → u=0.  (Both conditions coincide here.)
        Novel:       u_x≠0 → u ≠ 0 (memory writes the full observation).
        This is the critical episode-start regime for the Memory task.
        """
        u_h = torch.zeros(B, C)
        u_m = torch.zeros(B, C)

        # Condition 1 / 2 (coincide when pred=0)
        u_zero = cell._compute_u(torch.zeros(B, C), u_h, u_m)
        torch.testing.assert_close(u_zero, torch.zeros(B, C), atol=ATOL, rtol=0)

        # Novel observation at reset → should produce nonzero u
        u_x    = torch.randn(B, C)
        u_novel = cell._compute_u(u_x, u_h, u_m)
        assert u_novel.abs().max().item() > 1e-3, \
            "Novel obs at episode start should produce nonzero u"

    def test_no_bias_in_W_pre_is_required(self, cell):
        """
        W_pre(0) = 0 exactly is what makes u_null = pred provably.
        This test guards against anyone accidentally adding a bias to OrthoLayer.
        """
        assert not any(
            name == 'bias' for name, _ in cell.W_pre.named_parameters()
        ), "OrthoLayer must have no bias — u_null = pred proof depends on W_pre(0)=0"

    def test_conditions_are_exact_not_approximate(self, cell):
        """
        Both conditions must hold to machine precision (< 1e-6), not just
        approximately.  If they hold only approximately, the intrinsic reward
        will fire on predicted states, which is backwards.
        """
        u_h = torch.randn(B, C)
        u_m = torch.randn(B, C)

        # Condition 1
        u1   = cell._compute_u(torch.zeros(B, C), u_h, u_m)
        err1 = (u1 - (u_h + u_m)).abs().max().item()
        assert err1 < 1e-6, f"Condition 1 error too large: {err1:.2e}"

        # Condition 2
        u2   = cell._compute_u(u_h + u_m, u_h, u_m)
        err2 = (u2 - (u_h + u_m)).abs().max().item()
        assert err2 < 1e-6, f"Condition 2 error too large: {err2:.2e}"


# ─────────────────────────────────────────────────────────────────────────────
# TestRIntr
# ─────────────────────────────────────────────────────────────────────────────

class TestRIntr:

    def test_shape(self, cell, zero_state):
        h, m = zero_state
        x = torch.randn(B, C)
        _, _, r_intr = cell(x, h, m)
        assert r_intr.shape == (B,), f"Expected (B,)={B,}, got {r_intr.shape}"

    def test_nonnegative(self, cell, zero_state):
        """r_intr is an L2 norm — must be ≥ 0."""
        h, m = zero_state
        x = torch.randn(B, C)
        _, _, r_intr = cell(x, h, m)
        assert (r_intr >= 0).all()

    def test_bounded_by_sqrt_C(self, cell, zero_state):
        """
        r_intr = ‖gate ⊙ innov‖₂ where gate, innov ∈ (−1,1)^C.
        Upper bound: ‖gate ⊙ innov‖₂ ≤ ‖gate‖₂ ≤ √C.
        This bound is tight only when all tanh outputs are ±1, which requires
        very large inputs.  For normal inputs it will be well below √C.
        """
        sqrt_C = math.sqrt(C)
        h, m = zero_state
        # Use large inputs to push tanh toward ±1 to stress-test the bound
        x = 10.0 * torch.randn(B, C)
        _, _, r_intr = cell(x, h, m)
        assert (r_intr <= sqrt_C + 1e-5).all(), \
            f"r_intr exceeded √C={sqrt_C:.3f}: max={r_intr.max():.3f}"

    def test_zero_when_ux_is_zero(self, cell):
        """
        r_intr = ‖u_actual − u_null‖₂.
        When u_x=0: u_actual = u_null = pred → r_intr = 0.
        Test via zero input x with fresh (h=0, m=0) so u_h=u_m=0 → u_x=0
        only if e_x is zeroed.  Better: test _compute_u directly for r_intr.
        We use a fresh cell and manually compute the equivalent.
        """
        torch.manual_seed(1)
        c = LMUCell(input_size=C, hidden_size=N, memory_size=D, theta=THETA)
        u_h = torch.randn(B, C)
        u_m = torch.randn(B, C)
        u_x = torch.zeros(B, C)

        u_actual = c._compute_u(u_x, u_h, u_m)
        u_null   = c._compute_u(torch.zeros_like(u_x), u_h, u_m)
        r = (u_actual - u_null).norm(dim=-1)
        torch.testing.assert_close(r, torch.zeros(B), atol=ATOL, rtol=0)

    def test_zero_when_ux_equals_pred(self, cell):
        """
        When u_x = pred: u_actual = u_null = pred → r_intr = 0.
        A correctly predicted observation contributes no intrinsic reward.
        """
        u_h = torch.randn(B, C)
        u_m = torch.randn(B, C)
        u_x = u_h + u_m   # u_x = pred

        u_actual = cell._compute_u(u_x, u_h, u_m)
        u_null   = cell._compute_u(torch.zeros_like(u_x), u_h, u_m)
        r = (u_actual - u_null).norm(dim=-1)
        torch.testing.assert_close(r, torch.zeros(B), atol=ATOL, rtol=0)

    def test_nonzero_for_novel_input(self, cell, zero_state):
        """r_intr > 0 for genuinely novel observations."""
        h, m = zero_state
        x = torch.randn(B, C)
        _, _, r_intr = cell(x, h, m)
        assert (r_intr > 1e-4).all(), \
            "Novel observations at episode start should produce nonzero r_intr"

    def test_isometry_means_Wpre_drops_from_magnitude(self, cell):
        """
        r_intr = ‖W_pre(gate⊙innov)‖₂ = ‖gate⊙innov‖₂ (isometry).
        Verify numerically: compute both sides and compare.
        """
        u_h = torch.randn(B, C)
        u_m = torch.randn(B, C)
        u_x = torch.randn(B, C)

        pred  = u_h + u_m
        gate  = torch.tanh(u_x)
        innov = torch.tanh(u_x - pred)
        expected = (gate * innov).norm(dim=-1)

        u_actual = cell._compute_u(u_x, u_h, u_m)
        u_null   = cell._compute_u(torch.zeros_like(u_x), u_h, u_m)
        actual   = (u_actual - u_null).norm(dim=-1)

        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=0)


# ─────────────────────────────────────────────────────────────────────────────
# TestExNorm
# ─────────────────────────────────────────────────────────────────────────────

class TestExNorm:

    def test_ex_norm_is_one_at_init(self, cell):
        """e_x_n = F.normalize(e_x) must have L2 norm = 1.0 always."""
        e_x_n = F.normalize(cell.e_x, dim=0)
        assert abs(e_x_n.norm().item() - 1.0) < 1e-6

    def test_ex_norm_is_one_after_gradient_update(self, cell, zero_state):
        """
        F.normalize is applied in forward, so the norm is always 1 regardless
        of what gradient descent does to the underlying e_x parameter.
        """
        opt = torch.optim.SGD([cell.e_x], lr=1e-2)
        h, m = zero_state
        for _ in range(5):
            opt.zero_grad()
            x = torch.randn(B, C)
            h_new, m_new, r = cell(x, h, m)
            r.sum().backward()
            opt.step()
            # After each step, the normalised version must still be unit norm
            e_x_n = F.normalize(cell.e_x, dim=0)
            assert abs(e_x_n.norm().item() - 1.0) < 1e-6

    def test_ex_channels_can_have_unequal_weights(self, cell):
        """
        Unit norm does NOT mean all channels equal.
        Some channels should dominate others (that's the inductive bias).
        """
        e_x_n = F.normalize(cell.e_x, dim=0)
        # With random init, std of channel weights should be > 0
        assert e_x_n.std().item() > 1e-3, \
            "e_x channels should have unequal weights — channel selection is the point"


# ─────────────────────────────────────────────────────────────────────────────
# TestEpisodeBoundary
# ─────────────────────────────────────────────────────────────────────────────

class TestEpisodeBoundary:
    """
    At episode start: h=0, m=0 → u_h=0, u_m=0 → pred=0.
    Condition 2 (u_x=pred) cannot hold (would require u_x=0 too).
    Therefore gate=tanh(u_x)≠0 and innov=tanh(u_x)≠0 for any nonzero obs.
    r_intr = ‖tanh(u_x)²‖₂ > 0, and ball identity is written at full strength.
    This is the core argument in Section 8 of the design doc.
    """

    def test_pred_is_zero_at_episode_start(self, cell, zero_state):
        """u_h = E_h(0) = 0, u_m = einsum(e_m, 0) = 0 → pred = 0."""
        h, m = zero_state
        u_h = cell.E_h(h)
        u_m = torch.einsum('d,bdc->bc', cell.e_m, m)
        pred = u_h + u_m
        torch.testing.assert_close(pred, torch.zeros(B, C), atol=ATOL, rtol=0)

    def test_gate_open_at_episode_start(self, cell, zero_state):
        """With pred=0, gate=tanh(u_x) is fully determined by the observation."""
        h, m = zero_state
        x = torch.randn(B, C)
        e_x_n = F.normalize(cell.e_x, dim=0)
        u_x   = x * e_x_n

        u_h = cell.E_h(h)
        u_m = torch.einsum('d,bdc->bc', cell.e_m, m)

        gate  = torch.tanh(u_x)
        innov = torch.tanh(u_x - (u_h + u_m))   # = tanh(u_x) since pred=0

        # gate = innov at episode start (both equal tanh(u_x))
        torch.testing.assert_close(gate, innov, atol=ATOL, rtol=0)

    def test_rintr_nonzero_at_episode_start(self, cell, zero_state):
        """Nonzero observation at episode start must produce r_intr > 0."""
        h, m = zero_state
        x = torch.randn(B, C)
        _, _, r_intr = cell(x, h, m)
        assert (r_intr > 1e-3).all()

    def test_rintr_at_start_equals_tanh_ux_squared_norm(self, cell, zero_state):
        """
        At episode start (pred=0):
          r_intr = ‖tanh(u_x) ⊙ tanh(u_x)‖₂ = ‖tanh(u_x)²‖₂
        This simplification follows from gate=innov=tanh(u_x) when pred=0.
        """
        h, m = zero_state
        x = torch.randn(B, C)

        e_x_n = F.normalize(cell.e_x, dim=0)
        u_x   = x * e_x_n
        expected = (torch.tanh(u_x) ** 2).norm(dim=-1)

        _, _, r_intr = cell(x, h, m)
        torch.testing.assert_close(r_intr, expected, atol=1e-5, rtol=0)

    def test_full_forward_at_episode_start_matches_compute_u(self, cell, zero_state):
        """Cross-check: full forward at reset matches direct _compute_u call."""
        h, m = zero_state
        x = torch.randn(B, C)

        _, _, r_intr_forward = cell(x, h, m)

        e_x_n  = F.normalize(cell.e_x, dim=0)
        u_x    = x * e_x_n
        u_h    = cell.E_h(h)
        u_m    = torch.einsum('d,bdc->bc', cell.e_m, m)
        u_act  = cell._compute_u(u_x, u_h, u_m)
        u_null = cell._compute_u(torch.zeros_like(u_x), u_h, u_m)
        r_direct = (u_act - u_null).norm(dim=-1)

        torch.testing.assert_close(r_intr_forward, r_direct, atol=ATOL, rtol=0)


# ─────────────────────────────────────────────────────────────────────────────
# TestLMUCellForward
# ─────────────────────────────────────────────────────────────────────────────

class TestLMUCellForward:

    def test_output_shapes(self, cell, zero_state):
        h, m = zero_state
        x = torch.randn(B, C)
        h_new, m_new, r_intr = cell(x, h, m)
        assert h_new.shape  == (B, N),    f"h_new: {h_new.shape}"
        assert m_new.shape  == (B, D, C), f"m_new: {m_new.shape}"
        assert r_intr.shape == (B,),      f"r_intr: {r_intr.shape}"

    def test_returns_three_values(self, cell, zero_state):
        """Guard against accidentally reverting to the 2-return baseline."""
        h, m = zero_state
        result = cell(torch.randn(B, C), h, m)
        assert len(result) == 3, \
            f"forward should return (h, m, r_intr) — got {len(result)} values"

    def test_initial_state_shapes(self, cell):
        h, m = cell.initial_state(B, torch.device('cpu'))
        assert h.shape == (B, N)
        assert m.shape == (B, D, C)

    def test_initial_state_is_zero(self, cell):
        h, m = cell.initial_state(B, torch.device('cpu'))
        assert h.abs().max() == 0.0
        assert m.abs().max() == 0.0

    def test_legendre_update_uses_u_actual_not_u_x(self, cell, zero_state):
        """
        m_new = Ā·m_prev + B̄·u_actual.
        Verify the memory update uses u_actual (gated) not the raw u = u_x+u_h+u_m.
        We check by zeroing u_x and confirming the update matches pred-based u_actual.
        """
        h, m = zero_state
        # Zero input → u_x=0 → u_actual = pred = 0 (since h=m=0 too)
        x = torch.zeros(B, C)
        _, m_new, _ = cell(x, h, m)

        # With u_actual=0 and m_prev=0: m_new = Ā·0 + B̄·0 = 0
        torch.testing.assert_close(m_new, torch.zeros(B, D, C), atol=ATOL, rtol=0)

    def test_A_matrix_spectral_radius_stable(self, cell):
        """Ā spectral radius must be < 1 for stable memory dynamics."""
        rho = torch.linalg.eigvals(cell.A).abs().max().item()
        assert rho < 1.0, f"Unstable Ā: spectral radius {rho:.4f}"

    def test_AB_not_trainable(self, cell):
        """Frozen Ā, B̄ — training them destroys Legendre guarantees."""
        assert not cell.A.requires_grad
        assert not cell.B.requires_grad

    def test_W_pre_orthogonality_preserved_through_forward(self, cell, zero_state):
        """
        Forward pass itself does not corrupt orthogonality.
        Corruption only happens if W_pre is incorrectly included in Adam.
        """
        h, m = zero_state
        x = torch.randn(B, C)
        cell(x, h, m)
        assert cell.W_pre.orthogonality_error() < 1e-6

    def test_stability_over_many_steps(self, cell):
        """Norms of h and m should not explode over 100 steps."""
        h, m = cell.initial_state(B, torch.device('cpu'))
        for _ in range(100):
            x = torch.randn(B, C)
            with torch.no_grad():
                h, m, _ = cell(x, h, m)
        assert h.norm().item()  < 1e4, "h diverged"
        assert m.norm().item()  < 1e4, "m diverged"

    def test_h_is_bounded_by_tanh(self, cell, zero_state):
        """h = tanh(...) so |h| <= 1.0 always.
        FIXED: strict < 1.0 → <= 1.0.  torch.tanh returns *exactly* 1.0 in
        float32 for large inputs — the open interval (-1,1) only holds in real
        arithmetic.  The stability property we actually care about is that h
        doesn't exceed 1.0 in magnitude, which <= tests correctly.
        """
        h, m = zero_state
        x = 10.0 * torch.randn(B, C)   # large input to saturate tanh
        h_new, _, _ = cell(x, h, m)
        assert (h_new.abs() <= 1.0).all()

    def test_zero_input_does_not_crash(self, cell, zero_state):
        """Edge case: all-zero input + all-zero state should run cleanly."""
        h, m = zero_state
        x = torch.zeros(B, C)
        h_new, m_new, r_intr = cell(x, h, m)
        assert not torch.isnan(h_new).any()
        assert not torch.isnan(m_new).any()
        assert not torch.isnan(r_intr).any()


# ─────────────────────────────────────────────────────────────────────────────
# TestGradientFlow
# ─────────────────────────────────────────────────────────────────────────────

class TestGradientFlow:
    """
    These tests verify the gradient wiring described in Section 6 of the doc.
    They matter most at Step 2 (when r_intr is in the loss), but are checked
    now to confirm the graph structure is correct before β > 0.
    """

    def test_gradient_flows_through_h_new(self, cell, zero_state):
        """Policy loss must be able to reach encoder and LMU params via h."""
        h, m = zero_state
        x = torch.randn(B, C, requires_grad=False)
        h_new, _, _ = cell(x, h, m)
        h_new.sum().backward()
        # e_x should have gradient (flows from h via W_x and e_x normalization)
        assert cell.e_x.grad is not None

    def test_gradient_flows_through_r_intr(self, cell, zero_state):
        """
        r_intr must be differentiable.  At Step 2, the minimize-r_intr loss
        flows gradients through r_intr to E_h, e_m (world model training).
        FIXED: check cell.e_m.grad instead of cell.E_h.weight.grad.
        spectral_norm wraps E_h.weight into a computed non-leaf tensor
        (weight_orig / sigma); PyTorch only populates .grad for leaf tensors,
        so E_h.weight.grad is always None.  e_m is a plain nn.Parameter.
        """
        h, m = zero_state
        x = torch.randn(B, C)
        _, _, r_intr = cell(x, h, m)
        r_intr.sum().backward()
        assert cell.e_m.grad is not None, "Gradient must reach e_m via r_intr"

    def test_u_null_in_live_graph(self, cell):
        """
        u_null = _compute_u(zeros, u_h, u_m) depends on u_h and u_m.
        Gradient from r_intr flows through u_null → pred → u_m → e_m.

        FIXED (second pass): must use nonzero m_prev.
        ∂u_m/∂e_m[d] = m_prev[:, d, :] — so if m_prev is the zero initial
        state, the Jacobian is identically zero and e_m.grad = 0 regardless
        of graph connectivity.  This was a test-input bug, not a code bug.
        Using torch.randn for m gives a nonzero Jacobian and correctly verifies
        that the u_null path actually propagates gradient to e_m.
        """
        x = torch.randn(B, C)
        h = torch.zeros(B, N)
        m = torch.randn(B, D, C)   # nonzero — ∂u_m/∂e_m = m ≠ 0

        e_x_n  = F.normalize(cell.e_x, dim=0)
        u_x    = x * e_x_n
        u_h    = cell.E_h(h)
        u_m    = torch.einsum('d,bdc->bc', cell.e_m, m)
        u_act  = cell._compute_u(u_x, u_h, u_m)
        u_null = cell._compute_u(torch.zeros_like(u_x), u_h, u_m)
        r      = (u_act - u_null).norm(dim=-1)

        r.sum().backward()

        assert cell.e_m.grad is not None
        assert cell.e_m.grad.abs().max() > 0, (
            "e_m must receive gradient via the u_null path "
            "(u_null = pred = u_h + u_m, and u_m = einsum(e_m, m))"
        )

    def test_e_x_grad_not_through_u_null(self, cell, zero_state):
        """
        u_null = _compute_u(zeros, u_h, u_m) — no dependence on e_x.
        Therefore minimize-r_intr has no counterforce through u_null on e_x.
        This is the documented pathology (Section 6 table: e_x uncontested).
        The test confirms the pathology exists — we guard against it via F.normalize,
        not by removing the gradient.
        """
        h, m = zero_state
        x = torch.randn(B, C)

        e_x_n  = F.normalize(cell.e_x, dim=0)
        u_x    = x * e_x_n
        u_h    = cell.E_h(h)
        u_m    = torch.einsum('d,bdc->bc', cell.e_m, m)
        # Only compute u_null — which does NOT depend on u_x or e_x
        u_null = cell._compute_u(torch.zeros_like(u_x), u_h, u_m)
        u_null.sum().backward()

        # e_x gradient should be None — it does not appear in u_null
        assert cell.e_x.grad is None, \
            "e_x must not receive gradient from u_null — confirmed pathology"

    def test_W_pre_grad_populated_before_ortho_update(self, cell, zero_state):
        """After backward, W_pre.weights.grad must be non-None and nonzero."""
        h, m = zero_state
        x = torch.randn(B, C)
        _, _, r_intr = cell(x, h, m)
        r_intr.sum().backward()
        assert cell.W_pre.weights.grad is not None
        assert cell.W_pre.weights.grad.abs().max() > 0

    def test_W_pre_grad_zeroed_after_ortho_update(self, cell, zero_state):
        """ortho_update must zero W_pre.grad so Adam gets nothing."""
        h, m = zero_state
        x = torch.randn(B, C)
        _, _, r_intr = cell(x, h, m)
        r_intr.sum().backward()
        cell.W_pre.ortho_update(lr=1e-3)
        assert cell.W_pre.weights.grad.abs().max() == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# TestOptimizerExclusion
# ─────────────────────────────────────────────────────────────────────────────

class TestOptimizerExclusion:
    """
    Verifies the optimizer split: Adam updates all params except W_pre;
    ortho_update changes only W_pre.
    """

    def _make_split_optimizer(self, cell):
        ortho_params = set(cell.W_pre.parameters())
        main_params  = [p for p in cell.parameters() if p not in ortho_params]
        return torch.optim.Adam(main_params, lr=1e-3, eps=1e-5)

    def test_adam_does_not_update_W_pre(self, cell, zero_state):
        """After optimizer.step() WITHOUT ortho_update, W_pre must be unchanged."""
        optimizer = self._make_split_optimizer(cell)
        h, m = zero_state
        x = torch.randn(B, C)

        W_before = cell.W_pre.weights.data.clone()
        optimizer.zero_grad()
        h_new, _, r = cell(x, h, m)
        h_new.sum().backward()
        optimizer.step()   # ortho_update NOT called
        W_after = cell.W_pre.weights.data

        torch.testing.assert_close(W_before, W_after, atol=1e-8, rtol=0,
            msg="Adam must not update W_pre weights")

    def test_ortho_update_changes_only_W_pre(self, cell, zero_state):
        """ortho_update must not affect any other parameter."""
        h, m = zero_state
        x = torch.randn(B, C)

        # Snapshot all non-W_pre params
        other_params = {
            n: p.data.clone()
            for n, p in cell.named_parameters()
            if 'W_pre' not in n
        }

        _, _, r = cell(x, h, m)
        r.sum().backward()
        cell.W_pre.ortho_update(lr=1e-3)

        for name, before in other_params.items():
            after = dict(cell.named_parameters())[name].data
            torch.testing.assert_close(before, after, atol=1e-8, rtol=0,
                msg=f"ortho_update must not touch {name}")

    def test_W_pre_not_in_adam_param_groups(self, cell):
        """Verify W_pre parameter ID is absent from Adam's param groups."""
        optimizer = self._make_split_optimizer(cell)
        adam_param_ids = {
            id(p) for group in optimizer.param_groups for p in group['params']
        }
        for p in cell.W_pre.parameters():
            assert id(p) not in adam_param_ids, \
                "W_pre parameter found in Adam — must be excluded"


# ─────────────────────────────────────────────────────────────────────────────
# TestLMUSequenceWrapper
# ─────────────────────────────────────────────────────────────────────────────

class TestLMUSequenceWrapper:

    @pytest.fixture
    def lmu(self):
        torch.manual_seed(0)
        return LMU(input_size=C, hidden_size=N, memory_size=D, theta=THETA)

    def test_output_shapes(self, lmu):
        x = torch.randn(B, T, C)
        out, r_intrs, (h, m) = lmu(x)
        assert out.shape     == (B, T, N),    f"out: {out.shape}"
        assert r_intrs.shape == (B, T),        f"r_intrs: {r_intrs.shape}"
        assert h.shape       == (B, N)
        assert m.shape       == (B, D, C)

    def test_returns_three_values(self, lmu):
        """Guard: LMU wrapper must return (out, r_intrs, state), not (out, state)."""
        x = torch.randn(B, T, C)
        result = lmu(x)
        assert len(result) == 3, \
            f"LMU.forward should return 3 values — got {len(result)}"

    def test_r_intrs_nonnegative(self, lmu):
        x = torch.randn(B, T, C)
        _, r_intrs, _ = lmu(x)
        assert (r_intrs >= 0).all()

    def test_r_intrs_bounded(self, lmu):
        sqrt_C = math.sqrt(C)
        x = 5.0 * torch.randn(B, T, C)
        _, r_intrs, _ = lmu(x)
        assert (r_intrs <= sqrt_C + 1e-5).all()

    def test_state_passthrough(self, lmu):
        """Passing final state as initial state of next call is consistent.
        FIXED: call lmu.eval() before the test.
        spectral_norm updates its power-iteration u vector in-place during
        every forward call when training=True.  lmu(x1) and lmu(cat([x1,x2]))
        leave u in different states → different sigma → different normalized
        weight → different outputs even for the same input subsequence.
        eval() freezes u (spectral_norm skips power iteration), making all
        forward calls deterministic for the same weight.
        """
        lmu.eval()
        x1 = torch.randn(B, T, C)
        x2 = torch.randn(B, T, C)
        _, _, state1        = lmu(x1)
        out_fresh, _, _     = lmu(torch.cat([x1, x2], dim=1))
        out_continued, _, _ = lmu(x2, state=state1)
        torch.testing.assert_close(
            out_fresh[:, T:], out_continued, atol=1e-4, rtol=0
        )
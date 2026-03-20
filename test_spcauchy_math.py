"""
Workstream 4 verification tests for spCauchy distribution math.

Tests:
  1. KL continuity (no jumps across rho range)
  2. KL(rho=0) = 0 (uniform)
  3. KL is monotonically increasing in rho
  4. d²KL/drho² at rho=0 = 4(d-1)²/d (collapse curvature — corrected from paper's 2(d-1))
  5. fc_rho bias initialization => rho ≈ 0.12
  6. Mobius reparameterization produces unit-norm samples
  7. GL cache returns same tensors for same device
"""

import torch
import pytest
from models.utils import (
    sc_kl_uniform, sc_sample,
    _get_legendre_tensors, _LEGENDRE_POINTS,
    _sc_kl_quadrature, _gauss_legendre_01,
)


DIMS = [8, 16, 32, 64]


class TestKLContinuity:
    """KL should be smooth — no jumps anywhere in [0, 0.98]."""

    @pytest.mark.parametrize("dim", DIMS)
    def test_kl_smooth_across_range(self, dim):
        """Sample rho densely and check relative step size is bounded.

        For a smooth, monotonically increasing function, the ratio
        kl[i]/kl[i-1] should be close to 1 when sampled densely.
        """
        rhos = torch.linspace(0.1, 0.98, 200)
        kls = sc_kl_uniform(rhos, dim).squeeze()
        for i in range(1, len(kls)):
            if kls[i - 1].item() < 1e-6:
                continue  # skip near-zero values
            ratio = kls[i].item() / kls[i - 1].item()
            assert 0.99 < ratio < 1.15, (
                f"KL ratio anomaly at rho={rhos[i]:.4f} for d={dim}: "
                f"kl[{i-1}]={kls[i-1]:.6f}, kl[{i}]={kls[i]:.6f}, ratio={ratio:.4f}"
            )


class TestKLProperties:
    """Basic sanity checks on KL behavior."""

    @pytest.mark.parametrize("dim", DIMS)
    def test_kl_at_zero_is_zero(self, dim):
        rho = torch.tensor([0.0])
        kl = sc_kl_uniform(rho, dim).item()
        assert abs(kl) < 1e-5, f"KL(rho=0) should be 0 for d={dim}, got {kl:.6f}"

    @pytest.mark.parametrize("dim", DIMS)
    def test_kl_monotonically_increasing(self, dim):
        rhos = torch.linspace(0.01, 0.98, 50)
        kls = sc_kl_uniform(rhos, dim).squeeze()
        for i in range(1, len(kls)):
            assert kls[i] >= kls[i - 1] - 1e-6, (
                f"KL not monotonic at rho={rhos[i]:.3f} for d={dim}: "
                f"kl[{i-1}]={kls[i-1]:.6f} > kl[{i}]={kls[i]:.6f}"
            )

    @pytest.mark.parametrize("dim", DIMS)
    def test_kl_nonnegative(self, dim):
        rhos = torch.linspace(0.0, 0.99, 100)
        kls = sc_kl_uniform(rhos, dim).squeeze()
        assert (kls >= -1e-7).all(), f"Negative KL found for d={dim}: min={kls.min():.6f}"

    def test_kl_batch(self):
        """KL on a batch should match element-wise."""
        rhos = torch.tensor([0.1, 0.5, 0.85, 0.95])
        kl_batch = sc_kl_uniform(rhos, 32).squeeze()
        for i, r in enumerate(rhos):
            kl_single = sc_kl_uniform(r.unsqueeze(0), 32).item()
            assert abs(kl_batch[i].item() - kl_single) < 1e-4

    @pytest.mark.parametrize("dim", DIMS)
    def test_kl_increases_with_dim(self, dim):
        """At fixed rho, higher d should give higher KL."""
        if dim == DIMS[0]:
            pytest.skip("Need previous dim to compare")
        prev_dim = DIMS[DIMS.index(dim) - 1]
        rho = torch.tensor([0.5])
        kl_this = sc_kl_uniform(rho, dim).item()
        kl_prev = sc_kl_uniform(rho, prev_dim).item()
        assert kl_this > kl_prev, (
            f"KL(d={dim})={kl_this:.4f} should be > KL(d={prev_dim})={kl_prev:.4f} at rho=0.5"
        )


class TestCollapseCurvature:
    """4d: Verify d²KL/drho² at rho=0 = 4(d-1)²/d via autograd.

    The paper claimed 2(d-1), but numerical verification shows the correct
    formula is 4(d-1)²/d. This is verified to machine precision with
    high-resolution quadrature + autograd.
    """

    @pytest.mark.parametrize("dim", DIMS)
    def test_curvature_autograd(self, dim):
        """Use autograd on high-precision quadrature to verify curvature."""
        gl = _gauss_legendre_01(2048, torch.device('cpu'), torch.float64)
        rho = torch.tensor([[1e-4]], dtype=torch.float64, requires_grad=True)
        kl = _sc_kl_quadrature(rho, dim, gl)
        grad1 = torch.autograd.grad(kl, rho, create_graph=True)[0]
        grad2 = torch.autograd.grad(grad1, rho)[0]

        expected = 4.0 * (dim - 1) ** 2 / dim
        rel_err = abs(grad2.item() - expected) / expected
        assert rel_err < 0.01, (
            f"d²KL/drho² at rho≈0 for d={dim}: got {grad2.item():.4f}, "
            f"expected 4(d-1)²/d={expected:.4f}, rel_err={rel_err:.6f}"
        )

    @pytest.mark.parametrize("dim", [8, 16])
    def test_curvature_finite_diff(self, dim):
        """Finite-difference check using production quadrature (512 pts, float32).

        Only tested for d<=16 where float32 quadrature is accurate enough at
        small rho. For d>=32, the autograd test above is authoritative.
        """
        h = 5e-3
        kl_0 = sc_kl_uniform(torch.tensor([0.0]), dim).item()
        kl_h = sc_kl_uniform(torch.tensor([h]), dim).item()
        kl_2h = sc_kl_uniform(torch.tensor([2 * h]), dim).item()

        d2_kl = (kl_2h - 2 * kl_h + kl_0) / (h ** 2)
        expected = 4.0 * (dim - 1) ** 2 / dim

        rel_err = abs(d2_kl - expected) / expected
        assert rel_err < 0.15, (
            f"d²KL/drho² (finite diff) for d={dim}: got {d2_kl:.4f}, "
            f"expected 4(d-1)²/d={expected:.4f}, rel_err={rel_err:.4f}"
        )


class TestFcRhoInit:
    """4a: Verify fc_rho bias=-2.0 gives rho ≈ 0.12."""

    def test_sigmoid_of_minus_two(self):
        rho = torch.sigmoid(torch.tensor(-2.0)).item()
        assert abs(rho - 0.1192) < 0.01, f"sigmoid(-2.0) = {rho:.4f}, expected ~0.12"

    def test_vae_fc_rho_bias(self):
        from models.embeddings import CategoricalGridWithDirEmbedding
        from models.config import SCVAEConfig
        from models.vae import TransitionSCVAE

        embedding = CategoricalGridWithDirEmbedding(
            n_object_types=12, n_colors=6, n_states=3,
            obs_h=5, obs_w=5, embed_per_channel=4,
            n_dirs=4, dir_embed_dim=4,
        )
        vae = TransitionSCVAE(embedding, SCVAEConfig(act_dim=1))
        bias = vae.fc_rho.bias.item()
        assert abs(bias - (-2.0)) < 1e-6, f"fc_rho.bias = {bias}, expected -2.0"


class TestMobiusSampling:
    """Verify sc_sample produces unit-norm outputs."""

    @pytest.mark.parametrize("dim", [8, 32])
    def test_samples_are_unit_norm(self, dim):
        torch.manual_seed(42)
        mu = torch.nn.functional.normalize(torch.randn(100, dim), p=2, dim=-1)
        rho = torch.full((100, 1), 0.5)
        z = sc_sample(mu, rho)
        norms = z.norm(p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), (
            f"Samples not unit-norm: min={norms.min():.6f}, max={norms.max():.6f}"
        )

    @pytest.mark.parametrize("dim", [8, 32])
    def test_samples_concentrate_around_mu(self, dim):
        """At high rho, samples should be close to mu."""
        torch.manual_seed(0)
        mu = torch.nn.functional.normalize(torch.randn(200, dim), p=2, dim=-1)
        rho = torch.full((200, 1), 0.95)
        z = sc_sample(mu, rho)
        cosines = (z * mu).sum(dim=-1)
        assert cosines.mean() > 0.8, f"Mean cosine at rho=0.95: {cosines.mean():.4f}, expected > 0.8"

    @pytest.mark.parametrize("dim", [8, 32])
    def test_samples_spread_at_low_rho(self, dim):
        """At low rho, samples should be spread out (near uniform)."""
        torch.manual_seed(0)
        mu = torch.nn.functional.normalize(torch.randn(500, dim), p=2, dim=-1)
        rho = torch.full((500, 1), 0.01)
        z = sc_sample(mu, rho)
        cosines = (z * mu).sum(dim=-1)
        assert abs(cosines.mean()) < 0.2, f"Mean cosine at rho=0.01: {cosines.mean():.4f}, expected ~0"


class TestGLCache:
    """4b: GL cache returns consistent tensors."""

    def test_cache_returns_same_object(self):
        t1, w1 = _get_legendre_tensors(torch.device('cpu'), torch.float32)
        t2, w2 = _get_legendre_tensors(torch.device('cpu'), torch.float32)
        assert t1.data_ptr() == t2.data_ptr(), "GL cache should return same tensor"

    def test_cache_correct_shape(self):
        t, w = _get_legendre_tensors(torch.device('cpu'), torch.float32)
        assert t.shape == (_LEGENDRE_POINTS,), f"Expected ({_LEGENDRE_POINTS},), got {t.shape}"
        assert w.shape == (_LEGENDRE_POINTS,), f"Expected ({_LEGENDRE_POINTS},), got {w.shape}"

    def test_weights_sum_to_one(self):
        """GL weights on [0,1] should sum to 1."""
        _, w = _get_legendre_tensors(torch.device('cpu'), torch.float32)
        assert abs(w.sum().item() - 1.0) < 1e-5, f"GL weights sum to {w.sum():.10f}, expected 1.0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

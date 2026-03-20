"""
Workstream 4 verification tests for spCauchy distribution math.

Tests:
  1. Quadrature and asymptotic KL branches agree at rho=0.9 boundary
  2. KL(rho=0) = 0 (uniform)
  3. KL is monotonically increasing in rho
  4. d²KL/drho² at rho=0 ≈ 2(d-1) (collapse curvature proposition)
  5. fc_rho bias initialization => rho ≈ 0.12
  6. Mobius reparameterization produces unit-norm samples
  7. GL cache returns same tensors for same device
"""

import torch
import pytest
from models.utils import sc_kl_uniform, sc_sample, _get_legendre_tensors


DIMS = [8, 16, 32, 64]


class TestKLBranchAgreement:
    """4c: Verify quadrature/asymptotic KL branches agree at rho=0.9 +/- eps."""

    @pytest.mark.parametrize("dim", DIMS)
    def test_kl_branches_agree_at_boundary(self, dim):
        rho_lo = torch.tensor([0.8999])
        rho_hi = torch.tensor([0.9001])

        kl_lo = sc_kl_uniform(rho_lo, dim).item()
        kl_hi = sc_kl_uniform(rho_hi, dim).item()

        # Should be close (no discontinuity at boundary)
        rel_diff = abs(kl_hi - kl_lo) / max(abs(kl_lo), 1e-8)
        assert rel_diff < 0.01, (
            f"KL discontinuity at rho=0.9 for d={dim}: "
            f"kl(0.8999)={kl_lo:.6f}, kl(0.9001)={kl_hi:.6f}, rel_diff={rel_diff:.4f}"
        )

    @pytest.mark.parametrize("dim", DIMS)
    def test_kl_branches_agree_at_exact_boundary(self, dim):
        """Test values just below and just above the 0.9 threshold."""
        eps = 1e-4
        rho_below = torch.tensor([0.9 - eps])
        rho_above = torch.tensor([0.9 + eps])

        kl_below = sc_kl_uniform(rho_below, dim).item()
        kl_above = sc_kl_uniform(rho_above, dim).item()

        rel_diff = abs(kl_above - kl_below) / max(abs(kl_below), 1e-8)
        assert rel_diff < 0.005, (
            f"KL jump at rho=0.9±eps for d={dim}: "
            f"kl_below={kl_below:.6f}, kl_above={kl_above:.6f}, rel_diff={rel_diff:.6f}"
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
        rhos = torch.linspace(0.01, 0.99, 50)
        kls = sc_kl_uniform(rhos, dim).squeeze()
        for i in range(1, len(kls)):
            assert kls[i] >= kls[i - 1] - 1e-6, (
                f"KL not monotonic at rho={rhos[i]:.3f} for d={dim}: "
                f"kl[{i-1}]={kls[i-1]:.6f} > kl[{i}]={kls[i]:.6f}"
            )

    @pytest.mark.parametrize("dim", DIMS)
    def test_kl_nonnegative(self, dim):
        rhos = torch.linspace(0.0, 0.999, 100)
        kls = sc_kl_uniform(rhos, dim).squeeze()
        assert (kls >= -1e-7).all(), f"Negative KL found for d={dim}: min={kls.min():.6f}"

    def test_kl_batch(self):
        """KL on a batch should match element-wise."""
        rhos = torch.tensor([0.1, 0.5, 0.85, 0.95])
        kl_batch = sc_kl_uniform(rhos, 32).squeeze()
        for i, r in enumerate(rhos):
            kl_single = sc_kl_uniform(r.unsqueeze(0), 32).item()
            assert abs(kl_batch[i].item() - kl_single) < 1e-5


class TestCollapseCurvature:
    """4d: Verify d²KL/drho² at rho=0 = 2(d-1) via finite differences."""

    @pytest.mark.parametrize("dim", DIMS)
    def test_curvature_at_zero(self, dim):
        h = 1e-3
        rho_m = torch.tensor([0.0])
        rho_p = torch.tensor([h])
        rho_pp = torch.tensor([2 * h])

        kl_0 = sc_kl_uniform(rho_m, dim).item()
        kl_h = sc_kl_uniform(rho_p, dim).item()
        kl_2h = sc_kl_uniform(rho_pp, dim).item()

        # Second derivative via central finite difference: (f(2h) - 2f(h) + f(0)) / h²
        d2_kl = (kl_2h - 2 * kl_h + kl_0) / (h ** 2)
        expected = 2.0 * (dim - 1)

        rel_err = abs(d2_kl - expected) / expected
        assert rel_err < 0.05, (
            f"d²KL/drho² at rho=0 for d={dim}: got {d2_kl:.4f}, expected {expected:.4f}, "
            f"rel_err={rel_err:.4f}"
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


class TestGLCache:
    """4b: GL cache returns consistent tensors."""

    def test_cache_returns_same_object(self):
        t1, w1 = _get_legendre_tensors(torch.device('cpu'), torch.float32)
        t2, w2 = _get_legendre_tensors(torch.device('cpu'), torch.float32)
        assert t1.data_ptr() == t2.data_ptr(), "GL cache should return same tensor"

    def test_cache_correct_shape(self):
        t, w = _get_legendre_tensors(torch.device('cpu'), torch.float32)
        assert t.shape == (64,), f"Expected (64,), got {t.shape}"
        assert w.shape == (64,), f"Expected (64,), got {w.shape}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

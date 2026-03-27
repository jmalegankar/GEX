"""Component 1 test: WynerLBSVAE slot-conditioned prior."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.wyner import WynerLBSVAE

def test_slot_conditioned_prior():
    torch.manual_seed(42)

    wyner = WynerLBSVAE(
        recon_dim=288, mu_dim=32, latent_dim=64,
        latent_tokens=1, decode_hidden=128, pos_embed_dim=16, free_bits=0.0,
    )
    wyner.eval()

    B, K, D = 4, 8, 64
    h_prev = torch.zeros(B, 1, D)
    mu = torch.randn(B, 32)
    slots = torch.zeros(B, K, D)  # episode start — all zeros

    out = wyner.forward(h_prev, mu, mu_next=None, slots=slots)

    # Shape assertions
    assert out.w.shape == (B, 1, D), f"w shape: {out.w.shape}"
    assert out.posterior_mu.shape == (B, D), f"posterior_mu shape: {out.posterior_mu.shape}"
    assert out.prior_mu.shape == (B, D), f"prior_mu shape: {out.prior_mu.shape}"
    assert out.prior_logvar.shape == (B, D), f"prior_logvar shape: {out.prior_logvar.shape}"
    print("Shape assertions PASSED")

    # Sanity: prior from zero slots should be roughly isotropic
    prior_mu_norm = out.prior_mu.norm().item()
    prior_var_mean = out.prior_logvar.exp().mean().item()
    print(f"prior_mu  norm: {prior_mu_norm:.4f}")
    print(f"prior_var mean: {prior_var_mean:.4f}")
    assert prior_mu_norm < 5.0, f"prior_mu norm too large at zero slots: {prior_mu_norm}"

    # KL should be computable and finite
    loss = wyner.loss(out)
    assert not torch.isnan(loss.kl_loss).any(), "NaN in KL loss"
    assert loss.kl_loss.shape == (B,), f"KL shape: {loss.kl_loss.shape}"
    kl_mean = loss.kl_loss.mean().item()
    print(f"KL at zero slots: {kl_mean:.4f}")
    assert kl_mean > 0, f"KL should be positive, got {kl_mean}"

    # ValueError when slots=None
    try:
        wyner.forward(h_prev, mu, mu_next=None, slots=None)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"ValueError correctly raised: {str(e)[:60]}...")

    # Non-zero slots produce different prior
    slots_filled = torch.randn(B, K, D)
    out2 = wyner.forward(h_prev, mu, mu_next=None, slots=slots_filled)
    prior_diff = (out2.prior_mu - out.prior_mu).norm().item()
    print(f"Prior mu diff (zero vs filled slots): {prior_diff:.4f}")
    assert prior_diff > 0.01, f"Prior should change with different slots, diff={prior_diff}"

    # Reconstruction loss computable
    recon_target = torch.randn(B, 288)
    full_loss = wyner.loss(out, recon_target=recon_target)
    assert full_loss.recon_loss is not None
    assert not torch.isnan(full_loss.recon_loss).any()
    print(f"Recon loss: {full_loss.recon_loss.mean().item():.4f}")

    print("\nAll Component 1 tests PASSED.")


if __name__ == "__main__":
    test_slot_conditioned_prior()

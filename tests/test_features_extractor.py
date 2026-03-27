"""Component 4 test: HSWVIMEFeaturesExtractor with proj_pi."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from hswvime_ppo.policies import HSWVIMEFeaturesExtractor


def test_features_extractor():
    ext = HSWVIMEFeaturesExtractor(observation_space=None, mu_dim=32, slot_dim=64)
    B = 4

    slots = torch.randn(B, 8, 64)
    pi = torch.randn(B, 32)

    out = ext(slots, pi)
    assert out.shape == (B, 96), f"Expected (B, 96), got {out.shape}"
    assert not torch.isnan(out).any()
    print(f"Output shape: {out.shape} PASSED")
    print(f"features_dim: {ext.features_dim}")
    assert ext.features_dim == 96, f"Expected 96, got {ext.features_dim}"

    # Gradient flows through pi (PPO path) but not through slots
    pi_grad = torch.randn(B, 32, requires_grad=True)
    slots_no_grad = slots.detach()
    out2 = ext(slots_no_grad, pi_grad)
    loss = out2.sum()
    loss.backward()
    assert pi_grad.grad is not None, "Gradient must flow through pi"
    print(f"Gradient through pi: norm={pi_grad.grad.norm().item():.4f} PASSED")

    # proj_pi exists and has correct shape
    assert ext.proj_pi.in_features == 32
    assert ext.proj_pi.out_features == 64
    print(f"proj_pi: Linear(32 -> 64) PASSED")

    print("\nAll Component 4 tests PASSED.")


if __name__ == "__main__":
    test_features_extractor()

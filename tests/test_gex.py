import torch
import numpy as np
from intrinsic_reward.geodesic_bonus import GeodesicExplorationBonus

#positive
def test_first_visit_positive():
    gex = GeodesicExplorationBonus(mu_dim=8)
    mu = torch.randn(8)
    mu = mu / mu.norm()

    r, _ = gex.step(mu)
    assert r > 0



def test_repeat_same_transition_zero_epi():
    gex = GeodesicExplorationBonus(mu_dim=8, k=1)

    mu = torch.randn(8)
    mu = mu / mu.norm()

    r1, _ = gex.step(mu)
    r2, _ = gex.step(mu)

    assert r2 <= r1
    assert np.isclose(r2, 0.0, atol=1e-6)  # cosine distance = 0 → episodic = 0

def test_lifetime_decay_isolated():
    from intrinsic_reward.geodesic_bonus import AngularPseudoCounts

    pc = AngularPseudoCounts(mu_dim=8)

    mu = torch.randn(8)
    mu = mu / mu.norm()

    b1 = pc.bonus(mu)
    pc.increment(mu)

    b2 = pc.bonus(mu)
    pc.increment(mu)

    b3 = pc.bonus(mu)

    assert b3 < b2 < b1

def test_episodic_reset_restores_novelty():
    gex = GeodesicExplorationBonus(mu_dim=8, k=1)

    mu = torch.randn(8)
    mu = mu / mu.norm()

    r1, _ = gex.step(mu)
    # print(f"First visit reward: {r1:.4f}")
    r2, _ = gex.step(mu)
    # print(f"Second visit reward: {r2:.4f}")

    gex.reset()

    r3, _ = gex.step(mu)
    # print(f"After reset reward: {r3:.4f}")

    assert r3 > r2
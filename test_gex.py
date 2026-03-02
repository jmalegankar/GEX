import torch
from geodesic_bonus import GeodesicExplorationBonus


def test_first_visit_high():
    gex = GeodesicExplorationBonus(mu_dim=16)
    mu = torch.randn(16)
    mu = mu / mu.norm()

    r, _ = gex.step(mu)
    assert r > 0


def test_revisit_same_transition_decreases():
    gex = GeodesicExplorationBonus(mu_dim=8, k=1)

    mu = torch.randn(8)
    mu = mu / mu.norm()

    r1, _ = gex.step(mu)
    r2, _ = gex.step(mu)

    assert r2 < r1


def test_lifetime_decay():
    gex = GeodesicExplorationBonus(mu_dim=8)

    mu = torch.randn(8)
    mu = mu / mu.norm()

    r1, _ = gex.step(mu)
    r2, _ = gex.step(mu)
    r3, _ = gex.step(mu)

    assert r3 <= r2 <= r1

def test_lifetime_decay_only():
    from geodesic_bonus import AngularPseudoCounts

    pc = AngularPseudoCounts(mu_dim=8)
    mu = torch.randn(8)
    mu = mu / mu.norm()

    b1 = pc.bonus(mu)
    pc.increment(mu)

    b2 = pc.bonus(mu)
    pc.increment(mu)

    b3 = pc.bonus(mu)

    assert b3 < b2 < b1
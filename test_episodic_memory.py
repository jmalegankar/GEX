import torch
from episodic_memory import EpisodicMemory


def test_empty_memory_returns_max_bonus():
    mem = EpisodicMemory(mu_dim=16, k=3, max_size=10)
    mu = torch.randn(16)
    mu = mu / mu.norm()

    bonus = mem.query_and_add(mu)
    assert bonus == 2.0


def test_revisit_reduces_bonus():
    mem = EpisodicMemory(mu_dim=8, k=1, max_size=10)

    mu = torch.randn(8)
    mu = mu / mu.norm()

    b1 = mem.query_and_add(mu)
    b2 = mem.query_and_add(mu)

    assert b2 < b1


def test_capacity_caps_size():
    mem = EpisodicMemory(mu_dim=4, k=1, max_size=5)

    for _ in range(10):
        mu = torch.randn(4)
        mu = mu / mu.norm()
        mem.query_and_add(mu)

    assert mem.size == 5
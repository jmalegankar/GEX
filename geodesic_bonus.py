"""
Geodesic Exploration Bonus

Combines:
  - Episodic kNN novelty (cosine distance on S^{d-1})
  - Lifetime pseudo-counts via SimHash
"""

from __future__ import annotations

import math
import torch
import numpy as np

from episodic_memory import EpisodicMemory


# ============================================================
# Angular Pseudo-Counts (SimHash)
# ============================================================

class AngularPseudoCounts:
    def __init__(
        self,
        mu_dim: int,
        hash_bits: int = 32,
        seed: int = 0,
    ):
        self.mu_dim    = mu_dim
        self.hash_bits = hash_bits

        rng    = np.random.RandomState(seed)
        planes = rng.randn(mu_dim, hash_bits).astype(np.float32)
        planes /= np.linalg.norm(planes, axis=0, keepdims=True) + 1e-8

        self.planes   = torch.from_numpy(planes)   # CPU
        self._counts  = {}

    def _hash(self, mu: torch.Tensor) -> bytes:
        mu_cpu = mu.detach().cpu()
        proj   = mu_cpu @ self.planes   # (hash_bits,)
        bits   = (proj > 0).byte()
        return bits.numpy().tobytes()

    def get_count(self, mu: torch.Tensor) -> int:
        return self._counts.get(self._hash(mu), 0)

    def increment(self, mu: torch.Tensor):
        key              = self._hash(mu)
        self._counts[key] = self._counts.get(key, 0) + 1

    def bonus(self, mu: torch.Tensor) -> float:
        return 1.0 / math.sqrt(self.get_count(mu) + 1)

    @property
    def n_buckets(self) -> int:
        """Number of distinct SimHash buckets seen so far."""
        return len(self._counts)


# ============================================================
# Combined GEX Module
# ============================================================

class GeodesicExplorationBonus:
    """
    Pure intrinsic reward computation.  Stateless except for:
      - episodic ring buffer  (resets every episode)
      - lifetime hash counts  (never resets during training)
    """

    def __init__(
        self,
        mu_dim: int,
        k: int = 5,
        episodic_capacity: int = 2000,
        hash_bits: int = 32,
        device: str = "cpu",
    ):
        self.device = torch.device(device)

        self.episodic = EpisodicMemory(
            mu_dim=mu_dim,
            k=k,
            max_size=episodic_capacity,
            device=device,
        )

        self.lifetime = AngularPseudoCounts(
            mu_dim=mu_dim,
            hash_bits=hash_bits,
        )

    def reset(self):
        """Reset episodic memory only (call at episode boundary)."""
        self.episodic.reset()

    def step(self, mu: torch.Tensor):
        """
        Args:
            mu: (d,) unit vector on S^{d-1}
        Returns:
            r_int (float), info (dict)
        """
        if mu.dim() == 2:
            mu = mu.squeeze(0)

        # Episodic bonus: mean cosine distance to k nearest neighbours.
        r_epi = self.episodic.query_and_add(mu)

        # Lifetime bonus BEFORE incrementing so the first visit gets 1/sqrt(1).
        r_life = self.lifetime.bonus(mu)
        self.lifetime.increment(mu)

        r_int = r_epi * r_life

        info = {
            "r_episodic":        r_epi,
            "r_lifetime":        r_life,
            "r_int":             r_int,
            "episodic_size":     self.episodic.size,
            # FIX (Bug 4): expose the key that ppo_gex.py logs.
            "lifetime_n_buckets": self.lifetime.n_buckets,
        }

        return r_int, info
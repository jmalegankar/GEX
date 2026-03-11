from __future__ import annotations
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn



@torch.jit.interface
class EpisodicMemoryInterface:
    def query_and_add(self, mu: torch.Tensor) -> torch.Tensor:
        """
        Compute per-env novelty bonus and store the encoding.

        Args:
            mu: (n_envs, d) encoding vectors.
        Returns:
            (n_envs,) float32 bonus — 1.0 if novel, 0.0 if seen this episode.
        """
        pass

    def reset_envs(self, dones: torch.Tensor) -> None:
        """
        Clear memory for environments that finished an episode.

        Args:
            dones: (n_envs,) bool tensor, True for finished envs.
        """
        pass

    def reset_all(self) -> None:
        """Clear all memory across all envs."""
        pass


# ── Batched implementation ────────────────────────────────────────────────────

class BatchedNoveltyMemory(nn.Module):
    """
    Per-episode SimHash novelty for vectorized environments.

    Multiplicative usage with Wyner KL:
        intrinsic = wyner_kl * episodic_bonus
    suppresses revisited states within an episode even if the Wyner model
    finds them surprising.

    """

    # Class-level annotations required by TorchScript
    planes: torch.Tensor
    n_envs: int
    stores: List[Dict[int, bool]]

    def __init__(
        self,
        input_dim: int,
        n_envs: int = 1,
        hash_dim: int = 63,
        seed: int = 0,
    ):
        super().__init__()
        assert hash_dim <= 63, (
            f"hash_dim must be ≤ 63 for safe int64 bit-packing; got {hash_dim}"
        )
        self.n_envs = n_envs

        rng = np.random.RandomState(seed)
        planes = rng.randn(input_dim, hash_dim).astype(np.float32)
        norms = np.linalg.norm(planes, axis=0, keepdims=True)
        planes /= norms + 1e-8
        # register_buffer makes planes move with .to(device) calls
        self.register_buffer("planes", torch.from_numpy(planes))

        stores: List[Dict[int, bool]] = []
        for _ in range(n_envs):
            stores.append(torch.jit.annotate(Dict[int, bool], {}))
        self.stores = stores

    def _pack_bits(self, bits: torch.Tensor) -> int:
        """Pack a 1-D binary uint8 tensor into a single JIT int (bit-shift)."""
        h: int = 0
        for i in range(bits.shape[0]):
            h = h * 2 + int(bits[i].item())
        return h

    def query_and_add(self, mu: torch.Tensor) -> torch.Tensor:
        """Check novelty and record hash for each env."""
        proj = mu.detach().to(self.planes.device) @ self.planes  # (n_envs, hash_dim)
        binary = (proj > 0).to(torch.uint8)                       # (n_envs, hash_dim)
        bonuses = torch.zeros(self.n_envs, dtype=torch.float32)
        for i in range(self.n_envs):
            key = self._pack_bits(binary[i])
            if key not in self.stores[i]:
                bonuses[i] = 1.0
            self.stores[i][key] = True
        return bonuses

    def reset_envs(self, dones: torch.Tensor) -> None:
        """Reset hash stores for envs where dones[i] is True."""
        for i in range(self.n_envs):
            if bool(dones[i].item()):
                self.stores[i].clear()

    def reset_all(self) -> None:
        """Clear all env hash stores."""
        for i in range(self.n_envs):
            self.stores[i].clear()

    @property
    def bucket_counts(self) -> List[int]:
        """Number of distinct hash buckets seen per env (diagnostic)."""
        return [len(s) for s in self.stores]

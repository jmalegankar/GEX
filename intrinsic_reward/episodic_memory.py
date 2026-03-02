"""
Spherical Episodic Memory (Cosine k-NN)

Purpose:
  EpisodicMemory gives within-episode novelty.
  It answers: "Have I seen a transition like this earlier in this episode?"
  Not lifetime novelty. Not global coverage.
  Just: is this transition new right now?

Why we need it:
  Without episodic memory the agent may revisit the same state repeatedly
  within an episode, intrinsic reward stays high for loops, and exploration
  becomes cyclic farming.  Episodic memory prevents that:
    - First visit  → high bonus
    - Revisit soon → low bonus
    - Encourages forward progress
  This is the NGU / RIDE idea, but we use sphere geometry instead of
  Euclidean distance.

What it stores:
  Transition encodings μ_t ∈ S^{d-1} from the current episode.
  Not raw states. Not observations. Only μ vectors.
  Memory resets every episode.

How it computes novelty:
  Given current μ and memory matrix M (N × d):
    sim  = μ · M^T          (cosine similarity)
    dist = 1 - sim           (cosine distance, lies in [0, 2])
  Then take k nearest neighbours (largest sim / smallest dist).
  Surprise = mean(dist_k).

Why cosine instead of acos:
  acos gives the true geodesic distance, but it is expensive and
  unnecessary — ranking is preserved by cosine since it is a monotonic
  transformation.  We only need monotonic ordering, so cosine distance
  is correct and faster.

Design:
  - Fixed-capacity ring buffer (O(N) per step, not O(N²))
  - Fully GPU-resident — no CPU bouncing, no python-list stacking
  - Distance = 1 - cosine, bounded in [0, 2]

Edge cases:
  - Memory empty or size < k → return max bonus (2.0)
"""

from __future__ import annotations

import torch


class EpisodicMemory:
    """Within-episode novelty via cosine k-NN on μ ∈ S^{d-1}."""

    def __init__(
        self,
        mu_dim: int,
        k: int = 5,
        max_size: int = 2000,
        device: str = "cpu",
    ):
        self.mu_dim = mu_dim
        self.k = k
        self.max_size = max_size
        self.device = torch.device(device)

        # Ring buffer storing μ vectors seen this episode.
        self._memory = torch.empty(
            (max_size, mu_dim),
            device=self.device,
        )

        self._size = 0
        self._ptr = 0  # ring pointer

    # ============================================================
    # Core API
    # ============================================================

    def reset(self):
        """Clear memory at episode start."""
        self._size = 0
        self._ptr = 0

    def query_and_add(self, mu: torch.Tensor) -> float:
        """
        Compute episodic novelty bonus and store μ.

        Steps:
          1. If memory is empty → return max bonus (2.0).
          2. Otherwise compute sim = μ · M^T  (cosine similarity).
          3. Take k nearest neighbours (highest sim → smallest dist).
          4. bonus = mean(1 - topk_sim).
          5. Write μ into the ring buffer.

        Args:
            mu: (d,) unit vector on S^{d-1}.
        Returns:
            scalar bonus in [0, 2].
        """
        if mu.dim() != 1:
            raise ValueError("mu must be 1D (d,)")

        mu = mu.to(self.device)

        # Edge case: nothing in memory yet → maximally novel.
        if self._size == 0:
            bonus = 2.0
        else:
            mem = self._memory[: self._size]  # (N, d)

            # Cosine similarity: μ already unit-norm, mem rows unit-norm.
            sim = torch.matmul(mem, mu)  # (N,)

            # k nearest = highest similarity (smallest cosine distance).
            # If fewer than k items stored, use what we have.
            k = min(self.k, self._size)
            topk_sim, _ = torch.topk(sim, k=k, largest=True)

            # Cosine distance ∈ [0, 2].  Mean over k neighbours.
            dist = 1.0 - topk_sim  # (k,)
            bonus = dist.mean().item()

        # Add μ to ring buffer (overwrites oldest entry when full).
        self._memory[self._ptr] = mu.detach()
        self._ptr = (self._ptr + 1) % self.max_size

        if self._size < self.max_size:
            self._size += 1

        return bonus

    # ============================================================
    # Properties
    # ============================================================

    @property
    def size(self) -> int:
        return self._size

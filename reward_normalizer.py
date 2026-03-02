"""
Running Mean / Variance (Welford)

Used to normalize intrinsic rewards before adding to PPO reward.
"""

from __future__ import annotations
import torch


class RunningMeanStd:
    def __init__(self, epsilon: float = 1e-8, device: str = "cpu"):
        self.device = torch.device(device)
        self.mean = torch.zeros(1, device=self.device)
        self.var = torch.ones(1, device=self.device)
        self.count = torch.tensor(epsilon, device=self.device)

    def update(self, x: torch.Tensor):
        """
        x: (B,) tensor
        """
        x = x.to(self.device)

        batch_mean = x.mean()
        batch_var = x.var(unbiased=False)
        batch_count = torch.tensor(x.numel(), device=self.device, dtype=torch.float32)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total_count

        new_var = m2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x: torch.Tensor):
        return x / torch.sqrt(self.var + 1e-8)
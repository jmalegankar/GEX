import torch as th
import torch.nn as nn


class SlotMemory(nn.Module):
    """
    Fixed-size slot memory with surprise-gated LRU writes.

    Write: soft-blend h_t into the oldest slot, gated by Wyner KL.
    Read:  handled by HSWVIMEFeaturesExtractor (cross-attention over slots).
    """

    def __init__(
        self,
        num_slots: int,
        slot_dim: int,
        gate_mode: str = "detached",
        gate_scale: float = 1.0,
        gate_threshold: float = 0.0,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.gate_mode = gate_mode
        self.gate_scale = gate_scale
        self.gate_threshold = gate_threshold

        if gate_mode == "learned":
            self.gate_net = nn.Sequential(
                nn.Linear(slot_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )
            self._last_learned_gate = None
            self._last_kl = None

    def write(
        self,
        slots: th.Tensor,
        ages: th.Tensor,
        h_t: th.Tensor,
        kl: th.Tensor,
    ):
        """
        Soft-gated LRU write.

        Args:
            slots: (B, K, D) current slot contents
            ages:  (B, K) steps since each slot was last written
            h_t:   (B, D) new content to write (Wyner deterministic state)
            kl:    (B,) Wyner KL surprise signal

        Returns: (new_slots, new_ages, gate_values)
        """
        B, K, D = slots.shape

        if self.gate_mode == "detached":
            gate = th.sigmoid(self.gate_scale * (kl - self.gate_threshold))  # (B,)
        else:
            gate = th.sigmoid(self.gate_net(h_t).squeeze(-1))  # (B,)
            self._last_learned_gate = gate
            self._last_kl = kl.detach()

        # Oldest slot per batch element
        lru_idx = ages.argmax(dim=1)  # (B,)
        idx = lru_idx.view(B, 1, 1).expand(B, 1, D)

        old = slots.gather(1, idx).squeeze(1)  # (B, D)
        g = gate.unsqueeze(1)  # (B, 1)
        new_content = g * h_t + (1 - g) * old  # (B, D)

        new_slots = slots.clone()
        new_slots.scatter_(1, idx, new_content.unsqueeze(1))

        new_ages = ages + 1
        new_ages.scatter_(1, lru_idx.unsqueeze(1), th.zeros(B, 1, device=ages.device, dtype=ages.dtype))

        return new_slots, new_ages, gate

    def gate_correlation_loss(self) -> th.Tensor:
        """MSE between learned gate and sigmoid(scale * kl). Only for gate_mode='learned'."""
        if self._last_learned_gate is None or self._last_kl is None:
            return th.tensor(0.0)
        target = th.sigmoid(self.gate_scale * self._last_kl)
        return nn.functional.mse_loss(self._last_learned_gate, target)

    def init_state(self, batch_size: int, device: th.device):
        """Zero slots, staggered ages so first K writes fill distinct slots."""
        slots = th.zeros(batch_size, self.num_slots, self.slot_dim, device=device)
        ages = th.arange(self.num_slots, device=device).float().unsqueeze(0).expand(batch_size, -1).clone()
        return slots, ages

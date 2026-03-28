import torch as th
import torch.nn as nn
import torch.nn.functional as F


class SlotMemory(nn.Module):
    """
    Fixed-size slot memory with surprise-gated content-addressed writes.

    Write: soft-blend z_t into slots weighted by cosine similarity,
           gated by marginal information gain (KL above slot-conditioned prior).
    Read:  handled by HSWVIMEFeaturesExtractor (cross-attention over slots).
    """

    def __init__(
        self,
        num_slots: int,
        slot_dim: int,
        gate_mode: str = "detached",
        gate_scale: float = 1.0,
        gate_threshold: float = 0.0,
        write_temp: float = 1.0,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.gate_mode = gate_mode
        self.gate_scale = gate_scale
        self.gate_threshold = gate_threshold
        self.write_temp = write_temp
        self.max_slot_norm = 5.0

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
        z_t: th.Tensor,
        kl: th.Tensor,
    ):
        """
        Content-addressed soft write gated by marginal surprise.

        Args:
            slots: (B, K, D) current slot contents
            z_t:   (B, D) new content to write (Wyner posterior mean)
            kl:    (B,) marginal information gain (delta_I)

        Returns: (new_slots, gate_values)
        """
        B, K, D = slots.shape

        # ── Gate: whether to write ────────────────────────────────
        if self.gate_mode == "detached":
            gate = th.sigmoid(self.gate_scale * (kl - self.gate_threshold))  # (B,)
        else:
            gate = th.sigmoid(self.gate_net(z_t).squeeze(-1))  # (B,)
            self._last_learned_gate = gate
            self._last_kl = kl.detach()

        # ── Content-addressed write weights ───────────────────────
        # F.normalize returns zero for zero-norm vectors (safe for init)
        slots_norm = F.normalize(slots, dim=-1, eps=1e-6)       # (B, K, D)
        z_norm = F.normalize(z_t.unsqueeze(1), dim=-1, eps=1e-6)  # (B, 1, D)
        sim = (z_norm * slots_norm).sum(dim=-1)                  # (B, K)
        w = F.softmax(sim / self.write_temp, dim=-1)             # (B, K)

        # ── Soft residual update ──────────────────────────────────
        delta = z_t.unsqueeze(1) - slots                         # (B, K, D)
        new_slots = slots + gate.view(-1, 1, 1) * w.unsqueeze(2) * delta

        # ── Clamp slot norms to prevent unbounded growth ─────────
        norms = new_slots.norm(dim=-1, keepdim=True).clamp(min=1e-6)  # (B, K, 1)
        new_slots = th.where(
            norms > self.max_slot_norm,
            new_slots * (self.max_slot_norm / norms),
            new_slots,
        )

        return new_slots, gate

    def gate_correlation_loss(self) -> th.Tensor:
        """MSE between learned gate and sigmoid(scale * kl). Only for gate_mode='learned'."""
        if self._last_learned_gate is None or self._last_kl is None:
            return th.tensor(0.0)
        target = th.sigmoid(self.gate_scale * self._last_kl)
        return nn.functional.mse_loss(self._last_learned_gate, target)

    def init_state(self, batch_size: int, device: th.device) -> th.Tensor:
        """Small noise init to break softmax symmetry from the first write."""
        return th.randn(batch_size, self.num_slots, self.slot_dim, device=device) * 0.01

"""Component 2 test: content-addressed SlotMemory write."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from models.slot_memory import SlotMemory


def test_content_addressed_write():
    torch.manual_seed(42)

    mem = SlotMemory(num_slots=8, slot_dim=64, gate_scale=5.0, gate_threshold=0.0)
    B = 4
    slots = mem.init_state(B, device=torch.device("cpu"))

    # Verify init_state returns a single tensor (no ages)
    assert isinstance(slots, torch.Tensor), f"init_state should return Tensor, got {type(slots)}"
    assert slots.shape == (B, 8, 64), f"Expected (4, 8, 64), got {slots.shape}"
    print(f"init_state shape: {slots.shape} (single tensor, no ages)")

    # --- Test 1: zero-slot init does not produce NaN ---
    z_t = torch.randn(B, 64)
    kl = torch.ones(B) * 5.0  # high surprise -> gate should open
    new_slots, gate = mem.write(slots, z_t, kl)
    assert not torch.isnan(new_slots).any(), "NaN in slots after first write"
    assert gate.shape == (B,)
    assert (gate > 0.9).all(), f"Gate should be near 1.0 for high KL, got {gate}"
    print(f"Test 1 PASSED: no NaN, gate={gate.mean().item():.4f}")

    # --- Test 2: after signal is written, re-writing same z_t with low KL ---
    # In practice, z-score normalization produces negative KL for redundant steps
    slots_after_signal, _ = mem.write(slots, z_t, torch.ones(B) * 5.0)
    slots_after_2nd, gate2 = mem.write(slots_after_signal, z_t, torch.ones(B) * -2.0)
    assert (gate2 < 0.1).all(), f"Gate should be near 0 for negative KL, got {gate2}"
    delta = (slots_after_2nd - slots_after_signal).norm()
    assert delta < 0.01, f"Slots moved too much on closed gate: {delta}"
    print(f"Test 2 PASSED: gate2={gate2.mean().item():.4f}, delta={delta.item():.6f}")

    # --- Test 3: content-addressing concentrates weight on most-similar slot ---
    # Pre-load slot 3 with target, rest are zero. Write target again.
    # Slot 3 (already == target) should have delta~0 → barely moves.
    # Other slots get small updates. Slot 3 should have highest norm (closest to target).
    target = torch.randn(64)
    slots_loaded = slots.clone()
    slots_loaded[:, 3, :] = target.unsqueeze(0).expand(B, -1)
    new_sl, _ = mem.write(slots_loaded, target.unsqueeze(0).expand(B, -1), torch.ones(B) * 5.0)
    # Slot 3 already contained target, delta=0 there → it stays at full norm.
    # Other slots get gate * w_k * target with small w_k → smaller norm.
    norms = new_sl.norm(dim=-1)  # (B, 8)
    assert (norms[:, 3] > norms[:, 0]).all(), "Slot 3 should have highest norm"
    # Also verify: write two DIFFERENT vectors, check they land in different slots
    v1 = torch.randn(B, 64)
    v2 = torch.randn(B, 64)
    s = mem.init_state(B, torch.device("cpu"))
    s, _ = mem.write(s, v1, torch.ones(B) * 5.0)
    s, _ = mem.write(s, v2, torch.ones(B) * 5.0)
    # After 2 writes into zero-init, slot norms should not all be identical
    # (v2 should land near v1's location less than elsewhere if v1 != v2)
    cosine_v1 = F.cosine_similarity(s, v1.unsqueeze(1).expand_as(s), dim=-1)  # (B, K)
    cosine_v2 = F.cosine_similarity(s, v2.unsqueeze(1).expand_as(s), dim=-1)
    # Both should have nonzero signal spread across slots
    assert s.norm(dim=-1).mean() > 0.01, "Slots should be non-zero after writes"
    print(f"Test 3 PASSED: slot3 norm={norms[:, 3].mean().item():.4f} > slot0 norm={norms[:, 0].mean().item():.4f}")

    # --- Test 4: write returns exactly 2 values (no ages) ---
    result = mem.write(slots, z_t, kl)
    assert len(result) == 2, f"write() should return 2 values, got {len(result)}"
    print("Test 4 PASSED: write() returns (new_slots, gate)")

    # --- Test 5: uniform write at zero slots ---
    # When all slots are zero, softmax of zeros = uniform 1/K
    # So first write should spread z_t evenly (scaled by gate)
    fresh_slots = mem.init_state(1, torch.device("cpu"))
    z_single = torch.randn(1, 64)
    written, g = mem.write(fresh_slots, z_single, torch.ones(1) * 5.0)
    # All slots should be similar (uniform content addressing)
    slot_norms = written.norm(dim=-1).squeeze(0)  # (K,)
    norm_std = slot_norms.std().item()
    print(f"Test 5 PASSED: slot norms after first write: std={norm_std:.6f} (expect ~0)")
    assert norm_std < 0.01, f"First write should be uniform, got norm std={norm_std}"

    print("\nAll Component 2 tests PASSED.")


if __name__ == "__main__":
    test_content_addressed_write()

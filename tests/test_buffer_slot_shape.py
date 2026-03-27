"""Component 3 test: buffer handles slot_memory_shape=(8, 64)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from gymnasium import spaces
from hswvime_ppo.buffer import TransitionRolloutBuffer


def test_buffer_slot_shape():
    buf = TransitionRolloutBuffer(
        buffer_size=512, n_envs=4,
        observation_space=spaces.Box(low=0, high=1, shape=(5, 5, 4)),
        action_space=spaces.Discrete(7),
        slot_memory_shape=(8, 64),
    )
    buf.reset()
    assert buf.slot_memories.shape == (512, 4, 8, 64), f"Got {buf.slot_memories.shape}"
    print(f"slot_memories shape: {buf.slot_memories.shape} PASSED")

    # Add one step and verify no crash
    dummy_slots = torch.zeros(4, 8, 64)
    buf.add(
        obs=np.zeros((4, 5, 5, 4)),
        prev_obs=np.zeros((4, 5, 5, 4)),
        next_obs=np.zeros((4, 5, 5, 4)),
        action=np.zeros((4, 1)),
        reward=np.zeros(4),
        episode_start=np.zeros(4),
        value=torch.zeros(4),
        log_prob=torch.zeros(4),
        memory=torch.zeros(4, 1, 64),
        wyner_h=torch.zeros(4, 1, 64),
        slot_memory=dummy_slots,
        prev_action=np.zeros((4, 1)),
        timestep=np.zeros(4),
    )
    print("add() with slot_memory (4, 8, 64) PASSED")
    print("\nAll Component 3 tests PASSED.")


if __name__ == "__main__":
    test_buffer_slot_shape()

"""
play_model.py — visualise a trained LMU-PPO model.

Usage:
    python play_model.py --model_path lmu_ppo_MemoryS13_s0.zip --env MemoryS13 --episodes 5 --render
"""

import argparse
import io
import json
import zipfile

import numpy as np
import torch
import gymnasium as gym
import minigrid  # noqa: F401
from gymnasium.wrappers import FilterObservation
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from lmu_ppo.lmu_ppo import LMUPPO


ENV_IDS = {
    "MemoryS7":  "MiniGrid-MemoryS7-v0",
    "MemoryS9":  "MiniGrid-MemoryS9-v0",
    "MemoryS11": "MiniGrid-MemoryS11-v0",
    "MemoryS13": "MiniGrid-MemoryS13-v0",
}


def make_env(env_id: str, render: bool):
    def _init():
        env = gym.make(env_id, render_mode="human" if render else None)
        env = FilterObservation(env, filter_keys=["image", "direction"])
        env = Monitor(env)
        return env
    return _init


def load_lmuppo(model_path: str, env) -> LMUPPO:
    """
    Reconstruct LMUPPO from a .zip saved by SB3's model.save().

    SB3 (recent versions) stores:
      - 'data'       : JSON-encoded hyperparameters
      - 'policy.pth' : {"policy": state_dict, "policy.optimizer": ...}

    We only need the arch hyperparams from 'data' and the policy weights.
    Everything else (predictor, optimizer) is re-initialised from scratch
    since we're only running inference.
    """
    with zipfile.ZipFile(model_path) as zf:
        # ---- hyperparameters (JSON in recent SB3) ----------------------
        with zf.open("data") as f:
            raw = f.read()
        # SB3 wraps the JSON in a cloudpickle header in some versions;
        # try JSON first, fall back to cloudpickle if needed.
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            import cloudpickle
            data = cloudpickle.loads(raw)

        # ---- policy weights --------------------------------------------
        with zf.open("policy.pth") as f:
            policy_params = torch.load(
                io.BytesIO(f.read()), map_location="cpu", weights_only=False
            )

    # Pull arch params — fall back to the defaults we used in train.py
    # so old checkpoints without these keys still load cleanly.
    encoder_dim      = data.get("encoder_dim",      64)
    hidden_size      = data.get("hidden_size",       64)
    memory_size      = data.get("memory_size",       32)
    theta            = data.get("theta",            150.0)
    intrinsic_scale  = data.get("intrinsic_scale",   0.0)   # inference: no intrinsic
    predictor_hidden = data.get("predictor_hidden",  64)
    predictor_lr     = data.get("predictor_lr",      1e-3)

    model = LMUPPO(
        env              = env,
        encoder_dim      = encoder_dim,
        hidden_size      = hidden_size,
        memory_size      = memory_size,
        theta            = theta,
        intrinsic_scale  = intrinsic_scale,
        predictor_hidden = predictor_hidden,
        predictor_lr     = predictor_lr,
    )

    # policy.pth keys: "policy", "policy.optimizer"
    # We only need "policy" (the LMUActorCriticPolicy state dict).
    state_dict = policy_params.get("policy", policy_params)
    model.policy.load_state_dict(state_dict)
    model.policy.set_training_mode(False)

    print(f"  encoder_dim={encoder_dim}  hidden={hidden_size}  "
          f"memory={memory_size}  theta={theta}")
    return model


def run_episode(env, model, deterministic: bool = True) -> float:
    obs   = env.reset()
    done  = np.array([False])
    state = None
    total_reward = 0.0
    step = 0

    while not done[0]:
        action, state = model.predict(
            obs, state=state, episode_start=done, deterministic=deterministic
        )
        obs, reward, done, _ = env.step(action)
        total_reward += float(reward[0])
        step += 1

    return total_reward, step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",    type=str, required=True)
    parser.add_argument("--env",           type=str, required=True, choices=list(ENV_IDS))
    parser.add_argument("--episodes",      type=int, default=5)
    parser.add_argument("--render",        action="store_true")
    parser.add_argument("--stochastic",    action="store_true",
                        help="Sample actions instead of argmax")
    args = parser.parse_args()

    env_id = ENV_IDS[args.env]
    env    = VecTransposeImage(DummyVecEnv([make_env(env_id, args.render)]))

    print(f"Loading model from {args.model_path}")
    model = load_lmuppo(args.model_path, env)

    rewards, lengths = [], []
    for ep in range(args.episodes):
        r, l = run_episode(env, model, deterministic=not args.stochastic)
        rewards.append(r)
        lengths.append(l)
        print(f"  ep {ep+1:3d}  reward={r:.2f}  length={l}")

    print(f"\n=== {args.episodes} episodes ===")
    print(f"  mean reward : {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
    print(f"  mean length : {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")

    env.close()


if __name__ == "__main__":
    main()
"""
Stage 1: LMU-PPO baseline on MiniGrid-MemoryS5/S7.

Run:
    python train.py --env MemoryS5 --seed 0
    python train.py --env MemoryS7 --seed 0
"""

import argparse
import gymnasium as gym
import minigrid
from gymnasium.wrappers import FilterObservation
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from lmu_ppo.lmu_ppo import LMUPPO


ENV_IDS = {
    "MemoryS7":  "MiniGrid-MemoryS7-v0",
    "MemoryS9":  "MiniGrid-MemoryS9-v0",
    "MemoryS11": "MiniGrid-MemoryS11-v0",
    "MemoryS13": "MiniGrid-MemoryS13-v0",
}

# Approximate episode length for each env → used as theta
THETA = {
    "MemoryS7":  50.0,
    "MemoryS9":  75.0,
    "MemoryS11": 100.0,
    "MemoryS13": 150.0,
}


def make_env(env_id: str, seed: int, rank: int = 0):
    def _init():
        env = gym.make(env_id)
        env = FilterObservation(env, filter_keys=["image", "direction"])
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",        default="MemoryS7", choices=list(ENV_IDS))
    parser.add_argument("--seed",       type=int, default=0)
    parser.add_argument("--n_envs",     type=int, default=8)
    parser.add_argument("--total_steps",type=int, default=2_000_000)
    parser.add_argument("--n_steps",    type=int, default=512)   # steps per env per rollout
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--tb_log",     default="runs/lmu_ppo")
    args = parser.parse_args()

    env_id = ENV_IDS[args.env]

    train_env = VecTransposeImage(SubprocVecEnv([
        make_env(env_id, args.seed, i) for i in range(args.n_envs)
    ]))
    eval_env = VecTransposeImage(DummyVecEnv([make_env(env_id, args.seed + 1000)]))

    eval_cb = EvalCallback(
        eval_env,
        eval_freq=max(10_000 // args.n_envs, 1),
        n_eval_episodes=20,
        verbose=1,
    )

    model = LMUPPO(
        env=train_env,
        # Credit assignment — the whole point of Stage 1
        gamma=0.999,
        gae_lambda=0.95,
        # Architecture
        encoder_dim=64,
        hidden_size=64,
        memory_size=32,
        theta=THETA[args.env],
        # PPO
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=4,
        lr=3e-4,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        clip_range=0.2,
        # Infra
        tensorboard_log=args.tb_log,
        verbose=1,
        seed=args.seed,
        device="auto",
    )

    print(f"\nTraining LMU-PPO on {env_id}  |  seed={args.seed}")
    print(f"  hidden={model.hidden_size}  memory={model.memory_size}  theta={model.theta}")
    print(f"  gamma={model.gamma}  n_envs={args.n_envs}  total_steps={args.total_steps:,}\n")

    model.learn(
        total_timesteps=args.total_steps,
        callback=eval_cb,
        tb_log_name=f"lmu_{args.env}_s{args.seed}",
        progress_bar=True,
    )

    model.save(f"lmu_ppo_{args.env}_s{args.seed}")
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
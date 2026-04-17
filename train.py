"""
LMU-PPO baseline on MiniGrid-Memory envs.

Run:
    # Chunked TBPTT (default):
    python train.py --env MemoryS7  --seed 0
    python train.py --env MemoryS13 --seed 0

    # Full episode BPTT on S7:
    python train.py --env MemoryS7 --seed 0 --full_bptt
"""

import argparse
import gymnasium as gym
import minigrid  # noqa: F401
from gymnasium.wrappers import FilterObservation
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from lmu_ppo.lmu_ppo import LMUPPO
from mem_start import MemoryStartWrapper


ENV_IDS = {
    "MemoryS5":  "MiniGrid-MemoryS5-v0",
    "MemoryS7":  "MiniGrid-MemoryS7-v0",
    "MemoryS9":  "MiniGrid-MemoryS9-v0",
    "MemoryS11": "MiniGrid-MemoryS11-v0",
    "MemoryS13": "MiniGrid-MemoryS13-v0",
}

# theta = memory window horizon (steps).
# Should cover the full episode so the ball seen at step 1 is
# still reconstructable at episode end.
THETA = {
    "MemoryS5":   64,
    "MemoryS7":  100,
    "MemoryS9":  120,
    "MemoryS11": 160,
    "MemoryS13": 200,
}

ARCH = {
    "MemoryS5":  dict(hidden_size=64,  memory_size=32),
    "MemoryS7":  dict(hidden_size=64,  memory_size=32),
    "MemoryS9":  dict(hidden_size=128, memory_size=48),
    "MemoryS11": dict(hidden_size=128, memory_size=64),
    "MemoryS13": dict(hidden_size=128, memory_size=96),
}

# chunk_len for TBPTT.
# For S7 full BPTT we use max_episode_len (~50 steps) so every episode
# gets a full gradient unroll.  For harder envs we use a shorter chunk
# to keep memory and compute reasonable.
CHUNK_LEN_DEFAULT = {
    "MemoryS5":  16,
    "MemoryS7":  16,
    "MemoryS9":  32,
    "MemoryS11": 32,
    "MemoryS13": 32,
}

# MiniGrid Memory max_steps ≈ 5*(size-2) for size=grid_size
# S7→size=7: 5*5=25... actually it's set per-env; 50 is a safe upper bound for S7.
FULL_BPTT_CHUNK = {
    "MemoryS5":  32,
    "MemoryS7":  64,   # full episode (episode len ≤ 50, round up to next divisor)
    "MemoryS9":  64,
    "MemoryS11": 64,
    "MemoryS13": 64,
}


def make_env(env_id: str, seed: int, rank: int = 0):
    def _init():
        env = gym.make(env_id)
        env = MemoryStartWrapper(env)
        env = FilterObservation(env, filter_keys=["image", "direction"])
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",         default="MemoryS11", choices=list(ENV_IDS))
    parser.add_argument("--seed",        type=int,   default=0)
    parser.add_argument("--n_envs",      type=int,   default=16)
    parser.add_argument("--total_steps", type=int,   default=2_000_000)
    parser.add_argument("--n_steps",     type=int,   default=512)
    parser.add_argument("--batch_size",  type=int,   default=256)
    parser.add_argument("--n_epochs",    type=int,   default=4)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--tb_log",      default="runs/lmu_ppo")
    parser.add_argument("--device",      default="auto")
    parser.add_argument("--n_chunks_per_batch", type=int, default=16,
                        help="Number of K-step chunks per PPO update batch (for TBPTT). "
                             "Total batch size = n_chunks_per_batch * chunk_len.")
    parser.add_argument("--full_bptt",   action="store_true",
                        help="Use full-episode BPTT (chunk_len = max episode length)")
    args = parser.parse_args()

    env_id = ENV_IDS[args.env]
    arch   = ARCH[args.env]
    theta  = THETA[args.env]

    chunk_len = (
        FULL_BPTT_CHUNK[args.env] if args.full_bptt
        else CHUNK_LEN_DEFAULT[args.env]
    )

    # n_steps must be divisible by chunk_len
    if args.n_steps % chunk_len != 0:
        old = args.n_steps
        args.n_steps = (args.n_steps // chunk_len) * chunk_len
        print(f"  [warn] n_steps adjusted {old} → {args.n_steps} "
              f"to be divisible by chunk_len={chunk_len}")

    train_env = VecTransposeImage(SubprocVecEnv([
        make_env(env_id, args.seed, i) for i in range(args.n_envs)
    ]))
    eval_env = VecTransposeImage(DummyVecEnv([
        make_env(env_id, args.seed + 1000)
    ]))

    eval_cb = EvalCallback(
        eval_env,
        eval_freq=max(10_000 // args.n_envs, 1),
        n_eval_episodes=20,
        verbose=1,
    )

    model = LMUPPO(
        env=train_env,
        encoder_dim=64,
        hidden_size=arch["hidden_size"],
        memory_size=arch["memory_size"],
        theta=theta,
        chunk_len=chunk_len,
        gamma=0.999,
        gae_lambda=0.98,
        n_steps=args.n_steps,
        n_chunks_per_batch=args.n_chunks_per_batch,
        n_epochs=args.n_epochs,
        lr=args.lr,
        ent_coef=0.008,
        vf_coef=1.0,
        max_grad_norm=0.5,
        clip_range=0.2,
        tensorboard_log=args.tb_log,
        verbose=1,
        seed=args.seed,
        device=args.device,
    )

    # Total chunks available per rollout = (n_steps // chunk_len) * n_envs
    total_chunks = (args.n_steps // chunk_len) * args.n_envs
    bptt_mode = f"full_bptt (K={chunk_len})" if args.full_bptt else f"tbptt (K={chunk_len})"
    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"\nLMU-PPO  ·  {env_id}  ·  seed={args.seed}  ·  {bptt_mode}")
    print(f"  encoder=64  hidden={arch['hidden_size']}"
          f"  memory={arch['memory_size']}  theta={theta}")
    print(f"  gamma={model.gamma}  n_envs={args.n_envs}"
          f"  total_steps={args.total_steps:,}")
    print(f"  total_chunks={total_chunks}  n_chunks_per_batch={args.n_chunks_per_batch}"
          f"  → {args.n_chunks_per_batch * chunk_len} transitions/update")
    print(f"  policy params: {total_params:,}\n")

    model.learn(
        total_timesteps=args.total_steps,
        callback=eval_cb,
        tb_log_name=f"lmu_{args.env}_{'full_bptt' if args.full_bptt else 'tbptt'}_s{args.seed}",
        progress_bar=True,
    )

    save_path = f"lmu_ppo_{args.env}_s{args.seed}"
    model.save(save_path)
    print(f"\nSaved → {save_path}")
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
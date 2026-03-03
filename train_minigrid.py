"""
GEX on MiniGrid benchmarks.

Recommended envs (in order of difficulty):
    MiniGrid-KeyCorridorS3R3-v0   — key + door + goal, procedural
    MiniGrid-MultiRoom-N6-v0      — 6 rooms, procedural
    MiniGrid-ObstructedMaze-2Dlh-v0 — harder version

Usage:
    python train_minigrid.py
    python train_minigrid.py --env MiniGrid-MultiRoom-N6-v0 --timesteps 3_000_000
    python train_minigrid.py --no-intrinsic   # PPO baseline
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from envs.minigrid_wrapper import MiniGridWrapper
from models.sc_vae import TransitionSCVAE
from models.obs_embeddings import CategoricalGridEmbedding, CategoricalGridSpec
from models.config import SCVAEConfig
from intrinsic_reward.geodesic_bonus import GeodesicExplorationBonus
from intrinsic_reward.reward_normalizer import RunningMeanStd
from sb3.sc_vae_state_extractor import SCVAEStateExtractor
from sb3.ppo_gex import PPOGEX


# MiniGrid standard catalogue — same as MultiGrid
GRID_SPEC = CategoricalGridSpec(
    n_object_types=11,
    n_colors=6,
    n_states=4,
    embed_per_channel=4,
)


def make_env(env_id: str):
    def _init():
        return MiniGridWrapper(env_id)
    return _init


def build_model(env, *, env_id: str, n_envs: int, eta: float, tau: float,
                sc_vae_freeze_steps: int | None, use_intrinsic: bool, device: str) -> PPOGEX:

    # Infer obs shape and n_actions from a temp env
    tmp = MiniGridWrapper(env_id)
    obs_shape = tmp.observation_space.shape   # (H, W, 3)
    n_actions = tmp.action_space.n
    tmp.close()

    # ---- SC-VAE -------------------------------------------------------
    embedding = CategoricalGridEmbedding(GRID_SPEC)
    cfg = SCVAEConfig(
        conv_channels=(32, 64),
        hidden_dim=256,
        latent_dim=32,
        n_actions=n_actions,
        action_embed_dim=8,
        kl_max_terms=128,
        lr=1e-4,
        beta=0.005,
        no_op_action=0,
    )
    sc_vae = TransitionSCVAE(
        embedding=embedding,
        cfg=cfg,
        sample_input_shape=obs_shape,
    )

    # ---- GEX modules --------------------------------------------------
    if use_intrinsic:
        gex_modules = [
            GeodesicExplorationBonus(
                mu_dim=cfg.latent_dim,
                k=5,
                episodic_capacity=2000,
                hash_bits=32,
                device=device,
            )
            for _ in range(n_envs)
        ]
        rms = RunningMeanStd(device=device)
    else:
        sc_vae      = None
        gex_modules = None
        rms         = None

    # ---- CNN policy via SC-VAE state encoder --------------------------
    # Policy reuses sc_vae's conv encoder (EMA target copy).
    # No separate policy CNN — one set of weights does both jobs.
    # _setup_model swaps the extractor to sc_vae_target after creation.
    model = PPOGEX(
        "CnnPolicy",
        env,
        sc_vae=sc_vae,
        gex_modules=gex_modules,
        rms=rms,
        eta=eta if use_intrinsic else 0.0,
        tau=tau,
        sc_vae_freeze_steps=sc_vae_freeze_steps if use_intrinsic else None,
        policy_kwargs=dict(
            features_extractor_class=SCVAEStateExtractor,
            features_extractor_kwargs=dict(sc_vae=sc_vae),
            net_arch=[256, 256],
        ),
        n_steps=2048,
        batch_size=256,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.05,
        learning_rate=3e-4,
        clip_range=0.2,
        verbose=1,
        device=device,
        tensorboard_log="./tb_logs/minigrid",
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",          type=str,   default="MiniGrid-KeyCorridorS3R3-v0")
    parser.add_argument("--n-envs",       type=int,   default=8)
    parser.add_argument("--timesteps",    type=int,   default=3_000_000)
    parser.add_argument("--eta",                type=float, default=0.05)
    parser.add_argument("--tau",                type=float, default=0.005)
    parser.add_argument("--sc-vae-freeze-steps", type=int,  default=300_000)
    parser.add_argument("--device",       type=str,   default="auto")
    parser.add_argument("--no-intrinsic", action="store_true")
    parser.add_argument("--seed",         type=int,   default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_env = DummyVecEnv([make_env(args.env) for _ in range(args.n_envs)])
    train_env = VecMonitor(train_env)

    eval_env = DummyVecEnv([make_env(args.env)])
    eval_env = VecMonitor(eval_env)

    model = build_model(
        train_env,
        env_id=args.env,
        n_envs=args.n_envs,
        eta=args.eta,
        tau=args.tau,
        sc_vae_freeze_steps=args.sc_vae_freeze_steps,
        use_intrinsic=not args.no_intrinsic,
        device=device,
    )

    env_slug = args.env.replace("/", "_")
    tag = "gex" if not args.no_intrinsic else "ppo_baseline"

    os.makedirs(f"checkpoints/{env_slug}", exist_ok=True)
    os.makedirs(f"eval_logs/{env_slug}",   exist_ok=True)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=f"checkpoints/{env_slug}/best",
        log_path=f"eval_logs/{env_slug}",
        eval_freq=max(20_000 // args.n_envs, 1),
        n_eval_episodes=20,
        deterministic=True,
        verbose=1,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=max(200_000 // args.n_envs, 1),
        save_path=f"checkpoints/{env_slug}",
        name_prefix=f"ppogex_{tag}",
    )

    print(f"\n{'='*50}")
    print(f"  {args.env} — {tag}")
    print(f"  η={args.eta}  τ={args.tau}  n_envs={args.n_envs}")
    print(f"  timesteps={args.timesteps:,}")
    print(f"{'='*50}\n")

    model.learn(
        total_timesteps=args.timesteps,
        callback=[eval_cb, ckpt_cb],
        tb_log_name=f"{env_slug}_{tag}",
        progress_bar=True,
    )

    from stable_baselines3.common.evaluation import evaluate_policy
    mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=50)
    print(f"\nFinal eval (50 eps): mean_r={mean_r:.3f} ± {std_r:.3f}")
    model.save(f"checkpoints/{env_slug}/final_{tag}")


if __name__ == "__main__":
    main()
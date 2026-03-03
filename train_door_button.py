"""
Phase B training: GEX on DoorButton.

Usage:
    python train_door_button.py
    python train_door_button.py --size 15 --timesteps 2_000_000
    python train_door_button.py --eta 0.05 --no-intrinsic   # PPO baseline
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from envs.door_button_wrapper import DoorButtonSB3Env
from models.sc_vae import TransitionSCVAE
from models.obs_embeddings import CategoricalGridEmbedding, CategoricalGridSpec
from models.config import SCVAEConfig
from intrinsic_reward.geodesic_bonus import GeodesicExplorationBonus
from intrinsic_reward.reward_normalizer import RunningMeanStd
from sb3.gex_features_extractor import GEXFeaturesExtractor
from sb3.ppo_gex import PPOGEX


# -----------------------------------------------------------------------
# MultiGrid object catalogue (standard 11-type, 6-color, 4-state)
# -----------------------------------------------------------------------
GRID_SPEC = CategoricalGridSpec(
    n_object_types=12,  # 11 standard MultiGrid types + Button (id=11)
    n_colors=6,
    n_states=4,
    embed_per_channel=4,   # → out_channels = 12
)


def make_env(size: int, view_size: int, max_steps: int):
    def _init():
        return DoorButtonSB3Env(size=size, view_size=view_size, max_steps=max_steps)
    return _init


def build_model(
    env,
    *,
    view_size: int,
    n_envs: int,
    eta: float,
    tau: float,
    use_intrinsic: bool,
    device: str,
) -> PPOGEX:

    # ---- Embedding + SC-VAE ----------------------------------------
    embedding = CategoricalGridEmbedding(GRID_SPEC)
    

    cfg = SCVAEConfig(
        conv_channels=(32, 64),
        hidden_dim=256,
        latent_dim=32,
        n_actions=7,           # MultiGrid default
        action_embed_dim=8,
        kl_max_terms=128,
        lr=1e-4,
        beta=0.005,
        no_op_action=0,
    )

    sc_vae = TransitionSCVAE(
        embedding=embedding,
        cfg=cfg,
        sample_input_shape=(view_size, view_size, 3),
    )

    # ---- GEX modules (one per env) ---------------------------------
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
        # Pure PPO baseline: pass sc_vae=None so ppo_gex falls back to
        # standard collect_rollouts.
        sc_vae      = None
        gex_modules = None
        rms         = None

    # ---- PPOGEX -------------------------------------------------------
    policy_embedding = CategoricalGridEmbedding(GRID_SPEC)
    extractor_kwargs = dict(
        embedding=policy_embedding,
        conv_channels=(32, 64),
        features_dim=256,
        sample_obs_shape=(view_size, view_size, 3),
    )

    model = PPOGEX(
        "CnnPolicy",
        env,
        policy_kwargs=dict(
            features_extractor_class=GEXFeaturesExtractor,
            features_extractor_kwargs=extractor_kwargs,
            net_arch=[256, 256],
        ),
        sc_vae=sc_vae,
        gex_modules=gex_modules,
        rms=rms,
        eta=eta if use_intrinsic else 0.0,
        tau=tau,
        # PPO hyperparams
        n_steps=2048,
        batch_size=256,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        learning_rate=3e-4,
        clip_range=0.2,
        verbose=1,
        device=device,
        tensorboard_log="./tb_logs/door_button",
    )

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size",        type=int,   default=20)
    parser.add_argument("--view-size",   type=int,   default=5)
    parser.add_argument("--max-steps",   type=int,   default=400)
    parser.add_argument("--n-envs",      type=int,   default=4)
    parser.add_argument("--timesteps",   type=int,   default=1_000_000)
    parser.add_argument("--eta",         type=float, default=0.1)
    parser.add_argument("--tau",         type=float, default=0.005)
    parser.add_argument("--device",      type=str,   default="auto")
    parser.add_argument("--no-intrinsic", action="store_true",
                        help="Run plain PPO baseline (no GEX)")
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ---- Training env ------------------------------------------------
    train_env = DummyVecEnv([
        make_env(args.size, args.view_size, args.max_steps)
        for _ in range(args.n_envs)
    ])
    train_env = VecMonitor(train_env)

    # ---- Eval env (separate, single env) ----------------------------
    eval_env = DummyVecEnv([make_env(args.size, args.view_size, args.max_steps)])
    eval_env = VecMonitor(eval_env)

    # ---- Model -------------------------------------------------------
    model = build_model(
        train_env,
        view_size=args.view_size,
        n_envs=args.n_envs,
        eta=args.eta,
        tau=args.tau,
        use_intrinsic=not args.no_intrinsic,
        device=device,
    )

    # ---- Callbacks ---------------------------------------------------
    os.makedirs("checkpoints/door_button", exist_ok=True)
    os.makedirs("eval_logs/door_button",   exist_ok=True)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="checkpoints/door_button/best",
        log_path="eval_logs/door_button",
        eval_freq=max(10_000 // args.n_envs, 1),
        n_eval_episodes=20,
        deterministic=True,
        verbose=1,
    )

    ckpt_cb = CheckpointCallback(
        save_freq=max(100_000 // args.n_envs, 1),
        save_path="checkpoints/door_button",
        name_prefix="ppogex",
    )

    # ---- Train -------------------------------------------------------
    tag = "gex" if not args.no_intrinsic else "ppo_baseline"
    print(f"\n{'='*50}")
    print(f"  DoorButton {args.size}x{args.size} — {tag}")
    print(f"  η={args.eta}  τ={args.tau}  n_envs={args.n_envs}")
    print(f"  timesteps={args.timesteps:,}")
    print(f"{'='*50}\n")

    model.learn(
        total_timesteps=args.timesteps,
        callback=[eval_cb, ckpt_cb],
        tb_log_name=tag,
        reset_num_timesteps=True,
        progress_bar=True,
    )

    # ---- Final eval --------------------------------------------------
    from stable_baselines3.common.evaluation import evaluate_policy
    mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=50)
    print(f"\nFinal eval over 50 episodes: mean_r={mean_r:.3f} ± {std_r:.3f}")

    model.save(f"checkpoints/door_button/final_{tag}")
    print("Model saved.")


if __name__ == "__main__":
    main()
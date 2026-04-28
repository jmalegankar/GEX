"""
LMU-PPO training script — Phase 2 + Phase 3.1 (ObstructedMaze).

Standard runs (MiniGrid-Memory):
────────────────────────────────
# Baseline: lifelong only, with wrapper (reproduces Phase 0 validation)
python train.py --env MemoryS11 --seed 0 --use_wrapper --beta_ep 0.0

# Phase 2: lifelong + E3B, NO wrapper (target experiment)
python train.py --env MemoryS11 --seed 0 --beta_ep 0.03

# Phase 2 hyperparameter sweep:
# β_ep ∈ {0.01, 0.03, 0.1, 0.3}  ×  λ ∈ {0.1, 1.0, 10.0}
python train.py --env MemoryS11 --seed 0 --beta_ep 0.01 --lambda_reg 0.1
python train.py --env MemoryS11 --seed 0 --beta_ep 0.03 --lambda_reg 1.0
...

Memory validity ablation:
─────────────────────────
# Retrain with view_size=3 — definitive memory test
# If this solves S11 with similar sample efficiency, memory is load-bearing.
python train.py --env MemoryS11 --seed 0 --view_size 3 --beta_ep 0.03 \
                --total_steps 5_000_000

# Compare view sizes (run all three in parallel):
for vs in 3 5 7; do
    python train.py --env MemoryS11 --seed 0 --view_size $vs --beta_ep 0.0 \
                    --tb_log runs/view_ablation &
done

Phase 3.1 — ObstructedMaze runs (random-φ E3B, no wrapper):
───────────────────────────────────────────────────────────
# Headline external-validation runs vs. Henaff 2022 Fig. 4.
# 2Dlhb is the hardest 2D variant (locked + hidden + blocked).
python train.py --env ObstructedMaze-2Dlhb --seed 0 \
    --phi_source random_encoder --beta_ep 0.1 --beta 0.0 \
    --total_steps 10_000_000 --tb_log runs/obstructedmaze_phR

# Three seeds for the headline:
for s in 0 1 2; do
    python train.py --env ObstructedMaze-2Dlhb --seed $s \
        --phi_source random_encoder --beta_ep 0.1 --beta 0.0 \
        --total_steps 10_000_000 --tb_log runs/obstructedmaze_phR &
done

# Optional difficulty ladder (2Dl easiest → 2Dlhb hardest):
python train.py --env ObstructedMaze-2Dl   --seed 0 --phi_source random_encoder ...
python train.py --env ObstructedMaze-2Dlh  --seed 0 --phi_source random_encoder ...
python train.py --env ObstructedMaze-2Dlhb --seed 0 --phi_source random_encoder ...

# Sanity check (always run before a full 10M launch on a new env):
python train.py --env ObstructedMaze-2Dlhb --seed 0 \
    --phi_source random_encoder --beta_ep 0.1 --beta 0.0 \
    --total_steps 100_000 --tb_log runs/obstructedmaze_sanity

Success criterion (Phase 2):
────────────────────────────
eval/mean_reward > 0.9 within 5M steps on S11 WITHOUT --use_wrapper.
Run 3 seeds (0, 1, 2) before drawing conclusions.

Success criterion (Phase 3.1 ObstructedMaze):
─────────────────────────────────────────────
Match or beat Henaff 2022 Fig. 4 (E3B-IDM ~0.5 at 25M steps on 2Dlhb)
at 10M steps. Run 3 seeds.
"""

import argparse
import gymnasium as gym
import minigrid  # noqa: F401
from gymnasium.wrappers import FilterObservation
from stable_baselines3.common.vec_env import (
    DummyVecEnv, SubprocVecEnv, VecTransposeImage
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from lmu_ppo.lmu_ppo import LMUPPO
from mem_start import MemoryStartWrapper


ENV_IDS = {
    # MiniGrid-Memory (Phase 0–2)
    "MemoryS5":  "MiniGrid-MemoryS5-v0",
    "MemoryS7":  "MiniGrid-MemoryS7-v0",
    "MemoryS9":  "MiniGrid-MemoryS9-v0",
    "MemoryS11": "MiniGrid-MemoryS11-v0",
    "MemoryS13": "MiniGrid-MemoryS13-v0",

    # ObstructedMaze 2D variants (Phase 3.1).
    # All three share a 2×2 room layout (4 rooms × room_size=6).
    # Source: max_steps = 4 · num_rooms_visited · room_size² = 4·4·36 = 576.
    # Difficulty: l = locked doors, h = keys hidden in boxes, b = doors blocked by balls.
    # Using -v0 for direct comparability with Henaff 2022 (E3B paper, Fig. 4);
    # Minigrid -v1 fixes a rare unsolvability bug but post-dates the published baseline.
    "ObstructedMaze-2Dl":   "MiniGrid-ObstructedMaze-2Dl-v0",
    "ObstructedMaze-2Dlh":  "MiniGrid-ObstructedMaze-2Dlh-v0",
    "ObstructedMaze-2Dlhb": "MiniGrid-ObstructedMaze-2Dlhb-v0",
}

THETA = {
    "MemoryS5":   64,
    "MemoryS7":  100,
    "MemoryS9":  120,
    "MemoryS11": 160,
    "MemoryS13": 200,

    # ObstructedMaze θ calibration (LegT only — LegS ignores θ).
    # max_steps = 576 for all 2D variants. Ideal θ ≈ max_steps to cover the full
    # episode, but with memory_size=128 that gives θ/m ≈ 4.5 steps per Legendre
    # component — coarse but workable. These tasks are a strong argument for
    # LegS (θ-free): pass --measure LegS to sidestep this entirely.
    # Easier variants have shorter typical solution paths so smaller θ suffices.
    "ObstructedMaze-2Dl":   400,
    "ObstructedMaze-2Dlh":  500,
    "ObstructedMaze-2Dlhb": 600,
}

ARCH = {
    "MemoryS5":  dict(hidden_size=64,  memory_size=32),
    "MemoryS7":  dict(hidden_size=64,  memory_size=32),
    "MemoryS9":  dict(hidden_size=128, memory_size=48),
    "MemoryS11": dict(hidden_size=128, memory_size=64),
    "MemoryS13": dict(hidden_size=128, memory_size=96),

    # ObstructedMaze: more rooms + key/door/box bindings to remember + longer
    # episodes than S13 → bump memory_size to 128. Hidden size stays at 128
    # since the per-step encoding burden (egocentric 7×7×3 patch) is unchanged.
    "ObstructedMaze-2Dl":   dict(hidden_size=128, memory_size=128),
    "ObstructedMaze-2Dlh":  dict(hidden_size=128, memory_size=128),
    "ObstructedMaze-2Dlhb": dict(hidden_size=128, memory_size=128),
}

CHUNK_LEN_DEFAULT = {
    "MemoryS5":  16,
    "MemoryS7":  16,
    "MemoryS9":  32,
    "MemoryS11": 16,
    "MemoryS13": 16,

    # ObstructedMaze: same BPTT trade-off as MemoryS13. Increase via --chunk_len
    # if VRAM permits and you want longer credit assignment within a chunk.
    "ObstructedMaze-2Dl":   16,
    "ObstructedMaze-2Dlh":  16,
    "ObstructedMaze-2Dlhb": 16,
}


def make_env(env_id: str, seed: int, rank: int = 0,
             use_wrapper: bool = False, view_size: int = None):
    def _init():
        kwargs = {}
        if view_size is not None:
            kwargs['agent_view_size'] = view_size
        env = gym.make(env_id, **kwargs)
        if use_wrapper:
            env = MemoryStartWrapper(env)
        env = FilterObservation(env, filter_keys=["image", "direction"])
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ── Environment ──────────────────────────────────────────────────
    parser.add_argument("--env", default="MemoryS11", choices=list(ENV_IDS))
    parser.add_argument("--use_wrapper", action='store_true',
                        help="Use MemoryStartWrapper (places agent near hint). "
                             "MiniGrid-Memory only — invalid for ObstructedMaze.")
    parser.add_argument("--view_size", type=int, default=None,
                        help="Override agent view size. Default: env's built-in "
                             "(7 for MiniGrid-Memory and ObstructedMaze). Set to "
                             "3 for the memory ablation test.")

    # ── Training ─────────────────────────────────────────────────────
    parser.add_argument("--seed",        type=int,   default=0)
    parser.add_argument("--n_envs",      type=int,   default=16)
    parser.add_argument("--total_steps", type=int,   default=5_000_000)
    parser.add_argument("--n_steps",     type=int,   default=512)
    parser.add_argument("--n_epochs",    type=int,   default=4)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--n_chunks_per_batch", type=int, default=32)
    parser.add_argument("--chunk_len",   type=int,   default=None,
                        help="Override default chunk length (see CHUNK_LEN_DEFAULT). "
                             "Must divide n_steps.")

    # ── Intrinsic rewards ────────────────────────────────────────────
    parser.add_argument("--beta", type=float, default=0.001,
                        help="Lifelong r_intr weight. 0 = disabled.")
    parser.add_argument("--beta_ep", type=float, default=0.03,
                        help="Episodic E3B bonus weight. 0 = disabled "
                             "(reproduces Phase 0 baseline).")
    parser.add_argument("--lambda_reg", type=float, default=1.0,
                        help="E3B regularization λ. Smaller = more exploration "
                             "pressure; larger = tighter bonus near hint.")
    parser.add_argument(
        "--phi_source",
        default="y_readout",
        choices=["y_readout", "y_readout_unnorm",
                 "random_encoder", "encoder_detached", "innovation"],
        help="Source of phi for E3B bonus. "
             "y_readout: LMU's W_query readout (default, current). "
             "y_readout_unnorm: y_readout without F.normalize on C_t. "
             "random_encoder: fresh frozen CNN (Burda 2018 baseline). "
             "encoder_detached: policy encoder with stop-gradient. "
             "innovation: u_x - u_h - u_m (LMU world-model innovation).",
    )

    # ── Memory cell ──────────────────────────────────────────────────
    parser.add_argument(
        "--measure", default="LegT", choices=["LegT", "LegS"],
        help="Memory measure. LegT=sliding window (theta required). "
             "LegS=full history, timescale-free (recommended for ObstructedMaze "
             "and Craftax-class long-episode tasks).",
    )
    parser.add_argument(
        "--gate_type", default="softsign_sum",
        choices=["softsign_sum", "tanh_product", "none"],
        help="Gate type for the LMU write. "
             "softsign_sum: fastest convergence, approximate null conditions. "
             "tanh_product: exact null conditions, slower convergence. "
             "none: no gating (ablation, lower ceiling).",
    )
    parser.add_argument(
        "--residual_scale", type=float, default=0.05,
        help="Anti-collapse residual scale for gated variants. Adds "
             "residual_scale * u_x.detach() to the write, bypassing the gate. "
             "Set to 0.0 to disable. Ignored for gate_type='none'.",
    )

    # ── Logging ──────────────────────────────────────────────────────
    parser.add_argument("--tb_log", default="runs/lmu_ppo_e3b_rnd",)
    parser.add_argument("--device", default="auto")

    args = parser.parse_args()

    # ── Argument compatibility checks ────────────────────────────────
    # MemoryStartWrapper is MiniGrid-Memory-specific; would silently misbehave
    # or crash on ObstructedMaze (different env structure, no hint object).
    if args.use_wrapper and args.env.startswith("ObstructedMaze"):
        parser.error(
            f"--use_wrapper is MiniGrid-Memory specific (places agent near "
            f"the hint object). It does not apply to ObstructedMaze tasks "
            f"(got --env {args.env}). Re-run without --use_wrapper."
        )

    env_id = ENV_IDS[args.env]
    arch   = ARCH[args.env]
    theta  = THETA[args.env]


    
    if args.chunk_len is not None:
        chunk_len = args.chunk_len
    else:
        chunk_len = CHUNK_LEN_DEFAULT[args.env]

    # n_steps must be divisible by chunk_len
    if args.n_steps % chunk_len != 0:
        old = args.n_steps
        args.n_steps = (args.n_steps // chunk_len) * chunk_len
        print(f"  [warn] n_steps adjusted {old} → {args.n_steps} "
              f"(divisible by chunk_len={chunk_len})")

    # ── Env factories ─────────────────────────────────────────────────
    train_env = VecTransposeImage(SubprocVecEnv([
        make_env(env_id, args.seed, i,
                 use_wrapper=args.use_wrapper, view_size=args.view_size)
        for i in range(args.n_envs)
    ]))
    eval_env = VecTransposeImage(DummyVecEnv([
        make_env(env_id, args.seed + 1000,
                 use_wrapper=args.use_wrapper, view_size=args.view_size)
    ]))

    eval_cb = EvalCallback(
        eval_env,
        eval_freq=max(10_000 // args.n_envs, 1),
        n_eval_episodes=20,
        verbose=1,
    )

    # ── Model ─────────────────────────────────────────────────────────
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
        target_kl=0.05,
        beta=args.beta,
        beta_ep=args.beta_ep,
        lambda_reg=args.lambda_reg,
        phi_source=args.phi_source,
        measure=args.measure,
        gate_type=args.gate_type,
        residual_scale=args.residual_scale,
        tensorboard_log=args.tb_log,
        verbose=1,
        seed=args.seed,
        device=args.device,
    )

    # ── Print config ──────────────────────────────────────────────────
    total_chunks = (args.n_steps // chunk_len) * args.n_envs
    total_params = sum(p.numel() for p in model.policy.parameters())
    view_str = f"{args.view_size}" if args.view_size else "default(7)"

    print(f"\nLMU-PPO  ·  {env_id}  ·  seed={args.seed}")
    print(f"  wrapper={'ON' if args.use_wrapper else 'OFF'}  "
          f"view_size={view_str}")
    print(f"  beta={args.beta}  beta_ep={args.beta_ep}  "
          f"lambda_reg={args.lambda_reg}  phi_source={args.phi_source}")
    print(f"  measure={args.measure}  gate_type={args.gate_type}  "
          f"residual_scale={args.residual_scale}")
    print(f"  encoder=64  hidden={arch['hidden_size']}  "
          f"memory={arch['memory_size']}  theta={theta}")
    print(f"  gamma={model.gamma}  n_envs={args.n_envs}  "
          f"total_steps={args.total_steps:,}")
    print(f"  chunk_len={chunk_len}  total_chunks={total_chunks}  "
          f"n_chunks_per_batch={args.n_chunks_per_batch}")
    print(f"  policy params: {total_params:,}")

    if args.beta_ep == 0.0:
        print("  [mode] LIFELONG ONLY — E3B disabled (baseline run)")
    elif args.use_wrapper:
        print("  [mode] E3B + WRAPPER — for ablation comparison")
    else:
        print("  [mode] E3B active — no wrapper")
    print()

    # ── Run name for TensorBoard ──────────────────────────────────────
    wrapper_tag = "wrap" if args.use_wrapper else "nowrap"
    ep_tag = f"ep{args.beta_ep}" if args.beta_ep > 0 else "noep"
    vs_tag = f"vs{args.view_size}" if args.view_size else "vs7"
    gate_tag = {
        'softsign_sum': 'ss', 'tanh_product': 'tp', 'none': 'ng'
    }[args.gate_type]
    meas_tag = args.measure.lower()
    phi_tag = {
        'y_readout': 'phY', 'y_readout_unnorm': 'phYU',
        'random_encoder': 'phR', 'encoder_detached': 'phD',
        'innovation': 'phINN',
    }[args.phi_source]
    run_name = (f"lmu_{args.env}_{wrapper_tag}_{ep_tag}_{vs_tag}_"
                f"{meas_tag}_{gate_tag}_{phi_tag}_s{args.seed}")

    model.learn(
        total_timesteps=args.total_steps,
        callback=eval_cb,
        tb_log_name=run_name,
        progress_bar=True,
    )

    save_path = run_name
    model.save(save_path)
    print(f"\nSaved → {save_path}")
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
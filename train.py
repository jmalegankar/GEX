"""
LMU-PPO training script — Phase 2 / Phase 3.

Standard runs:
──────────────
# Baseline: lifelong only, with wrapper (reproduces Phase 0 validation)
python train.py --env MemoryS11 --seed 0 --use_wrapper --beta_ep 0.0

# Phase 2: lifelong + E3B, NO wrapper (target experiment)
python train.py --env MemoryS11 --seed 0 --beta_ep 0.03

# Phase 3: ObstructedMaze (5 seeds, 10M steps)
for seed in 0 1 2 3 4; do
  python train.py --env ObstructedMaze2Dlhb --seed $seed \
    --gate_type softsign_sum --beta_ep 0.1 --lambda_reg 1.0 --beta 0.0 \
    --total_steps 10_000_000 &
done

Success criterion (Phase 3):
────────────────────────────
Any non-zero eval/mean_reward within 10M steps on ObstructedMaze2Dlhb.
If zero across all 5 seeds at 10M, stop and diagnose before scaling up.
Watch ratio_b — if it drops below 1.5 within first 2M steps, add inv-dyn aux.
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
    "MemoryS5":             "MiniGrid-MemoryS5-v0",
    "MemoryS7":             "MiniGrid-MemoryS7-v0",
    "MemoryS9":             "MiniGrid-MemoryS9-v0",
    "MemoryS11":            "MiniGrid-MemoryS11-v0",
    "MemoryS13":            "MiniGrid-MemoryS13-v0",
    # Phase 3 — ObstructedMaze variants (2Dlhb = hardest, matches E3B paper)
    "ObstructedMaze1Dl":    "MiniGrid-ObstructedMaze-1Dl-v0",
    "ObstructedMaze1Dlhb":  "MiniGrid-ObstructedMaze-1Dlhb-v0",
    "ObstructedMaze2Dl":    "MiniGrid-ObstructedMaze-2Dl-v0",
    "ObstructedMaze2Dlhb":  "MiniGrid-ObstructedMaze-2Dlhb-v0",
}

# theta must cover EXPLORATION time, not just task length.
# ObstructedMaze: agent navigates 2 rooms with locked doors + obstructions.
# Conservative upper bound on steps-to-goal from random start.
THETA = {
    "MemoryS5":             64,
    "MemoryS7":             100,
    "MemoryS9":             120,
    "MemoryS11":            160,
    "MemoryS13":            200,
    "ObstructedMaze1Dl":    300,
    "ObstructedMaze1Dlhb":  300,
    "ObstructedMaze2Dl":    400,
    "ObstructedMaze2Dlhb":  400,
}

# ObstructedMaze obs space is identical (7x7x3 egocentric), so encoder_dim=64
# stays. Larger hidden/memory for harder task — same scale as moving S11→S13.
ARCH = {
    "MemoryS5":             dict(hidden_size=64,  memory_size=32),
    "MemoryS7":             dict(hidden_size=64,  memory_size=32),
    "MemoryS9":             dict(hidden_size=128, memory_size=48),
    "MemoryS11":            dict(hidden_size=128, memory_size=64),
    "MemoryS13":            dict(hidden_size=128, memory_size=96),
    "ObstructedMaze1Dl":    dict(hidden_size=256, memory_size=128),
    "ObstructedMaze1Dlhb":  dict(hidden_size=256, memory_size=128),
    "ObstructedMaze2Dl":    dict(hidden_size=256, memory_size=128),
    "ObstructedMaze2Dlhb":  dict(hidden_size=256, memory_size=128),
}

# chunk_len must divide n_steps (default 512).
# Longer chunk for ObstructedMaze: episodes are longer, BPTT needs more context.
CHUNK_LEN_DEFAULT = {
    "MemoryS5":             16,
    "MemoryS7":             16,
    "MemoryS9":             32,
    "MemoryS11":            16,
    "MemoryS13":            16,
    "ObstructedMaze1Dl":    32,
    "ObstructedMaze1Dlhb":  32,
    "ObstructedMaze2Dl":    32,
    "ObstructedMaze2Dlhb":  32,
}

# Whether the env supports MemoryStartWrapper (Memory family only)
SUPPORTS_WRAPPER = {k: k.startswith("Memory") for k in ENV_IDS}


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
                             "Memory envs only. Ignored for ObstructedMaze.")
    parser.add_argument("--view_size", type=int, default=None,
                        help="Override agent view size. Default: env's built-in "
                             "(7 for MiniGrid-Memory). Set to 3 for the memory "
                             "ablation test.")

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

    # ── Memory cell ──────────────────────────────────────────────────
    parser.add_argument(
        "--measure", default="LegT", choices=["LegT", "LegS"],
        help="Memory measure. LegT=sliding window (theta required). "
             "LegS=full history, timescale-free (Craftax target).",
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
    parser.add_argument("--tb_log", default="runs/lmu_ppo_e3b_test",)
    parser.add_argument("--device", default="auto")

    args = parser.parse_args()

    # Guard: wrapper only valid for Memory envs
    if args.use_wrapper and not SUPPORTS_WRAPPER[args.env]:
        print(f"  [warn] --use_wrapper ignored for {args.env} "
              f"(MemoryStartWrapper is Memory-env specific)")
        args.use_wrapper = False

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
        clip_range_vf=0.2,
        max_grad_norm=0.5,
        clip_range=0.2,
        target_kl=0.05,
        beta=args.beta,
        beta_ep=args.beta_ep,
        lambda_reg=args.lambda_reg,
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

    print(f"\nLMU-PPO Phase 3  ·  {env_id}  ·  seed={args.seed}")
    print(f"  wrapper={'ON' if args.use_wrapper else 'OFF'}  "
          f"view_size={view_str}")
    print(f"  beta={args.beta}  beta_ep={args.beta_ep}  "
          f"lambda_reg={args.lambda_reg}")
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
    elif not args.use_wrapper:
        print("  [mode] E3B active, no wrapper")
    else:
        print("  [mode] E3B active + wrapper")
    print()

    # ── Run name for TensorBoard ──────────────────────────────────────
    wrapper_tag = "wrap" if args.use_wrapper else "nowrap"
    ep_tag  = f"ep{args.beta_ep}" if args.beta_ep > 0 else "noep"
    vs_tag  = f"vs{args.view_size}" if args.view_size else "vs7"
    gate_tag = {
        'softsign_sum': 'ss', 'tanh_product': 'tp', 'none': 'ng'
    }[args.gate_type]
    meas_tag = args.measure.lower()
    run_name = (f"lmu_{args.env}_{wrapper_tag}_{ep_tag}_{vs_tag}_"
                f"{meas_tag}_{gate_tag}_s{args.seed}")

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
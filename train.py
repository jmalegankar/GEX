"""
LMU-PPO training script — POPGym only.

Tier 1 (pure memory, no exploration bonus):
    python train.py --env popgym-RepeatPreviousMedium-v0 --cell vanilla_lmu \\
        --beta 0.0 --beta_ep 0.0 --total_steps 5_000_000

Cell-type sweep on one task (single seed each):
    for cell in gated_lmu vanilla_lmu gru lstm; do
        python train.py --env popgym-RepeatPreviousMedium-v0 --cell $cell \\
            --beta 0.0 --beta_ep 0.0 --total_steps 5_000_000 --seed 0 \\
            --tb_log runs/popgym_cell_ablation &
    done

Tier 2 (random-φ E3B, agency-rich tasks only):
    python train.py --env popgym-ConcentrationMedium-v0 --cell gated_lmu \\
        --phi_source random_encoder --beta 0.0 --beta_ep 0.1 \\
        --total_steps 5_000_000

Env handling:
    --env accepts only 'popgym-…-v0' IDs. The factory applies SingleKeyDictWrapper
    and (for AutoencodeMedium's Tuple obs) TupleToMultiDiscreteWrapper.
"""

import argparse
import gymnasium as gym  # noqa: F401  -- ensures gymnasium import order
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from lmu_ppo.lmu_ppo import LMUPPO
from lmu_ppo.popgym_envs import make_popgym_env
from lmu_ppo.mmer_callback import MMERCallback


# ─────────────────────────────────────────────────────────────────────────────
# Per-env defaults (POPGym). Override via CLI flags.
# ─────────────────────────────────────────────────────────────────────────────

THETA = {
    "popgym-RepeatPreviousMedium-v0": 40,
    "popgym-AutoencodeMedium-v0":     80,
    "popgym-CountRecallMedium-v0":   120,
    "popgym-ConcentrationMedium-v0": 150,
}

ARCH = {
    "popgym-RepeatPreviousMedium-v0": dict(hidden_size=128, memory_size=32),
    "popgym-AutoencodeMedium-v0":     dict(hidden_size=128, memory_size=32),
    "popgym-CountRecallMedium-v0":    dict(hidden_size=128, memory_size=48),
    "popgym-ConcentrationMedium-v0":  dict(hidden_size=128, memory_size=64),
}

CHUNK_LEN_DEFAULT = {
    "popgym-RepeatPreviousMedium-v0": 16,
    "popgym-AutoencodeMedium-v0":     16,
    "popgym-CountRecallMedium-v0":    16,
    "popgym-ConcentrationMedium-v0":  32,
}

# Fallbacks for env IDs not in the dicts above (warn loudly).
FALLBACK_THETA     = 100
FALLBACK_ARCH      = dict(hidden_size=128, memory_size=32)
FALLBACK_CHUNK_LEN = 16


# ─────────────────────────────────────────────────────────────────────────────
# Resolution helpers
# ─────────────────────────────────────────────────────────────────────────────

def resolve_env(env_arg: str):
    """
    Validate env_arg is a POPGym ID and derive a short tag for run naming.
    Returns (env_id, env_short).
    """
    if not env_arg.startswith('popgym-'):
        raise ValueError(
            f"This branch supports POPGym envs only. Use a 'popgym-…-v0' env "
            f"ID (e.g. 'popgym-RepeatPreviousMedium-v0'). Got {env_arg!r}."
        )
    env_short = env_arg.replace('popgym-', '').replace('-v0', '')
    return env_arg, env_short


def resolve_arch_theta_chunk(env_id: str, chunk_override: int = None):
    if env_id not in ARCH:
        print(f"  [warn] no defaults for {env_id}; using fallback {FALLBACK_ARCH}")
    arch  = ARCH.get(env_id, FALLBACK_ARCH)
    theta = THETA.get(env_id, FALLBACK_THETA)
    chunk = chunk_override or CHUNK_LEN_DEFAULT.get(env_id, FALLBACK_CHUNK_LEN)
    return arch, theta, chunk


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ── Environment ──────────────────────────────────────────────────
    parser.add_argument("--env", default="popgym-RepeatPreviousMedium-v0",
                        help="POPGym env ID (popgym-…-v0).")

    # ── Training ─────────────────────────────────────────────────────
    parser.add_argument("--seed",        type=int,   default=0)
    parser.add_argument("--n_envs",      type=int,   default=16)
    parser.add_argument("--total_steps", type=int,   default=5_000_000)
    parser.add_argument("--n_steps",     type=int,   default=512)
    parser.add_argument("--n_epochs",    type=int,   default=4)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--gamma",       type=float, default=0.99,
                        help="Discount factor.")
    parser.add_argument("--n_chunks_per_batch", type=int, default=32)
    parser.add_argument("--chunk_len",   type=int,   default=None,
                        help="Override default chunk length. Must divide n_steps.")
    parser.add_argument("--hidden_size", type=int,   default=None,
                        help="Override default cell hidden_size for the env.")
    parser.add_argument("--memory_size", type=int,   default=None,
                        help="Override default cell memory_size for the env.")

    # ── Cell + memory ────────────────────────────────────────────────
    parser.add_argument(
        "--cell", default="gated_lmu",
        choices=["gated_lmu", "vanilla_lmu", "gru", "lstm"],
        help="Memory cell. gated_lmu (full innovations), vanilla_lmu "
             "(POPGym-style baseline), gru, lstm.",
    )
    parser.add_argument(
        "--read_head", default="dynamic",
        choices=["dynamic", "first_coef"],
        help="LMU read head. 'dynamic'=W_query·m, 'first_coef'=m[:,0,:]. "
             "Forced to 'first_coef' when --cell vanilla_lmu. Ignored for GRU/LSTM.",
    )
    parser.add_argument(
        "--measure", default="LegT", choices=["LegT", "LegS"],
        help="LMU measure. LegT=sliding window, LegS=full history. "
             "Ignored for GRU/LSTM.",
    )
    parser.add_argument(
        "--gate_type", default="softsign_sum",
        choices=["softsign_sum", "tanh_product", "none"],
        help="LMU gate. Forced to 'none' when --cell vanilla_lmu. "
             "Ignored for GRU/LSTM.",
    )
    parser.add_argument(
        "--residual_scale", type=float, default=0.05,
        help="LMU anti-collapse residual. Forced to 0.0 when --cell vanilla_lmu. "
             "Ignored for GRU/LSTM.",
    )

    # ── Intrinsic rewards ────────────────────────────────────────────
    parser.add_argument("--beta", type=float, default=0.001,
                        help="Lifelong r_intr weight. 0 = disabled.")
    parser.add_argument("--beta_ep", type=float, default=0.03,
                        help="Episodic E3B bonus weight. 0 = disabled.")
    parser.add_argument("--lambda_reg", type=float, default=1.0,
                        help="E3B regularization λ.")
    parser.add_argument(
        "--phi_source", default="y_readout",
        choices=["y_readout", "y_readout_unnorm",
                 "random_encoder", "encoder_detached", "innovation"],
        help="phi for E3B. y_readout requires gated_lmu+dynamic. "
             "innovation requires LMU cell. random_encoder/encoder_detached: any cell.",
    )

    # ── Logging ──────────────────────────────────────────────────────
    parser.add_argument("--tb_log", default="runs/lmu_popgym")
    parser.add_argument("--device", default="auto")

    args = parser.parse_args()

    # ── Resolve env-dependent config ─────────────────────────────────
    env_id, env_short = resolve_env(args.env)
    arch, theta, chunk_len = resolve_arch_theta_chunk(
        env_id, chunk_override=args.chunk_len
    )

    hidden_size = args.hidden_size or arch['hidden_size']
    memory_size = args.memory_size or arch['memory_size']

    if args.n_steps % chunk_len != 0:
        old = args.n_steps
        args.n_steps = (args.n_steps // chunk_len) * chunk_len
        print(f"  [warn] n_steps adjusted {old} → {args.n_steps} "
              f"(divisible by chunk_len={chunk_len})")

    # ── Env factories + VecEnv construction ──────────────────────────
    train_thunks = [
        make_popgym_env(env_id, args.seed, i)
        for i in range(args.n_envs)
    ]
    train_env = SubprocVecEnv(train_thunks)
    eval_env  = DummyVecEnv([make_popgym_env(env_id, args.seed + 1000, 0)])

    eval_cb = MMERCallback(
        eval_env,
        eval_freq=max(10_000 // args.n_envs, 1),
        n_eval_episodes=20,
        verbose=1,
    )

    # ── Model ─────────────────────────────────────────────────────────
    model = LMUPPO(
        env=train_env,
        encoder_dim=64,
        hidden_size=hidden_size,
        memory_size=memory_size,
        theta=theta,
        chunk_len=chunk_len,
        gamma=args.gamma,
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
        phi_source=args.phi_source,
        measure=args.measure,
        gate_type=args.gate_type,
        residual_scale=args.residual_scale,
        read_head=args.read_head,
        cell_type=args.cell,
        tensorboard_log=args.tb_log,
        verbose=1,
        seed=args.seed,
        device=args.device,
    )

    # ── Print config ──────────────────────────────────────────────────
    total_chunks = (args.n_steps // chunk_len) * args.n_envs
    total_params = sum(p.numel() for p in model.policy.parameters())

    print(f"\nLMU-PPO  ·  POPGym: {env_id}  ·  seed={args.seed}")
    print(f"  cell={args.cell}  measure={args.measure}  read_head={args.read_head}")
    if args.cell in ('gated_lmu', 'vanilla_lmu'):
        print(f"  gate_type={args.gate_type}  residual_scale={args.residual_scale}  "
              f"theta={theta}")
    print(f"  encoder_dim=64  hidden={hidden_size}  memory={memory_size}")
    print(f"  gamma={args.gamma}  gae_lambda=0.98  n_envs={args.n_envs}  "
          f"total_steps={args.total_steps:,}")
    print(f"  chunk_len={chunk_len}  n_steps={args.n_steps}  "
          f"total_chunks={total_chunks}  n_chunks_per_batch={args.n_chunks_per_batch}")
    print(f"  beta={args.beta}  beta_ep={args.beta_ep}  "
          f"lambda_reg={args.lambda_reg}  phi_source={args.phi_source}")
    print(f"  policy params: {total_params:,}")

    if args.beta_ep == 0.0:
        print("  [mode] LIFELONG ONLY — E3B disabled (Tier 1)")
    else:
        print("  [mode] E3B active")
    print()

    # ── Run name for TensorBoard ──────────────────────────────────────
    cell_tag = {'gated_lmu': 'glmu', 'vanilla_lmu': 'vlmu',
                'gru': 'gru', 'lstm': 'lstm'}[args.cell]
    parts = [cell_tag, env_short]
    parts.append(f"ep{args.beta_ep}" if args.beta_ep > 0 else "noep")
    if args.cell in ('gated_lmu', 'vanilla_lmu'):
        parts.append(args.measure.lower())
        gate_tag = {'softsign_sum': 'ss', 'tanh_product': 'tp', 'none': 'ng'}[args.gate_type]
        parts.append(gate_tag)
        read_tag = {'dynamic': 'rdyn', 'first_coef': 'rfc'}[args.read_head]
        parts.append(read_tag)
    if args.beta_ep > 0:
        phi_tag = {'y_readout': 'phY', 'y_readout_unnorm': 'phYU',
                   'random_encoder': 'phR', 'encoder_detached': 'phD',
                   'innovation': 'phINN'}[args.phi_source]
        parts.append(phi_tag)
    parts.append(f"s{args.seed}")
    run_name = "_".join(parts)

    model.learn(
        total_timesteps=args.total_steps,
        callback=eval_cb,
        tb_log_name=run_name,
        progress_bar=True,
    )

    save_path = run_name
    model.save(save_path)
    print(f"\nSaved → {save_path}  ·  Final MMER: {eval_cb.mmer:.3f}")
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
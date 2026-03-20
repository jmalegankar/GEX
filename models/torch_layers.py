import torch as th

import math


def sinusoidal_timestep_encoding(timesteps: th.Tensor, embed_dim: int) -> th.Tensor:
    """Convert integer timesteps (B,) or (B,1) to sinusoidal positional encoding (B, embed_dim)."""
    timesteps = timesteps.view(-1)  # ensure (B,)
    half = embed_dim // 2
    freqs = th.exp(-math.log(10000.0) * th.arange(half, dtype=th.float32, device=timesteps.device) / half)
    angles = timesteps.unsqueeze(1).float() * freqs.unsqueeze(0)  # (B, half)
    pe = th.cat([th.sin(angles), th.cos(angles)], dim=-1)  # (B, 2*half)
    if embed_dim % 2 == 1:
        pe = th.cat([pe, th.zeros(pe.size(0), 1, device=pe.device)], dim=-1)
    return pe

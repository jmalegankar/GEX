"""
Observation encoders for POPGym + dispatch factory.

Each encoder maps a single-key Dict observation to a fixed-dim feature vector
of size out_dim (default 64), used as the cell input.

Encoders:
    POPGymDiscreteEncoder      — Discrete(n)            → embed → linear
    POPGymMultiDiscreteEncoder — MultiDiscrete([...])   → per-pos embed → MLP
    POPGymBoxEncoder           — Box(low, high, shape)  → MLP

Tuple obs is converted to MultiDiscrete by TupleToMultiDiscreteWrapper
upstream (popgym_envs.py), so no Tuple encoder is needed.

The dispatch factory `make_encoder` peels the outer single-key Dict and
routes on the inner space type. Both the policy encoder and the random-φ
encoder (lmu_ppo._setup_learn when phi_source='random_encoder') use this
same factory.
"""

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from typing import Dict


# ─────────────────────────────────────────────────────────────────────────────
# Discrete (RepeatPreviousMedium et al.)
# ─────────────────────────────────────────────────────────────────────────────

class POPGymDiscreteEncoder(nn.Module):
    """
    Encoder for Discrete(n) observations.

    Architecture: Embedding(n, embed_dim) → Linear(embed_dim, out_dim) → ELU.
    embed_dim default 32 — for n=4 alphabets this is much larger than needed
    but the projection cost is negligible.
    """

    def __init__(
        self,
        obs_space: spaces.Discrete,
        out_dim: int = 64,
        embed_dim: int = 32,
        key: str = 'obs',
    ):
        super().__init__()
        self._key = key
        self.embed = nn.Embedding(int(obs_space.n), embed_dim)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, out_dim),
            nn.ELU(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        # obs[key] arrives as int64 tensor of shape (B,) for Discrete.
        x = obs[self._key].long().view(-1)
        return self.proj(self.embed(x))


# ─────────────────────────────────────────────────────────────────────────────
# MultiDiscrete (Concentration, CountRecall, Autoencode-via-wrapper)
# ─────────────────────────────────────────────────────────────────────────────

class POPGymMultiDiscreteEncoder(nn.Module):
    """
    Encoder for MultiDiscrete([n_0, n_1, ...]) observations.

    If all positions share cardinality, uses one shared embedding
    (Concentration's MD([3]*100) becomes one tiny embedding). Otherwise per-
    position embeddings.

    Architecture: per-position embed → flatten → 2-layer MLP → out_dim.
    """

    def __init__(
        self,
        obs_space: spaces.MultiDiscrete,
        out_dim: int = 64,
        embed_dim: int = 8,
        proj_hidden: int = 128,
        key: str = 'obs',
    ):
        super().__init__()
        self._key = key
        self.nvec = np.asarray(obs_space.nvec).flatten().astype(np.int64)
        self.n_positions = len(self.nvec)

        if (self.nvec == self.nvec[0]).all():
            self.embed = nn.Embedding(int(self.nvec[0]), embed_dim)
            self.embeds = None
            self.shared = True
        else:
            self.embed = None
            self.embeds = nn.ModuleList([
                nn.Embedding(int(n), embed_dim) for n in self.nvec
            ])
            self.shared = False

        in_dim = embed_dim * self.n_positions
        self.proj = nn.Sequential(
            nn.Linear(in_dim, proj_hidden),
            nn.ELU(),
            nn.Linear(proj_hidden, out_dim),
            nn.ELU(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = obs[self._key].long()
        if x.dim() == 1:
            x = x.unsqueeze(-1)

        if self.shared:
            e = self.embed(x)
        else:
            e = torch.stack(
                [emb(x[..., i]) for i, emb in enumerate(self.embeds)],
                dim=-2,
            )

        return self.proj(e.flatten(start_dim=-2))


# ─────────────────────────────────────────────────────────────────────────────
# Box (control tasks)
# ─────────────────────────────────────────────────────────────────────────────

class POPGymBoxEncoder(nn.Module):
    """
    Encoder for Box observations (continuous vectors).
    Architecture: Linear → ELU → Linear → ELU.
    """

    def __init__(
        self,
        obs_space: spaces.Box,
        out_dim: int = 64,
        hidden: int = 128,
        key: str = 'obs',
    ):
        super().__init__()
        self._key = key
        in_dim = int(np.prod(obs_space.shape))
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, out_dim),
            nn.ELU(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = obs[self._key].float().flatten(start_dim=1)
        return self.mlp(x)


# ─────────────────────────────────────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────────────────────────────────────

def is_singlekey_dict_space(obs_space: spaces.Space, key: str = 'obs') -> bool:
    """True if obs_space is Dict({key: ...}) with exactly one key."""
    return (
        isinstance(obs_space, spaces.Dict)
        and len(obs_space.spaces) == 1
        and key in obs_space.spaces
    )


def make_encoder(
    obs_space: spaces.Space,
    out_dim: int = 64,
    key: str = 'obs',
) -> nn.Module:
    """
    Factory: return an encoder appropriate to obs_space.

    Routes:
        Single-key Dict({key: Discrete})       → POPGymDiscreteEncoder
        Single-key Dict({key: MultiDiscrete})  → POPGymMultiDiscreteEncoder
        Single-key Dict({key: Box})            → POPGymBoxEncoder

    Tuple is not handled here — TupleToMultiDiscreteWrapper converts it
    upstream in popgym_envs.make_popgym_env.
    """
    if not is_singlekey_dict_space(obs_space, key):
        raise ValueError(
            f"Expected single-key Dict obs (POPGym wrapped via "
            f"SingleKeyDictWrapper), got {type(obs_space).__name__}"
            f"{f' with keys {list(obs_space.spaces)}' if isinstance(obs_space, spaces.Dict) else ''}."
        )

    inner = obs_space.spaces[key]

    if isinstance(inner, spaces.Discrete):
        return POPGymDiscreteEncoder(inner, out_dim=out_dim, key=key)
    if isinstance(inner, spaces.MultiDiscrete):
        return POPGymMultiDiscreteEncoder(inner, out_dim=out_dim, key=key)
    if isinstance(inner, spaces.Box):
        return POPGymBoxEncoder(inner, out_dim=out_dim, key=key)

    raise ValueError(
        f"Unsupported inner POPGym obs space inside Dict[{key!r}]: "
        f"{type(inner).__name__}. Expected Discrete, MultiDiscrete, or Box. "
        f"For Tuple(Discrete, ...), wrap with TupleToMultiDiscreteWrapper "
        f"upstream of SingleKeyDictWrapper."
    )
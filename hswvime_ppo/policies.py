import warnings

import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule

from typing import Any, Optional, Union, Tuple

from models.vae import VAEInterface, TransitionSCVAE
from models.wyner import WynerVAE


class HSWVIMEFeaturesExtractor(nn.Module):
    """
    Policy feature extractor.

    Concatenates the Wyner latent z_t (common cause of past/future) with
    the current SCVAE latent mu_t (immediate observation encoding).

        features_t = concat( sg(z_t), sg(mu_t) )   shape: (B, wyner_dim + mu_dim)

    Both inputs are stop-gradiented before concatenation. The policy MLP
    trains on these features via PPO loss, but PPO gradients must not reach
    the WynerVAE or SCVAE — those are updated by their own dedicated losses.

    Why not attention here:
        The old MHA attended mu_t over the Wyner latent, trying to extract
        memory-like context from w_t. That role is now properly handled by
        h_slow inside the WynerVAE. z_t IS the distilled common-cause
        representation; concatenating it with mu_t gives the policy both
        "what is happening now" (mu_t) and "what is causally happening"
        (z_t, which was forced to explain both past and future).
    """

    def __init__(self, observation_space=None, *, wyner_dim: int, mu_dim: int):
        super().__init__()
        self._features_dim = wyner_dim + mu_dim
        self.wyner_dim = wyner_dim
        self.mu_dim = mu_dim

    @property
    def features_dim(self) -> int:
        return self._features_dim

    def forward(self, z_t: th.Tensor, mu_t: th.Tensor) -> th.Tensor:
        # Both already detached by extract_features — assert defensively
        return th.cat([z_t, mu_t], dim=-1)   # (B, wyner_dim + mu_dim)


class HSWVIMEActorCriticPolicy(ActorCriticPolicy):

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: type[HSWVIMEFeaturesExtractor] = HSWVIMEFeaturesExtractor,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        vae_features_extractor_class: VAEInterface = TransitionSCVAE,
        vae_features_extractor_kwargs: Optional[dict[str, Any]] = None,
        wyner_features_extractor_class = WynerVAE,
        wyner_features_extractor_kwargs: Optional[dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
    ):
        assert share_features_extractor, \
            "HSWVIME does not support separate feature extractors for policy and value networks"

        self.vae_features_extractor_class  = vae_features_extractor_class
        self.vae_features_extractor_kwargs = vae_features_extractor_kwargs or {}
        self.wyner_features_extractor_class  = wyner_features_extractor_class
        self.wyner_features_extractor_kwargs = wyner_features_extractor_kwargs or {}

        super().__init__(
            observation_space, action_space, lr_schedule,
            net_arch, activation_fn, ortho_init, use_sde, log_std_init,
            full_std, use_expln, squash_output,
            features_extractor_class, features_extractor_kwargs,
            share_features_extractor, normalize_images,
            optimizer_class, optimizer_kwargs,
        )

    def make_features_extractor(self):
        self.vae_feature_extractor = self.vae_features_extractor_class(
            **self.vae_features_extractor_kwargs
        )
        self.wyner_feature_extractor = self.wyner_features_extractor_class(
            **self.wyner_features_extractor_kwargs
        )
        return super().make_features_extractor()

    # ── Core feature extraction ──────────────────────────────────────────────

    def extract_features(
        self,
        s_tm1:  th.Tensor,   # (B, *obs_shape)  previous obs
        a_tm1:  th.Tensor,   # (B, act_dim)     previous action
        s_t:    th.Tensor,   # (B, *obs_shape)  current obs
        h_prev: th.Tensor,   # (B, context_dim) WynerVAE slow-path hidden state h_{t-1}
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        Returns (features, h_t).

        features = concat(sg(z_t), sg(mu_t))   — detached; PPO does not touch SCVAE/WynerVAE
        h_t      = updated slow-path hidden state; stored in buffer as memory for next step

        Gradient isolation:
          - SCVAE is trained only by its own VAE loss (computed in train())
          - WynerVAE is trained only by its own Wyner loss (computed in train())
          - PPO loss sees only detached features
        """
        s_tm1 = preprocess_obs(s_tm1, self.observation_space, normalize_images=self.normalize_images)
        s_t   = preprocess_obs(s_t,   self.observation_space, normalize_images=self.normalize_images)

        # SCVAE encode: mu_t is the spherical Cauchy latent for the (t-1 → t) transition
        mu_t, _, _ = self.vae_feature_extractor.encode(s_tm1, a_tm1, s_t)

        # WynerVAE rollout: sample z_t from prior (mu_{t+1} unavailable at act time),
        # update slow-path context.
        # h_prev = h_{t-1}; h_t will be stored in buffer for next step.
        z_t, h_t = self.wyner_feature_extractor.forward_rollout(mu_t, h_prev)

        # Detach both before the policy MLP — PPO gradient stops here
        features = self.features_extractor(z_t.detach(), mu_t.detach())
        return features, h_t

    # ── Policy interface ─────────────────────────────────────────────────────

    def forward(
        self,
        s_tm1: th.Tensor,
        a_tm1: th.Tensor,
        s_t:   th.Tensor,
        h_prev: th.Tensor,
        deterministic: bool = False,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """Returns (actions, h_t, values, log_probs)."""
        features, h_t = self.extract_features(s_tm1, a_tm1, s_t, h_prev)
        latent_pi, latent_vf = self.mlp_extractor(features)
        values       = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions      = distribution.get_actions(deterministic=deterministic)
        log_prob     = distribution.log_prob(actions)
        actions      = actions.reshape((-1, *self.action_space.shape))
        return actions, h_t, values, log_prob

    def get_distribution(
        self,
        s_tm1: th.Tensor, a_tm1: th.Tensor,
        s_t: th.Tensor, h_prev: th.Tensor,
    ) -> Tuple[Distribution, th.Tensor]:
        features, h_t = self.extract_features(s_tm1, a_tm1, s_t, h_prev)
        latent_pi = self.mlp_extractor.forward_actor(features)
        return self._get_action_dist_from_latent(latent_pi), h_t

    def predict_values(
        self,
        s_tm1: th.Tensor, a_tm1: th.Tensor,
        s_t: th.Tensor, h_prev: th.Tensor,
    ) -> th.Tensor:
        features, _ = self.extract_features(s_tm1, a_tm1, s_t, h_prev)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf)

    def evaluate_actions(
        self,
        s_tm1:  th.Tensor,
        a_tm1:  th.Tensor,
        s_t:    th.Tensor,
        h_prev: th.Tensor,
        action: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        features, h_t = self.extract_features(s_tm1, a_tm1, s_t, h_prev)
        latent_vf    = self.mlp_extractor.forward_critic(features)
        values       = self.value_net(latent_vf)
        latent_pi    = self.mlp_extractor.forward_actor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob     = distribution.log_prob(action)
        return values, log_prob, distribution.entropy(), h_t

    def predict(
        self,
        s_tm1: th.Tensor, a_tm1: th.Tensor,
        s_t: th.Tensor, h_prev: th.Tensor,
        deterministic: bool = False,
    ) -> Tuple[th.Tensor, th.Tensor]:
        self.set_training_mode(False)

        if isinstance(s_tm1, tuple) and len(s_tm1) == 2 and isinstance(s_tm1[1], dict):
            raise ValueError(
                "You have passed a tuple to predict() instead of an array. "
                "You are likely mixing Gym API with SB3 VecEnv API."
            )

        s_tm1, vectorized_env = self.obs_to_tensor(s_tm1)
        s_t, _  = self.obs_to_tensor(s_t)
        a_tm1   = th.as_tensor(a_tm1, device=s_tm1.device)

        with th.no_grad():
            distribution, h_t = self.get_distribution(s_tm1, a_tm1, s_t, h_prev)
            actions = distribution.get_actions(deterministic=deterministic)

        actions = actions.cpu().numpy().reshape((-1, *self.action_space.shape))

        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                actions = self.unscale_action(actions)
            else:
                actions = np.clip(actions, self.action_space.low, self.action_space.high)

        if not vectorized_env:
            actions = actions.squeeze(axis=0)

        return actions, h_t
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


class SimpleVAEFeaturesExtractor(nn.Module):
    """
    Features extractor that uses the VAE mu directly as policy features.
    """

    def __init__(self, observation_space=None, *, mu_dim: int):
        super().__init__()
        self._features_dim = mu_dim

    @property
    def features_dim(self) -> int:
        return self._features_dim

    def forward(self, mu_features: th.Tensor) -> th.Tensor:
        return mu_features


class HSWVIMEActorCriticPolicy(ActorCriticPolicy):
    """
    Actor-critic policy that integrates a transition VAE.
    Features are the VAE's mu (direction on the sphere).
    """

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
        features_extractor_class: type[SimpleVAEFeaturesExtractor] = SimpleVAEFeaturesExtractor,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        vae_features_extractor_class: VAEInterface = TransitionSCVAE,
        vae_features_extractor_kwargs: Optional[dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
    ):
        assert share_features_extractor, "Does not support separate feature extractors for policy and value networks"
        self.vae_features_extractor_class = vae_features_extractor_class
        self.vae_features_extractor_kwargs = vae_features_extractor_kwargs or {}

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )

    def make_features_extractor(self):
        self.vae_feature_extractor: VAEInterface = self.vae_features_extractor_class(
            **self.vae_features_extractor_kwargs
        )
        return super().make_features_extractor()

    def forward(
        self,
        s_tm1: th.Tensor,
        a_tm1: th.Tensor,
        s_t: th.Tensor,
        deterministic: bool = False,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic).

        :return: action, value, log probability of the action
        """
        features = self.extract_features(s_tm1, a_tm1, s_t)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob

    def extract_features(
        self,
        s_tm1: th.Tensor,
        a_tm1: th.Tensor,
        s_t: th.Tensor,
    ) -> th.Tensor:
        s_tm1 = preprocess_obs(s_tm1, self.observation_space, normalize_images=self.normalize_images)
        s_t = preprocess_obs(s_t, self.observation_space, normalize_images=self.normalize_images)
        mu, _, skips = self.vae_feature_extractor.encode(s_tm1, a_tm1, s_t)
        features = self.features_extractor(mu)
        return features

    def get_distribution(
        self,
        s_tm1: th.Tensor,
        a_tm1: th.Tensor,
        s_t: th.Tensor,
    ) -> Distribution:
        features = self.extract_features(s_tm1, a_tm1, s_t)
        latent_pi = self.mlp_extractor.forward_actor(features)
        return self._get_action_dist_from_latent(latent_pi)

    def predict_values(
        self,
        s_tm1: th.Tensor,
        a_tm1: th.Tensor,
        s_t: th.Tensor,
    ) -> th.Tensor:
        features = self.extract_features(s_tm1, a_tm1, s_t)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf)

    def evaluate_actions(
        self,
        s_tm1: th.Tensor,
        a_tm1: th.Tensor,
        s_t: th.Tensor,
        action: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        features = self.extract_features(s_tm1, a_tm1, s_t)
        latent_vf = self.mlp_extractor.forward_critic(features)
        values = self.value_net(latent_vf)
        latent_pi = self.mlp_extractor.forward_actor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(action)
        return values, log_prob, distribution.entropy()

    def predict(
        self,
        s_tm1: th.Tensor,
        a_tm1: th.Tensor,
        s_t: th.Tensor,
        deterministic: bool = False,
    ) -> th.Tensor:
        self.set_training_mode(False)

        if isinstance(s_tm1, tuple) and len(s_tm1) == 2 and isinstance(s_tm1[1], dict):
            raise ValueError(
                "You have passed a tuple to the predict() function instead of a Numpy array or a Dict. "
                "You are probably mixing Gym API with SB3 VecEnv API."
            )

        s_tm1, vectorized_env = self.obs_to_tensor(s_tm1)
        s_t, _ = self.obs_to_tensor(s_t)
        a_tm1 = th.as_tensor(a_tm1, device=s_tm1.device)

        with th.no_grad():
            distribution = self.get_distribution(s_tm1, a_tm1, s_t)
            actions = distribution.get_actions(deterministic=deterministic)

        actions = actions.cpu().numpy().reshape((-1, *self.action_space.shape))

        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                actions = self.unscale_action(actions)
            else:
                actions = np.clip(actions, self.action_space.low, self.action_space.high)

        if not vectorized_env:
            assert isinstance(actions, np.ndarray)
            actions = actions.squeeze(axis=0)

        return actions

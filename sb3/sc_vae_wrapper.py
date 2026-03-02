import numpy as np
import torch as th

class SCVAEEncoderWrapper:
    """
    Wraps TransitionSCVAE to accept numpy batches from SB3 VecEnv and return torch mu.
    """

    def __init__(self, sc_vae, device: th.device):
        self.sc_vae = sc_vae
        self.device = device

    @th.no_grad()
    def encode_mu(self, obs: np.ndarray, actions: np.ndarray, next_obs: np.ndarray) -> th.Tensor:
        """
        Args:
            obs:      (n_envs, *obs_shape) numpy
            actions:  (n_envs,) or (n_envs, 1) numpy
            next_obs: (n_envs, *obs_shape) numpy
        Returns:
            mu: (n_envs, d) torch float on device
        """
        if actions.ndim == 2 and actions.shape[1] == 1:
            actions = actions[:, 0]

        s_t = th.as_tensor(obs, device=self.device)
        a_t = th.as_tensor(actions, device=self.device)
        s_n = th.as_tensor(next_obs, device=self.device)

        # ensure batch dimension exists
        if s_t.dim() == 1:
            s_t = s_t.unsqueeze(0)
            s_n = s_n.unsqueeze(0)
            a_t = a_t.unsqueeze(0)

        mu, _rho = self.sc_vae.encode(s_t, a_t, s_n)
        return mu
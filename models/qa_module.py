import numpy as np
from scipy.special import erf
from hswvime_ppo.buffer import TransitionRolloutBuffer

from typing import Optional

class QASampler:
    def __init__(self, buffer: TransitionRolloutBuffer):
        self.buffer = buffer
        self.means = np.zeros(buffer.n_envs, dtype=np.float32)
        self.exmeansq = np.zeros(buffer.n_envs, dtype=np.float32)
        self.num_qa = buffer.n_envs
        self.scores = np.zeros((buffer.n_envs, buffer.num_qa), dtype=np.float32)
        self.timesteps = np.zeros((buffer.n_envs), dtype=np.long)
    
    def reset(self, idx: Optional[int] = None):
        if idx is not None:
            self.scores[idx, :] = 0
            self.timesteps[idx] = 0
        else:
            self.scores[...] = 0
            self.timesteps[...] = 0
    
    def update(self, scores: np.ndarray, answers: np.ndarray):
        num_timesteps = self.timesteps + 1
        self.means = (self.timesteps/num_timesteps) * self.means + scores / num_timesteps
        self.exmeansq = (self.timesteps/num_timesteps) * self.exmeansq + (scores**2) / num_timesteps

        std = np.sqrt(np.maximum(self.exmeansq - self.means**2, 1e-8))
        mask = self.timesteps > 0
        # Sample replacement entry with probability given by cdf of z-score
        z_scores = (self.scores.T - self.means) / std
        # Calculate the cdf of the z-scores
        cdf = 0.5 * (1 + erf(z_scores / np.sqrt(2)))
        
        z_scores = (scores - self.means) / std
        cdf_new = 0.5 * (1 + erf(z_scores / np.sqrt(2)))

        replace = (np.random.rand(*cdf.shape) < cdf_new / (cdf + 1e-8))
        replace = replace.argmax(axis=0)

        pos = self.buffer.pos - 1

        self.buffer.questions[pos, ~mask, :] = 0
        self.buffer.answers[pos, ~mask, :, :] = answers[~mask].reshape(-1, 1, self.buffer.answer_dim)
        self.scores[~mask, :] = scores[~mask].reshape(-1, 1)

        self.buffer.questions[pos, mask, replace[mask]] = self.timesteps[mask]
        self.buffer.answers[pos, mask, replace[mask], :] = answers[mask]
        self.scores[mask, replace[mask]] = scores[mask]

        self.timesteps += 1
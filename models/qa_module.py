import numpy as np
from scipy.special import erf
from lmu_ppo.buffer import TransitionRolloutBuffer

from typing import Optional

class QASampler:
    def __init__(self, buffer: TransitionRolloutBuffer):
        self.buffer = buffer
        self.means = np.zeros(buffer.n_envs, dtype=np.float32)
        self.exmeansq = np.zeros(buffer.n_envs, dtype=np.float32)
        self.num_qa = buffer.n_envs
        self.scores = np.zeros((buffer.n_envs, buffer.num_qa), dtype=np.float32)
        self.timesteps = np.zeros((buffer.n_envs), dtype=np.float64)
        self._env_idx = np.arange(buffer.n_envs)
        self._cur_questions = np.zeros((buffer.n_envs, buffer.num_qa), dtype=np.float64)
        self._cur_answers = np.zeros((buffer.n_envs, buffer.num_qa, buffer.answer_dim), dtype=np.float32)
    
    def reset(self, idx: Optional[int] = None):
        if idx is not None:
            self.scores[idx, :] = 0
            self.timesteps[idx] = 0
            self.means[idx] = 0.0
            self.exmeansq[idx] = 0.0
            self._cur_questions[idx, :] = 0
            self._cur_answers[idx, :, :] = 0.0
        else:
            self.scores[...] = 0
            self.timesteps[...] = 0
            self.means[...] = 0.0
            self.exmeansq[...] = 0.0
            self._cur_questions[...] = 0
            self._cur_answers[...] = 0.0
    
    def update(self, scores: np.ndarray, answers: np.ndarray):
        assert np.isnan(scores).any() == False, "Scores contain NaN values"
        num_timesteps = self.timesteps + 1
        self.means = (self.timesteps/num_timesteps) * self.means + scores / num_timesteps
        self.exmeansq = (self.timesteps/num_timesteps) * self.exmeansq + (scores**2) / num_timesteps

        std = np.sqrt(np.maximum(self.exmeansq - self.means**2, 1e-8))
        mask = self.timesteps > 0
        # Sample replacement entry with probability given by cdf of z-score
        z_scores = (self.scores.T - self.means) / std
        # Calculate the cdf of the z-scores
        cdf = 0.5 * (1 + erf(z_scores / np.sqrt(2))) + 1e-8 # add small constant to prevent division by zero
        
        z_scores = (scores - self.means) / std
        cdf_new = 0.5 * (1 + erf(z_scores / np.sqrt(2)))

        replace = cdf_new / cdf + np.random.rand(*cdf.shape)*1e-6 # add small noise to break ties
        replace = replace.argmax(axis=0)

        pos = self.buffer.pos - 1

        self.buffer.questions[pos, ...] = self._cur_questions
        self.buffer.answers[pos, ...] = self._cur_answers

        self.buffer.questions[pos, ~mask, ...] = 0
        self.buffer.answers[pos, ~mask, :, :] = answers[~mask].reshape(-1, 1, self.buffer.answer_dim)
        self.scores[~mask, :] = scores[~mask].reshape(-1, 1)

        mask = mask & (np.random.rand(*cdf_new.shape) < (cdf_new / cdf[replace, self._env_idx]))

        self.buffer.questions[pos, self._env_idx[mask], replace[mask]] = self.timesteps[self._env_idx[mask]]
        self.buffer.answers[pos, self._env_idx[mask], replace[mask], :] = answers[self._env_idx[mask], ...]
        self.scores[self._env_idx[mask], replace[mask]] = scores[self._env_idx[mask]]

        self._cur_questions[...] = self.buffer.questions[pos, ...]
        self._cur_answers[...] = self.buffer.answers[pos, ...]

        self.timesteps += 1
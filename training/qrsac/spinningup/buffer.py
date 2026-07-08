"""MPI-free FIFO experience replay buffer.

Backs the legacy distributed/tabled replay path (``qrsac/replay.py``). The
single-process QR-SAC trainer uses its own ``NStepReplayBuffer`` defined in
``standalone_trainer.py`` and does not depend on this module.
"""

import numpy as np
import torch

from . import core


class ReplayBuffer:
    """A simple FIFO experience replay buffer for SAC agents."""

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)  # type: ignore
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)  # type: ignore
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)  # type: ignore
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs=self.obs_buf[idxs],
            next_obs=self.obs2_buf[idxs],
            action=self.act_buf[idxs],
            reward=self.rew_buf[idxs],
            done=self.done_buf[idxs],
        )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}

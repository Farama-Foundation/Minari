from typing import NamedTuple
import numpy as np
from minari.dataset.minari_dataset import MinariDataset
import gymnasium as gym


class ReplayBuffer:

    def __init__(self, buffer_size: int) -> None:
        self.max_size = buffer_size
        self.size = 0

        self._obs = None
        self._actions = None
        self._rewards = None
        self._terminations = None
        self._truncations = None
        self._next_obs = None

    def load(self, dataset: MinariDataset) -> None:
        if self.is_empty():
            self._allocate_memory(
                observation_space=dataset.observation_space,
                action_space=dataset.action_space
            )
        if self.is_full():
            raise ValueError("ReplayBuffer is already full!")

        for ep_data in dataset:
            loaded_length = min(self.max_size - self.size, ep_data.total_timesteps)
            end_idx = self.size + loaded_length

            self._obs[self.size:end_idx] = ep_data.observations[:loaded_length]
            self._next_obs[self.size:end_idx - 1] = ep_data.observations[1:loaded_length]
            self._actions[self.size:end_idx] = ep_data.actions[:loaded_length]
            self._rewards[self.size:end_idx] = ep_data.rewards[:loaded_length]
            self._terminations[self.size:end_idx] = ep_data.terminations[:loaded_length]
            self._truncations[self.size:end_idx] = ep_data.truncations[:loaded_length]

            self.size = end_idx
            if self.is_full():
                break

    def sample(self, batch_size: int):
        if self.is_empty():
            raise ValueError("ReplayBuffer is empty!")

        indices = np.random.randint(0, self.size, size=batch_size)
        obs = self._obs[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_obs = self._next_obs[indices]
        terminations = self._terminations[indices]
        truncations = self._truncations[indices]

        return NamedTuple(
            obs=obs,
            actions=actions,
            rewards=rewards,
            terminations=terminations,
            truncations=truncations,
            next_obs=next_obs,
        )

    def _allocate_memory(self, observation_space: gym.Space, action_space: gym.Space):
        assert observation_space.shape is not None
        assert action_space.shape is not None
    
        self._obs = np.zeros((self.max_size, *observation_space.shape), dtype=observation_space.dtype)
        self._actions = np.zeros((self.max_size, *action_space.shape), dtype=action_space.dtype)
        self._rewards = np.zeros((self.max_size, 1), dtype=np.float32)
        self._terminations = np.zeros((self.max_size, 1), dtype=np.bool8)
        self._truncations = np.zeros((self.max_size, 1), dtype=np.bool8)
        self._next_obs = np.zeros((self.max_size, *observation_space.shape), dtype=observation_space.dtype)

        self.size = 0

    def is_empty(self):
        return (
            self.size == 0 or
            self._obs is None or
            self._actions is None or
            self._terminations is None or
            self._truncations is None or
            self._next_obs is None
        )

    def is_full(self):
        return self.size >= self.max_size

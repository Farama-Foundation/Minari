import logging
from typing import NamedTuple
import numpy as np
from minari.dataset.minari_dataset import EpisodeData, MinariDataset
from gymnasium import spaces


class ReplayBuffer:

    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space
    ) -> None:
        self.max_size = buffer_size
        self.size = 0

        self.observation_space = observation_space
        self.action_space = action_space

        obs_shape = spaces.flatten(observation_space, observation_space.sample()).shape
        action_shape = spaces.flatten(action_space, action_space.sample()).shape

        self._obs = np.zeros((self.max_size, *obs_shape), dtype=observation_space.dtype)
        self._actions = np.zeros((self.max_size, *action_shape), dtype=action_space.dtype)
        self._rewards = np.zeros((self.max_size, 1), dtype=np.float32)
        self._terminations = np.zeros((self.max_size, 1), dtype=np.bool8)
        self._truncations = np.zeros((self.max_size, 1), dtype=np.bool8)
        self._next_obs = np.zeros((self.max_size, *obs_shape), dtype=observation_space.dtype)

    def load(self, dataset: MinariDataset) -> None:
        for ep_data in dataset:
            self.add_episode(ep_data)

            if self.is_full():
                logging.warn("The dataset was partially loaded because it exceeds the buffer size")
                break

    def add_episode(self, episode_data: EpisodeData):
        if self.is_full():
            raise ValueError("ReplayBuffer is already full!")

        loaded_length = min(self.max_size - self.size, episode_data.total_timesteps)
        end_idx = self.size + loaded_length

        self._obs[self.size:end_idx] = episode_data.observations[:loaded_length]
        self._next_obs[self.size:end_idx - 1] = episode_data.observations[1:loaded_length]
        self._actions[self.size:end_idx] = episode_data.actions[:loaded_length]
        self._rewards[self.size:end_idx] = episode_data.rewards[:loaded_length]
        self._terminations[self.size:end_idx] = episode_data.terminations[:loaded_length]
        self._truncations[self.size:end_idx] = episode_data.truncations[:loaded_length]

        self.size = end_idx

    def sample(self, batch_size: int):
        if self.size == 0:
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

    def is_full(self):
        return self.size >= self.max_size

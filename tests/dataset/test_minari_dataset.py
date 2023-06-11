import gymnasium as gym
import numpy as np
import pytest

from minari.dataset.minari_dataset import EpisodeData
from tests.test_common import test_spaces


@pytest.mark.parametrize("space", test_spaces)
def test_episode_data(space: gym.Space):
    id = np.random.randint(1024)
    seed = np.random.randint(1024)
    total_timestep = 100
    rewards = np.random.randn(total_timestep)
    terminations = np.random.choice([True, False], size=(total_timestep,))
    truncations = np.random.choice([True, False], size=(total_timestep,))
    episode_data = EpisodeData(
        id=id,
        seed=seed,
        total_timesteps=total_timestep,
        observations=space.sample(),
        actions=space.sample(),
        rewards=rewards,
        terminations=terminations,
        truncations=truncations,
    )

    repr(episode_data)

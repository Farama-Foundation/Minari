import gymnasium as gym
import numpy as np
import pytest

from minari.data_collector.episode_buffer import EpisodeBuffer
from minari.dataset.minari_storage import MinariStorage


def _storage(path, format, lengths):
    storage = MinariStorage.new(
        path,
        observation_space=gym.spaces.Box(-1e6, 1e6, (1,)),
        action_space=gym.spaces.Discrete(2),
        data_format=format,
    )
    for index, length in enumerate(lengths):
        storage.update_episodes(
            [
                EpisodeBuffer(
                    observations=np.arange(length + 1, dtype=np.float32).reshape(-1, 1),
                    actions=np.zeros(length, dtype=np.int64),
                    rewards=[float(index + 1)] * length,
                    terminations=[False] * length,
                    truncations=[False] * length,
                )
            ]
        )
    return storage


@pytest.mark.parametrize("format", ["arrow", "parquet"])
def test_get_episodes_accepts_one_shot_indices(tmp_path, format):
    storage = _storage(tmp_path, format, [2, 3])
    episodes = list(storage.get_episodes(iter([1, 0])))
    assert [episode["id"] for episode in episodes] == [1, 0]
    assert [len(episode["rewards"]) for episode in episodes] == [3, 2]


@pytest.mark.parametrize("format", ["arrow", "parquet"])
def test_get_episodes_does_not_split_long_episodes(tmp_path, format):
    length = 140000
    storage = _storage(tmp_path, format, [length, 2])
    episodes = list(storage.get_episodes([0, 1]))
    assert [len(episode["rewards"]) for episode in episodes] == [length, 2]
    np.testing.assert_array_equal(
        episodes[0]["observations"].ravel(), np.arange(length + 1)
    )
    np.testing.assert_array_equal(episodes[1]["rewards"], [2, 2])

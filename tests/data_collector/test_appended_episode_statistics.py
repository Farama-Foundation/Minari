import gymnasium as gym
import numpy as np
import pytest

from minari import DataCollector, EpisodeMetadataCallback, MinariDataset
from minari.dataset._storages import get_storage_keys


class CountingMetadataCallback(EpisodeMetadataCallback):
    def __init__(self):
        self.calls = 0

    def __call__(self, episode):
        self.calls += 1
        return {
            **super().__call__(episode),
            "callback_call": self.calls,
            "callback_episode_id": int(episode["id"]),
        }


def collect_episode(env, seed):
    env.reset(seed=seed)
    for _ in range(5):
        env.step(np.array([0.5], dtype=np.float32))


@pytest.mark.parametrize("data_format", get_storage_keys())
@pytest.mark.parametrize(
    "callback", [EpisodeMetadataCallback, CountingMetadataCallback]
)
@pytest.mark.parametrize("filtered", [False, True])
def test_appended_episode_statistics(data_format, callback, filtered):
    """Checkpointing computes metadata once for each newly saved episode."""
    with DataCollector(
        gym.make("Pendulum-v1", max_episode_steps=5),
        data_format=data_format,
        episode_metadata_callback=callback,
    ) as collector:
        for seed in range(2):
            collect_episode(collector, seed)
        dataset = collector.create_dataset("pendulum-statistics-v0")
        if filtered:
            dataset = dataset.filter_episodes(lambda episode: episode.id == 1)

        for seed in range(2, 4):
            previous = list(dataset.storage.get_episode_metadata(range(seed)))
            collect_episode(collector, seed)
            collector.add_to_dataset(dataset)
            assert list(dataset.storage.get_episode_metadata(range(seed))) == previous

    # Read the full dataset again to check persisted values, including episodes
    # excluded from the view used for appending.
    dataset = MinariDataset(dataset.storage.data_path)
    metadata = dataset.storage.get_episode_metadata(dataset.episode_indices)
    for episode, values in zip(dataset, metadata):
        rewards = episode.rewards
        assert values["rewards_sum"] == rewards.sum()
        assert values["rewards_mean"] == rewards.mean()
        assert values["rewards_std"] == rewards.std()
        assert values["rewards_min"] == rewards.min()
        assert values["rewards_max"] == rewards.max()
        if callback is CountingMetadataCallback:
            assert values["callback_call"] == episode.id + 1
            assert values["callback_episode_id"] == episode.id

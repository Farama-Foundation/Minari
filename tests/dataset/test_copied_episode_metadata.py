import gymnasium as gym
import numpy as np
import pytest

import minari
from minari import DataCollector
from minari.dataset._storages import get_storage_keys


def collect_episode(env, seed, options):
    env.reset(seed=seed, options=options)
    for _ in range(5):
        env.step(np.array([0.5], dtype=np.float32))


@pytest.mark.parametrize("source_format", get_storage_keys())
@pytest.mark.parametrize("target_format", get_storage_keys())
@pytest.mark.parametrize("seed", [0, 2**64 - 1, None])
def test_appended_episodes_keep_reset_metadata(source_format, target_format, seed):
    """An appended episode can be replayed using its saved reset arguments."""
    options = {"x_init": 1, "y_init": 0.25, "minari_autoseed": False}
    with DataCollector(
        gym.make("Pendulum-v1", max_episode_steps=5), data_format=target_format
    ) as target:
        collect_episode(target, 42, options)
        dataset = target.create_dataset("pendulum-copy-v0")

    with DataCollector(
        gym.make("Pendulum-v1", max_episode_steps=5), data_format=source_format
    ) as source:
        collect_episode(source, seed, options)
        source.add_to_dataset(dataset)

    dataset = minari.load_dataset(dataset.id)
    metadata = list(dataset.storage.get_episode_metadata([0, 1]))
    assert [m["id"] for m in metadata] == [0, 1]
    assert [m["total_steps"] for m in metadata] == [5, 5]
    assert metadata[0]["seed"] == 42
    assert metadata[1].get("seed") == seed
    assert metadata[1]["options"] == options

    if seed is not None:
        episode = dataset[1]
        with dataset.recover_environment() as replay:
            obs, _ = replay.reset(
                seed=metadata[1]["seed"], options=metadata[1]["options"]
            )
            np.testing.assert_array_equal(obs, episode.observations[0])
            for i, action in enumerate(episode.actions):
                obs, reward, terminated, truncated, _ = replay.step(action)
                np.testing.assert_array_equal(obs, episode.observations[i + 1])
                assert reward == episode.rewards[i]
                assert terminated == episode.terminations[i]
                assert truncated == episode.truncations[i]


@pytest.mark.parametrize("data_format", get_storage_keys())
def test_combined_episodes_keep_reset_metadata(data_format):
    datasets = []
    for index, seed in enumerate([0, 42]):
        with DataCollector(
            gym.make("Pendulum-v1", max_episode_steps=5), data_format=data_format
        ) as collector:
            collect_episode(collector, seed, None)
            datasets.append(collector.create_dataset(f"pendulum-{index}-v0"))

    combined = minari.combine_datasets(datasets, "pendulum-combined-v0")
    metadata = list(combined.storage.get_episode_metadata([0, 1]))
    assert [m["id"] for m in metadata] == [0, 1]
    assert [m["seed"] for m in metadata] == [0, 42]
    assert all("options" not in m for m in metadata)

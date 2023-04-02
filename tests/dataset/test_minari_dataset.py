import os
from pathlib import Path
from typing import Dict, List, Tuple

import gymnasium
import numpy as np
import pytest

import minari
from minari.dataset.minari_dataset import MinariDataset


@pytest.mark.parametrize("environment_id", ["CartPole-v1", "Blackjack-v1"])
@pytest.mark.parametrize("num_episodes", [1, 10])
def test_compute_stats(tmp_path, environment_id, num_episodes):
    """Test the computation of dataset statistics."""
    dataset_id = "test-dataset-v0"
    dataset, buffers = _generate_local_minari_dataset(
        dataset_id, environment_id, num_episodes, tmp_path
    )

    expected_stats = {}
    expected_stats["actions"] = _compute_expected_stats(buffers, "actions")
    expected_stats["observations"] = _compute_expected_stats(buffers, "observations")
    expected_stats["rewards"] = _compute_expected_stats(buffers, "rewards")

    computed_stats = dataset.compute_stats()

    for key, value in computed_stats.items():
        assert np.array_equal(value["mean"], expected_stats[key]["mean"])
        assert np.array_equal(value["std"], expected_stats[key]["std"])
        assert np.array_equal(value["max"], expected_stats[key]["max"])
        assert np.array_equal(value["min"], expected_stats[key]["min"])
        assert len(value["histogram"]) == len(expected_stats[key]["histogram"])
        for i, histogram in enumerate(value["histogram"]):
            assert np.array_equal(histogram[0], expected_stats[key]["histogram"][i][0])
            assert np.array_equal(histogram[1], expected_stats[key]["histogram"][i][1])


def _compute_expected_stats(buffers: List[Dict[str, np.ndarray]], target_data: str):
    """Compute the stats over the complete data given a list of episode data."""
    data = np.concatenate([buffer[target_data] for buffer in buffers], axis=0)
    if len(data.shape) == 1:
        histograms = [np.histogram(data, bins=20)]
    else:
        num_variables = data.shape[1]
        histograms = [np.histogram(data[:, i], bins=20) for i in range(num_variables)]
    return {
        "mean": np.mean(data, axis=0),
        "std": np.std(data, axis=0),
        "min": np.min(data, axis=0),
        "max": np.max(data, axis=0),
        "histogram": histograms,
    }


def _generate_local_minari_dataset(
    dataset_id: str, environment_id: str, num_episodes: int, dataset_path: Path
) -> Tuple[MinariDataset, List[Dict[str, np.ndarray]]]:
    """Generate a local MinariDataset from a Gymnasium environment.

    Args:
        dataset_id: ID of the newly created dataset.
        environment_id: ID of the Gymnasium environment to use for data generation.
        num_episodeds: The number of episodes to generate data for.
        dataset_path: Path to local folder where to create dataset files.

    Returns:
        A tuple containing the MinariDataset and the data used to create it.
    """
    os.environ["MINARI_DATASETS_PATH"] = dataset_path.as_posix()
    buffers = []
    env = gymnasium.make(environment_id)

    observations = []
    actions = []
    rewards = []
    terminations = []
    truncations = []

    observation, _ = env.reset(seed=42)

    observation, _ = env.reset()
    observations.append(observation)
    for _ in range(num_episodes):
        terminated = False
        truncated = False

        while not terminated and not truncated:
            action = env.action_space.sample()
            observation, reward, terminated, truncated, _ = env.step(action)
            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            terminations.append(terminated)
            truncations.append(truncated)

        episode_buffer = {
            "observations": np.asarray(observations),
            "actions": np.asarray(actions),
            "rewards": np.asarray(rewards),
            "terminations": np.asarray(terminations),
            "truncations": np.asarray(truncations),
        }
        buffers.append(episode_buffer)

        observations.clear()
        actions.clear()
        rewards.clear()
        terminations.clear()
        truncations.clear()

        observation, _ = env.reset()
        observations.append(observation)

    # Create Minari dataset and store locally
    dataset = minari.create_dataset_from_buffers(
        dataset_id=dataset_id,
        env=env,
        buffer=buffers,
        algorithm_name="test_policy",
        code_permalink="https://test.com",
        author="Minari",
        author_email="minari@farama.org",
    )
    return dataset, buffers

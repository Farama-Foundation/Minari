import numpy as np
import pytest

import minari


pytest.importorskip("d3rlpy")
from d3rlpy.datasets import get_minari  # noqa: E402


def test_d3rlpy():
    dataset_name = "D4RL/door/human-v2"
    minari_dataset = minari.load_dataset(dataset_name, download=True)
    dataset, env = get_minari(dataset_name)

    env_spec1, env_spec2 = minari_dataset.recover_environment().spec, env.spec
    assert env_spec1 is not None
    assert env_spec2 is not None
    assert env_spec1.id == env_spec2.id
    assert env_spec1.kwargs == env_spec2.kwargs

    d3rlpy_episode = dataset.episodes[0]
    minari_episode = minari_dataset[0]

    assert len(d3rlpy_episode) == len(minari_episode)

    for step_idx in range(len(d3rlpy_episode)):
        assert env.observation_space.contains(d3rlpy_episode.observations[step_idx])
        assert env.action_space.contains(d3rlpy_episode.actions[step_idx])

        assert np.all(
            d3rlpy_episode.observations[step_idx]
            == minari_episode.observations[step_idx]
        )
        assert np.all(
            d3rlpy_episode.actions[step_idx] == minari_episode.actions[step_idx]
        )
        assert d3rlpy_episode.rewards[step_idx] == minari_episode.rewards[step_idx]

import tempfile

import numpy as np
import pytest
from gymnasium import spaces

from minari.dataset.minari_storage import MinariStorage


@pytest.fixture(autouse=True)
def tmp_dir():
    tmp_dir = tempfile.TemporaryDirectory()
    yield tmp_dir.name
    tmp_dir.cleanup()


def _generate_episode_dict(
    observation_space: spaces.Space, action_space: spaces.Space, length=25
):
    terminations = np.zeros(length, dtype=np.bool_)
    truncations = np.zeros(length, dtype=np.bool_)
    terminated = np.random.randint(2, dtype=np.bool_)
    terminations[-1] = terminated
    truncations[-1] = not terminated

    return {
        "observations": [observation_space.sample() for _ in range(length + 1)],
        "actions": [action_space.sample() for _ in range(length)],
        "rewards": np.random.randn(length),
        "terminations": terminations,
        "truncations": truncations,
    }


def test_non_existing_data(tmp_dir):
    with pytest.raises(ValueError, match="The data path foo doesn't exist"):
        MinariStorage("foo")

    with pytest.raises(ValueError, match="No data found in data path"):
        MinariStorage(tmp_dir)


def test_metadata(tmp_dir):
    action_space = spaces.Box(-1, 1)
    observation_space = spaces.Box(-1, 1)
    storage = MinariStorage.new(
        data_path=tmp_dir,
        observation_space=observation_space,
        action_space=action_space,
    )
    assert storage.data_path == tmp_dir

    extra_metadata = {"float": 3.2, "string": "test-value", "int": 2, "bool": True}
    storage.update_metadata(extra_metadata)

    storage_metadata = storage.metadata
    assert storage_metadata.keys() == {
        "action_space",
        "bool",
        "float",
        "int",
        "observation_space",
        "string",
        "total_episodes",
        "total_steps",
    }

    for key, value in extra_metadata.items():
        assert storage_metadata[key] == value

    storage2 = MinariStorage(tmp_dir)
    assert storage_metadata == storage2.metadata


def test_add_episodes(tmp_dir):
    action_space = spaces.Box(-1, 1, shape=(10,))
    observation_space = spaces.Text(max_length=5)
    n_episodes = 10
    steps_per_episode = 25
    episodes = [
        _generate_episode_dict(
            observation_space, action_space, length=steps_per_episode
        )
        for _ in range(n_episodes)
    ]
    storage = MinariStorage.new(
        data_path=tmp_dir,
        observation_space=observation_space,
        action_space=action_space,
    )
    storage.update_episodes(episodes)
    del storage
    storage = MinariStorage(tmp_dir)

    assert storage.total_episodes == n_episodes
    assert storage.total_steps == n_episodes * steps_per_episode

    for i, ep in enumerate(episodes):
        storage_ep = storage.get_episodes([i])[0]

        assert np.all(ep["observations"] == storage_ep["observations"])
        assert np.all(ep["actions"] == storage_ep["actions"])
        assert np.all(ep["rewards"] == storage_ep["rewards"])
        assert np.all(ep["terminations"] == storage_ep["terminations"])
        assert np.all(ep["truncations"] == storage_ep["truncations"])


def test_append_episode_chunks(tmp_dir):
    action_space = spaces.Discrete(10)
    observation_space = spaces.Text(max_length=5)
    lens = [10, 7, 15]
    chunk1 = _generate_episode_dict(observation_space, action_space, length=lens[0])
    chunk2 = _generate_episode_dict(observation_space, action_space, length=lens[1])
    chunk3 = _generate_episode_dict(observation_space, action_space, length=lens[2])
    chunk1["terminations"][-1] = False
    chunk1["truncations"][-1] = False
    chunk2["terminations"][-1] = False
    chunk2["truncations"][-1] = False
    chunk2["observations"] = chunk2["observations"][:-1]
    chunk3["observations"] = chunk3["observations"][:-1]

    storage = MinariStorage.new(tmp_dir, observation_space, action_space)
    storage.update_episodes([chunk1])
    assert storage.total_episodes == 1
    assert storage.total_steps == lens[0]

    chunk2["id"] = 0
    chunk3["id"] = 0
    storage.update_episodes([chunk2, chunk3])
    assert storage.total_episodes == 1
    assert storage.total_steps == sum(lens)


def test_apply(tmp_dir):
    action_space = spaces.Box(-1, 1, shape=(10,))
    observation_space = spaces.Text(max_length=5)
    n_episodes = 10
    episodes = [
        _generate_episode_dict(observation_space, action_space)
        for _ in range(n_episodes)
    ]
    storage = MinariStorage.new(
        data_path=tmp_dir,
        observation_space=observation_space,
        action_space=action_space,
    )
    storage.update_episodes(episodes)

    def f(ep):
        return ep["actions"].sum()

    episode_indices = [1, 3, 5]
    outs = storage.apply(f, episode_indices=episode_indices)
    assert len(episode_indices) == len(list(outs))
    for i, result in zip(episode_indices, outs):
        assert np.array(episodes[i]["actions"]).sum() == result


def test_episode_metadata(tmp_dir):
    action_space = spaces.Box(-1, 1, shape=(10,))
    observation_space = spaces.Text(max_length=5)
    n_episodes = 10
    episodes = [
        _generate_episode_dict(observation_space, action_space)
        for _ in range(n_episodes)
    ]
    storage = MinariStorage.new(
        data_path=tmp_dir,
        observation_space=observation_space,
        action_space=action_space,
    )
    storage.update_episodes(episodes)

    ep_metadatas = [
        {"foo1-1": True, "foo1-2": 7},
        {"foo2-1": 3.14},
        {"foo3-1": "foo", "foo3-2": 42, "foo3-3": "test"},
    ]

    ep_indices = [1, 4, 5]
    storage.update_episode_metadata(ep_metadatas, episode_indices=ep_indices)

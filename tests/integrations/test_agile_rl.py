import gymnasium as gym
import h5py
import pytest

import minari
from minari import DataCollector
from tests.common import create_dummy_dataset_with_collecter_env_helper


pytest.importorskip("agile_rl")
from agilerl.components.replay_buffer import ReplayBuffer  # noqa: E402
from agilerl.utils.minari_utils import (  # noqa: E402
    minari_to_agile_buffer,
    minari_to_agile_dataset,
)


@pytest.fixture(name="dataset_id")
def dataset_id():
    return "cartpole/test-v0"


@pytest.fixture(autouse=True)
def create_and_destroy_minari_dataset(dataset_id):
    env = gym.make("CartPole-v1")
    env = DataCollector(env, record_infos=True)

    create_dummy_dataset_with_collecter_env_helper(dataset_id, env, num_episodes=10)

    yield

    minari.delete_dataset(dataset_id)


def test_agile_create_dataset(dataset_id):
    """
    Tests that the AgileRL MinariToAgileDataset method works as expected.
    """

    dataset = minari_to_agile_dataset(dataset_id)

    assert isinstance(dataset, h5py.File)
    observations = dataset["observations"]
    next_observations = dataset["next_observations"]
    actions = dataset["actions"]
    rewards = dataset["rewards"]
    terminals = dataset["terminals"]
    assert isinstance(observations, h5py.Dataset)
    assert isinstance(next_observations, h5py.Dataset)
    assert isinstance(actions, h5py.Dataset)
    assert isinstance(rewards, h5py.Dataset)
    assert isinstance(terminals, h5py.Dataset)
    assert observations.size is not None and observations.size > 0
    assert next_observations.size is not None and next_observations.size > 0
    assert actions.size is not None and actions.size > 0
    assert rewards.size is not None and rewards.size > 0
    assert terminals.size is not None and terminals.size > 0


def test_agile_create_buffer(dataset_id):
    """
    Tests that the AgileRL MinariToAgileBuffer method works as expected.
    """

    field_names = ["state", "action", "reward", "next_state", "done"]
    memory = ReplayBuffer(memory_size=10000, field_names=field_names, device="cpu")

    assert memory.counter == 0

    memory = minari_to_agile_buffer(dataset_id, memory)

    assert isinstance(memory, ReplayBuffer)
    assert memory.counter > 0

import gymnasium as gym
import h5py
import pytest
from agilerl.components.replay_buffer import ReplayBuffer
from agilerl.utils.minari_utils import MinariToAgileBuffer, MinariToAgileDataset

import minari
from minari import DataCollector
from tests.common import create_dummy_dataset_with_collecter_env_helper


@pytest.fixture(name="dataset_id")
def dataset_id():
    return "cartpole-test-v0"


@pytest.fixture(autouse=True)
def createAndDestroyMinariDataset(dataset_id):
    env = gym.make("CartPole-v1")
    env = DataCollector(env, record_infos=True, max_buffer_steps=100000)

    create_dummy_dataset_with_collecter_env_helper(dataset_id, env, num_episodes=10)

    yield

    minari.delete_dataset(dataset_id)


def test_agile_create_dataset(dataset_id):
    """
    Tests that the AgileRL MinariToAgileDataset method works as expected.
    """

    dataset = MinariToAgileDataset(dataset_id)

    assert isinstance(dataset, h5py.File)
    assert dataset["observations"].size > 0
    assert dataset["next_observations"].size > 0
    assert dataset["actions"].size > 0
    assert dataset["rewards"].size > 0
    assert dataset["terminals"].size > 0


def test_agile_create_buffer(dataset_id):
    """
    Tests that the AgileRL MinariToAgileBuffer method works as expected.
    """

    field_names = ["state", "action", "reward", "next_state", "done"]
    memory = ReplayBuffer(
        action_dim=2, memory_size=10000, field_names=field_names, device="cpu"
    )

    assert memory.counter == 0

    memory = MinariToAgileBuffer(dataset_id, memory)

    assert isinstance(memory, ReplayBuffer)
    assert memory.counter > 0

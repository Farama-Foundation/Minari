import gymnasium as gym
import h5py
import pytest

import minari
from minari import DataCollector
from tests.common import create_dummy_dataset_with_collecter_env_helper


pytest.importorskip("agilerl")
from agilerl.components.replay_buffer import ReplayBuffer  # noqa: E402
from agilerl.utils.minari_utils import (  # noqa: E402
    MinariToAgileBuffer,
    MinariToAgileDataset,
)


@pytest.fixture(name="dataset_id")
def dataset_id():
    return "cartpole-test-v0"


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

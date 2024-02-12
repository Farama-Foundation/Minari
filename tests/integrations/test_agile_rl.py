import gymnasium as gym
import h5py

from agilerl.components.replay_buffer import ReplayBuffer
from agilerl.utils.minari_utils import MinariToAgileBuffer, MinariToAgileDataset
from minari import DataCollector, delete_dataset
from tqdm import trange

def _create_test_dataset(dataset_id):
    """
    Helper method that creates temporary dataset for use in testing
    """

    env = gym.make('CartPole-v1')
    env = DataCollector(env, record_infos=True, max_buffer_steps=100000)
    
    for _ in trange(1000):
        env.reset()
        terminated, truncated = False, False
        while not terminated and not truncated:
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
    
    env.create_dataset(
        dataset_id=dataset_id,
        algorithm_name="Random-Policy",
        code_permalink="https://github.com/Farama-Foundation/Minari/blob/main/tests/integrations/test_agile_rl.py",
        author="cmcirvin",
        author_email=""
    )
    
    env.close()

def _delete_test_dataset(dataset_id):
    """
    Helper method that deletes temporary testing dataset
    """

    delete_dataset(dataset_id)

def test_agile_create_dataset():

    test_dataset_id = "cartpole-test-v0"
    _create_test_dataset(test_dataset_id)

    dataset = MinariToAgileDataset(test_dataset_id)

    assert isinstance(dataset, h5py.File)
    assert dataset["observations"].size > 0
    assert dataset["next_observations"].size > 0
    assert dataset["actions"].size > 0
    assert dataset["rewards"].size > 0
    assert dataset["terminals"].size > 0

    _delete_test_dataset(test_dataset_id)

def test_agile_create_buffer():
    test_dataset_id = "cartpole-test-v0"
    _create_test_dataset(test_dataset_id)

    field_names = ["state", "action", "reward", "next_state", "done"]
    memory = ReplayBuffer(action_dim=2,
                          memory_size=10000,
                          field_names=field_names,
                          device="cpu")

    assert memory.counter == 0

    memory = MinariToAgileBuffer("cartpole-test-v0", memory)

    assert isinstance(memory, ReplayBuffer)
    assert memory.counter > 0

    _delete_test_dataset(test_dataset_id)

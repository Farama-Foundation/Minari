import pytest
import gymnasium as gym

from minari.integrations.hugging_face import  convert_minari_dataset_to_hugging_face_dataset
from minari import MinariDataset, DataCollectorV0





from tests.common import (
    create_dummy_dataset_with_collecter_env_helper,
    register_dummy_envs
)


register_dummy_envs()

@pytest.mark.parametrize(
    "dataset_id,env_id",
    [
        ("dummy-dict-test-v0", "DummyDictEnv-v0"),
    ],
)
def test_convert_minari_dataset_to_hugging_face_dataset(dataset_id, env_id):
    num_episodes = 10

    env = gym.make(env_id)

    env = DataCollectorV0(env)

    dataset = create_dummy_dataset_with_collecter_env_helper(dataset_id, env, num_episodes = num_episodes)

    print(dataset)
    convert_minari_dataset_to_hugging_face_dataset(dataset)
import gymnasium as gym
import pytest

from minari import DataCollectorV0
from minari.integrations.hugging_face import (
    convert_minari_dataset_to_hugging_face_dataset,
    convert_hugging_face_dataset_to_minari_dataset,
)
from tests.common import (
    create_dummy_dataset_with_collecter_env_helper,
    register_dummy_envs,
)


register_dummy_envs()


@pytest.mark.parametrize(
    "dataset_id,env_id",
    [
        ("dummy-dict-test-v0", "DummyDictEnv-v0"),
    ],
)
def test_convert_minari_dataset_to_hugging_face_dataset_and_back(dataset_id, env_id):
    num_episodes = 10

    env = gym.make(env_id)

    env = DataCollectorV0(env)

    dataset = create_dummy_dataset_with_collecter_env_helper(
        dataset_id, env, num_episodes=num_episodes
    )

    print(dataset)
    hugging_face_dataset = convert_minari_dataset_to_hugging_face_dataset(dataset)
    print(hugging_face_dataset)
    print(hugging_face_dataset["action"])
    print(hugging_face_dataset["id"])

    reconstructed_minari_dataset = convert_hugging_face_dataset_to_minari_dataset(hugging_face_dataset)


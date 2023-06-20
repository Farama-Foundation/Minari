import gymnasium as gym
import pytest

import minari
from minari import DataCollectorV0
from minari.dataset.minari_dataset import MinariDataset
from minari.integrations.hugging_face import (
    convert_hugging_face_dataset_to_minari_dataset,
    convert_minari_dataset_to_hugging_face_dataset,
    pull_dataset_from_hugging_face,
    push_dataset_to_hugging_face,
)
from tests.common import (
    check_data_integrity,
    check_env_recovery,
    check_load_and_delete_dataset,
    create_dummy_dataset_with_collecter_env_helper,
    register_dummy_envs,
)


register_dummy_envs()


@pytest.mark.parametrize(
    "dataset_id,env_id",
    [
        ("cartpole-test-v0", "CartPole-v1"),
        ("dummy-dict-test-v0", "DummyDictEnv-v0"),
        ("dummy-box-test-v0", "DummyBoxEnv-v0"),
        ("dummy-tuple-test-v0", "DummyTupleEnv-v0"),
        ("dummy-combo-test-v0", "DummyComboEnv-v0"),
        ("dummy-tuple-discrete-box-test-v0", "DummyTupleDisceteBoxEnv-v0"),
    ],
)
def test_convert_minari_dataset_to_hugging_face_dataset_and_back(dataset_id, env_id):

    local_datasets = minari.list_local_datasets()
    if dataset_id in local_datasets:
        minari.delete_dataset(dataset_id)

    num_episodes = 10

    env = gym.make(env_id)

    env = DataCollectorV0(env)

    dataset = create_dummy_dataset_with_collecter_env_helper(
        dataset_id, env, num_episodes=num_episodes
    )

    hugging_face_dataset = convert_minari_dataset_to_hugging_face_dataset(dataset)

    minari.delete_dataset(dataset_id)

    reconstructed_minari_dataset = convert_hugging_face_dataset_to_minari_dataset(
        hugging_face_dataset
    )

    assert isinstance(reconstructed_minari_dataset, MinariDataset)
    assert reconstructed_minari_dataset.total_episodes == num_episodes
    assert reconstructed_minari_dataset.spec.total_episodes == num_episodes
    assert len(reconstructed_minari_dataset.episode_indices) == num_episodes

    check_data_integrity(
        reconstructed_minari_dataset._data, reconstructed_minari_dataset.episode_indices
    )
    check_env_recovery(env.env, dataset)

    env.close()

    check_load_and_delete_dataset(dataset_id)


@pytest.mark.skip(
    reason="relies on a private repo, just using this for testing while developing"
)
@pytest.mark.parametrize(
    "dataset_id,env_id",
    [
        ("dummy-combo-test-v0", "DummyComboEnv-v0"),
    ],
)
def test_hugging_face_push_and_pull_dataset(dataset_id, env_id):

    local_datasets = minari.list_local_datasets()
    if dataset_id in local_datasets:
        minari.delete_dataset(dataset_id)

    num_episodes = 10

    env = gym.make(env_id)

    env = DataCollectorV0(env)

    dataset = create_dummy_dataset_with_collecter_env_helper(
        dataset_id, env, num_episodes=num_episodes
    )

    hugging_face_dataset = convert_minari_dataset_to_hugging_face_dataset(dataset)
    print("start")
    print(hugging_face_dataset.info)

    push_dataset_to_hugging_face(hugging_face_dataset, "balisujohn/minari_test")
    minari.delete_dataset(dataset_id)
    recovered_hugging_face_dataset = pull_dataset_from_hugging_face(
        "balisujohn/minari_test"
    )
    print(type(recovered_hugging_face_dataset))
    reconstructed_minari_dataset = convert_hugging_face_dataset_to_minari_dataset(
        recovered_hugging_face_dataset
    )
    assert isinstance(reconstructed_minari_dataset, MinariDataset)
    assert reconstructed_minari_dataset.total_episodes == num_episodes
    assert reconstructed_minari_dataset.spec.total_episodes == num_episodes
    assert len(reconstructed_minari_dataset.episode_indices) == num_episodes

    check_data_integrity(
        reconstructed_minari_dataset._data, reconstructed_minari_dataset.episode_indices
    )
    check_env_recovery(env.env, dataset)

    env.close()

    check_load_and_delete_dataset(dataset_id)

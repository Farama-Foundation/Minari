import gymnasium as gym
import numpy as np
import pytest

from minari import DataCollector, EpisodeData, MinariDataset, StepDataCallback
from minari.dataset._storages import get_storage_keys
from minari.dataset.minari_dataset import parse_dataset_id
from minari.namespace import get_namespace_metadata, list_local_namespaces
from tests.common import (
    check_infos_equal,
    check_load_and_delete_dataset,
    dummy_test_datasets,
    get_info_at_step_index,
)


MAX_UINT64 = np.iinfo(np.uint64).max


class ForceTruncateStepDataCallback(StepDataCallback):
    episode_steps = 10

    def __init__(self) -> None:
        super().__init__()
        self.time_steps = 0

    def __call__(self, env, **kwargs):
        step_data = super().__call__(env, **kwargs)
        if self.time_steps != 0:
            step_data["termination"] = False
            if self.time_steps % self.episode_steps == 0:
                step_data["truncation"] = True

        self.time_steps += 1
        return step_data


def _get_step_from_dictionary_space(episode_data, step_index):
    step_data = {}
    assert isinstance(episode_data, dict)
    for key, value in episode_data.items():
        if isinstance(value, dict):
            step_data[key] = _get_step_from_dictionary_space(value, step_index)
        elif isinstance(value, tuple):
            step_data[key] = _get_step_from_tuple_space(value, step_index)
        else:
            step_data[key] = value[step_index]
    return step_data


def _get_step_from_tuple_space(episode_data, step_index):
    step_data = []
    assert isinstance(episode_data, tuple)
    for element in episode_data:
        if isinstance(element, dict):
            step_data.append(_get_step_from_dictionary_space(element, step_index))
        elif isinstance(element, tuple):
            step_data.append(_get_step_from_tuple_space(element, step_index))
        else:
            step_data.append(element[step_index])

    return tuple(step_data)


def get_single_step_from_episode(episode: EpisodeData, index: int) -> EpisodeData:
    """Get a single step EpisodeData from a full episode."""
    if isinstance(episode.observations, dict):
        observation = _get_step_from_dictionary_space(episode.observations, index)
    elif isinstance(episode.observations, tuple):
        observation = _get_step_from_tuple_space(episode.observations, index)
    else:
        observation = episode.observations[index]
    if isinstance(episode.actions, dict):
        action = _get_step_from_dictionary_space(episode.actions, index)
    elif isinstance(episode.actions, tuple):
        action = _get_step_from_tuple_space(episode.actions, index)
    else:
        action = episode.actions[index]

    infos = get_info_at_step_index(episode.infos, index)

    step_data = {
        "id": episode.id,
        "observations": observation,
        "actions": action,
        "rewards": episode.rewards[index],
        "terminations": episode.terminations[index],
        "truncations": episode.truncations[index],
        "infos": infos,
    }

    return EpisodeData(**step_data)


@pytest.mark.parametrize("data_format", get_storage_keys())
@pytest.mark.parametrize("dataset_id,env_id", dummy_test_datasets)
def test_truncation_without_reset(dataset_id, env_id, data_format, register_dummy_envs):
    """Test new episode creation when environment is truncated and env.reset is not called."""
    num_steps = 50
    num_episodes = int(num_steps / ForceTruncateStepDataCallback.episode_steps)
    env = gym.make(env_id, max_episode_steps=50)
    env = DataCollector(
        env,
        step_data_callback=ForceTruncateStepDataCallback,
        record_infos=True,
        data_format=data_format,
    )

    env.reset()
    for _ in range(num_steps):
        env.step(env.action_space.sample())

    dataset = env.create_dataset(
        dataset_id=dataset_id,
        algorithm_name="random_policy",
        author="Farama",
        author_email="farama@farama.org",
        code_permalink=str(__file__),
        description="Test truncation without reset.",
    )

    env.close()

    assert isinstance(dataset, MinariDataset)
    assert dataset.total_episodes == num_episodes
    assert dataset.spec.total_episodes == num_episodes
    assert len(dataset.episode_indices) == num_episodes

    assert dataset.spec.namespace == parse_dataset_id(dataset_id)[0]

    episodes_generator = dataset.iterate_episodes()
    last_step = get_single_step_from_episode(next(episodes_generator), -1)
    for episode in episodes_generator:
        assert len(episode) == ForceTruncateStepDataCallback.episode_steps
        first_step = get_single_step_from_episode(episode, 0)
        # Check that the last observation of the previous episode is carried over to the next episode
        # as the reset observation.
        if isinstance(first_step.observations, dict) or isinstance(
            first_step.observations, tuple
        ):
            assert first_step.observations == last_step.observations
        else:
            assert np.array_equal(first_step.observations, last_step.observations)

        check_infos_equal(last_step.infos, first_step.infos)
        last_step = get_single_step_from_episode(episode, -1)
        assert bool(last_step.truncations) is True

    if "/" in dataset_id:
        namespace, _, _ = parse_dataset_id(dataset_id)
        assert namespace in list_local_namespaces()
        assert get_namespace_metadata(namespace) == {}

    # check load and delete local dataset
    check_load_and_delete_dataset(dataset_id)


@pytest.mark.parametrize("data_format", get_storage_keys())
@pytest.mark.parametrize("seed", [None, 0, 42, MAX_UINT64])
@pytest.mark.parametrize("options", [None, {"max_timesteps": 3}])
def test_reproducibility(seed, data_format, options, register_dummy_envs):
    """Test episodes are reproducible, even if an explicit reset seed is not set."""
    dataset_id = "dummy-box/test-v0"
    env_id = "DummyBoxEnv-v0"
    num_episodes = 5

    env = DataCollector(gym.make(env_id), data_format=data_format)

    for _ in range(num_episodes):
        env.reset(seed=seed, options=options)

        trunc = False
        term = False

        while not (trunc or term):
            _, _, trunc, term, _ = env.step(env.action_space.sample())

    dataset = env.create_dataset(
        dataset_id=dataset_id,
        algorithm_name="random_policy",
        author="Farama",
        author_email="farama@farama.org",
        code_permalink=str(__file__),
        description="Test reproducibility",
    )
    env.close()

    # Step through the env again using the stored seed and check it matches
    env = dataset.recover_environment()

    assert len(dataset) == num_episodes
    episodes = dataset.iterate_episodes()
    metadatas = dataset.storage.get_episode_metadata(range(num_episodes))
    for episode, episode_metadata in zip(episodes, metadatas):
        episode_seed = episode_metadata["seed"]
        assert episode_seed >= 0
        if seed is not None:
            assert seed == episode_seed

        obs, _ = env.reset(
            seed=int(episode_seed), options=episode_metadata.get("options")
        )

        assert np.allclose(obs, episode.observations[0])

        for k in range(len(episode)):
            obs, rew, term, trunc, _ = env.step(episode.actions[k])
            assert np.allclose(obs, episode.observations[k + 1])
            assert rew == episode.rewards[k]
            assert term == episode.terminations[k]
            assert trunc == episode.truncations[k]

    check_load_and_delete_dataset(dataset_id)

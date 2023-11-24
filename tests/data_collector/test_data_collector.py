import gymnasium as gym
import numpy as np
import pytest

from minari import DataCollectorV0, EpisodeData, MinariDataset, StepDataCallback
from tests.common import check_load_and_delete_dataset, register_dummy_envs


register_dummy_envs()


class ForceTruncateStepDataCallback(StepDataCallback):
    episode_steps = 10

    def __init__(self) -> None:
        super().__init__()
        self.time_steps = 0

    def __call__(self, env, **kwargs):
        step_data = super().__call__(env, **kwargs)

        step_data["terminations"] = False
        if self.time_steps % self.episode_steps == 0:
            step_data["truncations"] = True

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

    step_data = {
        "id": episode.id,
        "total_timesteps": 1,
        "seed": None,
        "observations": observation,
        "actions": action,
        "rewards": episode.rewards[index],
        "terminations": episode.terminations[index],
        "truncations": episode.truncations[index],
    }

    return EpisodeData(**step_data)


@pytest.mark.parametrize(
    "dataset_id,env_id",
    [
        ("dummy-dict-test-v0", "DummyDictEnv-v0"),
        ("dummy-box-test-v0", "DummyBoxEnv-v0"),
        ("dummy-tuple-test-v0", "DummyTupleEnv-v0"),
        ("dummy-combo-test-v0", "DummyComboEnv-v0"),
        ("dummy-tuple-discrete-box-test-v0", "DummyTupleDiscreteBoxEnv-v0"),
    ],
)
def test_truncation_without_reset(dataset_id, env_id):
    """Test new episode creation when environment is truncated and env.reset is not called."""
    num_steps = 50
    num_episodes = int(num_steps / ForceTruncateStepDataCallback.episode_steps)
    env = gym.make(env_id, max_episode_steps=50)
    env = DataCollectorV0(
        env,
        step_data_callback=ForceTruncateStepDataCallback,
    )

    env.reset()

    for _ in range(num_steps):
        env.step(env.action_space.sample())

    dataset = env.create_dataset(
        dataset_id=dataset_id,
        algorithm_name="random_policy",
        author="Farama",
        author_email="farama@farama.org",
    )

    env.close()

    assert isinstance(dataset, MinariDataset)
    assert dataset.total_episodes == num_episodes
    assert dataset.spec.total_episodes == num_episodes
    assert len(dataset.episode_indices) == num_episodes

    episodes_generator = dataset.iterate_episodes()
    last_step = None
    for episode in episodes_generator:
        assert episode.total_timesteps == ForceTruncateStepDataCallback.episode_steps
        if last_step is not None:
            first_step = get_single_step_from_episode(episode, 0)
            # Check that the last observation of the previous episode is carried over to the next episode
            # as the reset observation.
            if isinstance(first_step.observations, dict) or isinstance(
                first_step.observations, tuple
            ):
                assert first_step.observations == last_step.observations
            else:
                assert np.array_equal(first_step.observations, last_step.observations)
        last_step = get_single_step_from_episode(episode, -1)
        assert bool(last_step.truncations) is True

    # check load and delete local dataset
    check_load_and_delete_dataset(dataset_id)

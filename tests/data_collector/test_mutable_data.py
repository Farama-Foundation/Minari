"""Collection must preserve transitions when callers reuse mutable buffers."""

import gymnasium as gym
import numpy as np
import pytest

from minari import DataCollector, StepDataCallback, load_dataset
from minari.dataset._storages import get_storage_keys


def _wrap(array, nested):
    return {"state": (array,)} if nested else array


def _unwrap(value, nested):
    return value["state"][0] if nested else value


class ReusedBufferEnv(gym.Env):
    def __init__(self, nested=False, ending="terminated", reuse=True):
        self.nested = nested
        self.ending = ending
        self.reuse = reuse
        box = gym.spaces.Box(0, 100, shape=(1,), dtype=np.float32)
        self.observation_space = (
            gym.spaces.Dict({"state": gym.spaces.Tuple((box,))}) if nested else box
        )
        self.action_space = self.observation_space
        self.state = np.zeros(2, dtype=np.float32)
        # Both the observation and the recorded info are views of a reused array.
        self.observation = _wrap(self.state[:1], nested)
        self.info = {"position": self.state[1:]}

    def _outputs(self):
        if self.reuse:
            return self.observation, self.info
        return _wrap(self.state[:1].copy(), self.nested), {
            "position": self.state[1:].copy()
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.state[:] = 0
        return self._outputs()

    def step(self, action):
        self.steps += 1
        self.state += _unwrap(action, self.nested)
        observation, info = self._outputs()
        boundary = self.steps % 3 == 0
        return (
            observation,
            float(self.state[0]),
            boundary and self.ending == "terminated",
            boundary and self.ending in ("truncated", "continue"),
            info,
        )


def _save_and_reload(collector):
    collector.create_dataset("mutable-data-v0", ref_min_score=0, ref_max_score=100)
    collector.close()
    return load_dataset("mutable-data-v0")


@pytest.mark.parametrize("data_format", get_storage_keys())
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("ending", ["terminated", "truncated", "reset", "continue"])
@pytest.mark.parametrize("reuse", [False, True])
def test_reused_transition_buffers(data_format, nested, ending, reuse):
    env = ReusedBufferEnv(nested=nested, ending=ending, reuse=reuse)
    collector = DataCollector(env, data_format=data_format, record_infos=True)
    observation, info = collector.reset(seed=0)
    if reuse:
        assert observation is env.observation
        assert info is env.info
    action_array = np.zeros(1, dtype=np.float32)
    action = _wrap(action_array, nested)

    for episode_index in range(2):
        for value in (1, 2, 3):
            action_array[:] = value
            collector.step(action if reuse else _wrap(action_array.copy(), nested))
        if ending != "continue":
            collector.reset(seed=episode_index + 1)

    dataset = _save_and_reload(collector)
    assert dataset.total_episodes == 2
    assert dataset.total_steps == 6
    for index, episode in enumerate(dataset):
        offset = 6 * index if ending == "continue" else 0
        states = np.asarray([0, 1, 3, 6], dtype=np.float32) + offset
        observations = _unwrap(episode.observations, nested)
        actions = _unwrap(episode.actions, nested)
        np.testing.assert_array_equal(observations[:, 0], states)
        np.testing.assert_array_equal(actions[:, 0], [1, 2, 3])
        np.testing.assert_array_equal(episode.infos["position"][:, 0], states)
        np.testing.assert_array_equal(episode.rewards, states[1:])
        np.testing.assert_array_equal(
            episode.terminations, [False, False, ending == "terminated"]
        )
        np.testing.assert_array_equal(
            episode.truncations, [False, False, ending != "terminated"]
        )
        assert observations.dtype == actions.dtype == np.float32


class ReusedCallback(StepDataCallback):
    def __init__(self):
        self.observation = np.zeros(1, dtype=np.float32)
        self.info = {"position": np.zeros(1, dtype=np.float32)}

    def __call__(self, env, **kwargs):
        data = super().__call__(env, **kwargs)
        self.observation[:] = data["observation"] + 10
        self.info["position"][:] = data["observation"] + 20
        data["observation"] = self.observation
        data["info"] = self.info
        return data


@pytest.mark.parametrize("data_format", get_storage_keys())
def test_reused_callback_outputs(data_format):
    collector = DataCollector(
        ReusedBufferEnv(reuse=False),
        data_format=data_format,
        record_infos=True,
        step_data_callback=ReusedCallback,
    )
    collector.reset(seed=0)
    for value in (1, 2, 3):
        collector.step(np.asarray([value], dtype=np.float32))
    episode = _save_and_reload(collector)[0]
    np.testing.assert_array_equal(episode.observations[:, 0], [10, 11, 13, 16])
    np.testing.assert_array_equal(episode.infos["position"][:, 0], [20, 21, 23, 26])


class UncopyableInfo:
    def __deepcopy__(self, memo):
        raise AssertionError("Unrecorded info must not be copied")


class UnrecordedInfoEnv(ReusedBufferEnv):
    def _outputs(self):
        observation, _ = super()._outputs()
        return observation, {"unrecorded": UncopyableInfo()}


@pytest.mark.parametrize("data_format", get_storage_keys())
def test_unrecorded_info_is_not_copied(data_format):
    collector = DataCollector(UnrecordedInfoEnv(reuse=False), data_format=data_format)
    collector.reset(seed=0)
    for value in (1, 2, 3):
        collector.step(np.asarray([value], dtype=np.float32))
    episode = _save_and_reload(collector)[0]
    np.testing.assert_array_equal(episode.observations[:, 0], [0, 1, 3, 6])
    assert episode.infos in (None, {})


@pytest.mark.parametrize("data_format", get_storage_keys())
def test_mutating_returned_values_does_not_change_collected_data(data_format):
    collector = DataCollector(
        ReusedBufferEnv(reuse=False), data_format=data_format, record_infos=True
    )
    observation, info = collector.reset(seed=0)
    observation[:] = 99
    info["position"][:] = 99
    for value in (1, 2, 3):
        observation, _, _, _, info = collector.step(
            np.asarray([value], dtype=np.float32)
        )
        observation[:] = 99
        info["position"][:] = 99
    episode = _save_and_reload(collector)[0]
    np.testing.assert_array_equal(episode.observations[:, 0], [0, 1, 3, 6])
    np.testing.assert_array_equal(episode.infos["position"][:, 0], [0, 1, 3, 6])

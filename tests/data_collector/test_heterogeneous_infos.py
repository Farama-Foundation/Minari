"""Infos whose keys change between steps are stored padded with NaN."""

import gymnasium as gym
import numpy as np
import pytest

from minari import DataCollector, MinariDataset, StepData
from minari.data_collector import EpisodeBuffer
from minari.dataset._storages import get_storage_keys


NAN = np.nan
EPISODE_STEPS = 3


class ChangingInfoEnv(gym.Env):
    """Report different info keys at reset, at every step and at the last step."""

    observation_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
    action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.timestep = 0
        info = {
            "x_position": 0.0,
            "initial_qpos": np.array([1, 2], dtype=np.int32),
            "is_initial": True,
        }
        return self.observation_space.sample(), info

    def step(self, action):
        self.timestep += 1
        info = {
            "x_position": float(self.timestep),
            "x_velocity": np.float32(0.5 * self.timestep),
            "contact": {"force": np.full((2, 2), self.timestep, dtype=np.float32)},
        }
        terminated = self.timestep == EPISODE_STEPS
        info["is_success"] = np.bool_(terminated)
        if terminated:
            info["episode"] = {"l": self.timestep, "success": True}
        return self.observation_space.sample(), 1.0, terminated, False, info


def _step_data(info) -> StepData:
    return {
        "observation": np.zeros(1, dtype=np.float32),
        "action": 0,
        "reward": 0.0,
        "terminated": False,
        "truncated": False,
        "info": info,
    }


@pytest.mark.parametrize("data_format", get_storage_keys())
def test_collect_infos_with_changing_keys(data_format):
    """Every info array keeps one entry per observation, NaN where not reported."""
    with DataCollector(
        ChangingInfoEnv(), record_infos=True, data_format=data_format
    ) as collector:
        collector.reset(seed=0)
        for _ in range(EPISODE_STEPS):
            collector.step(0)
        # The second episode is cut short by `reset`, before `episode` is reported.
        collector.reset(seed=1)
        collector.step(0)
        collector.reset(seed=2)
        dataset = collector.create_dataset("changing-infos/test-v0")

    dataset = MinariDataset(dataset.storage.data_path)
    assert dataset.total_episodes == 2
    complete, cut_short = dataset.iterate_episodes()

    force = np.stack([np.full((2, 2), step) for step in (NAN, 1, 2, 3)])
    np.testing.assert_equal(
        complete.infos,
        {
            "x_position": np.array([0.0, 1.0, 2.0, 3.0]),
            "initial_qpos": np.array([[1, 2], [NAN, NAN], [NAN, NAN], [NAN, NAN]]),
            "is_initial": np.array([1, NAN, NAN, NAN]),
            "x_velocity": np.array([NAN, 0.5, 1.0, 1.5]),
            "contact": {"force": force},
            "is_success": np.array([NAN, 0, 0, 1]),
            "episode": {
                "l": np.array([NAN, NAN, NAN, 3]),
                "success": np.array([NAN, NAN, NAN, 1]),
            },
        },
    )
    # Floating-point infos keep their precision, other numeric types become float64.
    assert complete.infos["x_velocity"].dtype == np.float32
    assert complete.infos["contact"]["force"].dtype == np.float32
    assert complete.infos["initial_qpos"].dtype == np.float64
    assert complete.infos["episode"]["l"].dtype == np.float64

    np.testing.assert_equal(
        cut_short.infos,
        {
            "x_position": np.array([0.0, 1.0]),
            "initial_qpos": np.array([[1, 2], [NAN, NAN]]),
            "is_initial": np.array([1, NAN]),
            "x_velocity": np.array([NAN, 0.5]),
            "contact": {"force": force[:2]},
            "is_success": np.array([NAN, 0]),
        },
    )


@pytest.mark.parametrize("data_format", get_storage_keys())
def test_collect_infos_with_episode_statistics(data_format):
    """`RecordEpisodeStatistics` only adds its info at the end of an episode."""
    env = gym.wrappers.RecordEpisodeStatistics(gym.make("CartPole-v1"))
    with DataCollector(env, record_infos=True, data_format=data_format) as collector:
        collector.reset(seed=0)
        collector.action_space.seed(0)
        terminated = truncated = False
        while not (terminated or truncated):
            _, _, terminated, truncated, info = collector.step(
                collector.action_space.sample()
            )
        dataset = collector.create_dataset("cartpole/statistics-infos-v0")

    episode = dataset[0]
    statistics = episode.infos["episode"]
    assert statistics.keys() == info["episode"].keys()
    for key, values in statistics.items():
        assert values.shape == (len(episode) + 1,)
        assert np.isnan(values[:-1]).all()
        assert values[-1] == info["episode"][key]
    assert statistics["r"][-1] == episode.rewards.sum()
    assert statistics["l"][-1] == len(episode)


@pytest.mark.parametrize("data_format", get_storage_keys())
def test_collect_infos_from_mujoco_env(data_format):
    """MuJoCo v5 environments report fewer infos at reset than at each step."""
    pytest.importorskip("mujoco")
    with DataCollector(
        gym.make("Hopper-v5"), record_infos=True, data_format=data_format
    ) as collector:
        _, reset_info = collector.reset(seed=0)
        collector.action_space.seed(0)
        step_infos = [
            collector.step(collector.action_space.sample())[4] for _ in range(3)
        ]
        dataset = collector.create_dataset("hopper/infos-v0")

    assert reset_info.keys() < step_infos[0].keys()
    infos = dataset[0].infos
    assert infos.keys() == step_infos[0].keys()
    for key, values in infos.items():
        expected = [reset_info.get(key, NAN)] + [info[key] for info in step_infos]
        np.testing.assert_array_equal(values, expected)


def test_pad_infos_without_reset_info():
    """A key reported late is padded to the length of the infos already stored."""
    buffer = EpisodeBuffer(observations=np.zeros(1, dtype=np.float32))
    buffer = buffer.add_step_data(_step_data({"a": 1.0}))
    buffer = buffer.add_step_data(_step_data({"a": 2.0, "b": (3.0, np.ones(2))}))
    buffer = buffer.add_step_data(_step_data({}))

    np.testing.assert_equal(
        buffer.infos,
        {
            "a": [1.0, 2.0, NAN],
            "b": ([NAN, 3.0, NAN], [np.full(2, NAN), np.ones(2), np.full(2, NAN)]),
        },
    )


def test_pad_infos_rejects_non_numeric_values():
    buffer = EpisodeBuffer(
        observations=np.zeros(1, dtype=np.float32), infos={"task": "reach"}
    )
    with pytest.raises(ValueError, match="'task'.*StepDataCallback"):
        buffer.add_step_data(_step_data({}))

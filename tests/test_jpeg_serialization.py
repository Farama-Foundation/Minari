"""
import pathlib
import tempfile
import numpy as np
import gymnasium as gym
import pytest
from PIL import Image
import io
import json

from minari.data_collector.episode_buffer import EpisodeBuffer
from gymnasium import spaces

class TestEnv:
    def __init__(self):
        self.observation_space = spaces.Box(low=0, high=255, shape=(4, 84, 84), dtype=np.uint8)
        self.action_space = spaces.Discrete(2)

    def reset(self):
        obs = np.random.randint(0, 256, (4, 84, 84), dtype=np.uint8)
        return obs, {}

    def step(self, action):
        obs = np.random.randint(0, 256, (4, 84, 84), dtype=np.uint8)
        reward = 1.0
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

def generate_episode_buffer_from_env(env: TestEnv, length=3) -> EpisodeBuffer:
    initial_obs , _ = env.reset()
    buffer = EpisodeBuffer(observations=initial_obs)
    for i in range(length):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        step_data = {
            "observation": obs,
            "action": action,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info
        }
        buffer = buffer.add_step_data(step_data)
    return buffer

def test_arrow_storage_serialization():
    env = TestEnv()
    episode = generate_episode_buffer_from_env(env, length=3)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = pathlib.Path(tmpdir)

        METADATA_FILE_NAME = "metadata.json"
        default_metadata = {
            "total_steps": 0,
            "total_episodes": 0,
            "data_format": "arrow",
            "observation_space": str(env.observation_space),
            "action_space": str(env.action_space)
        }
        metadata_path = tmp_path.joinpath(METADATA_FILE_NAME)
        metadata_path.write_text(json.dumps(default_metadata))

        from minari.dataset._storages.arrow_storage import ArrowStorage
        storage = ArrowStorage(tmp_path, env.observation_space, env.action_space)
        storage.update_episodes([episode])
        loaded_episode = list(storage.get_episodes([0]))
        loaded_obs = loaded_episode[0]["observations"]
        np.testing.assert_array_equal(episode.observations, loaded_obs)

def test_hdf5_serialization():
    env = TestEnv()
    episode = generate_episode_buffer_from_env(env, length=3)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = pathlib.Path(tmpdir)

        METADATA_FILE_NAME = "metadata.json"
        default_metadata = {
            "total_steps": 0,
            "total_episodes": 0,
            "data_format": "hdf5",
            "observation_space": str(env.observation_space),
            "action_space": str(env.action_space)
        }
        metadata_path = tmp_path.joinpath(METADATA_FILE_NAME)
        metadata_path.write_text(json.dumps(default_metadata))

        from minari.dataset._storages.hdf5_storage import HDF5Storage
        storage = HDF5Storage._create(tmp_path, env.observation_space, env.action_space)
        storage.update_episodes([episode])
        loaded_episodes = list(storage.get_episodes([0]))
        loaded_obs = loaded_episodes[0]["observations"]
        np.testing.assert_array_equal(episode.observations, loaded_obs)
"""
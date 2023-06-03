import os
import shutil

import gymnasium as gym
import h5py
import numpy as np
import pytest
from gymnasium import spaces
from gymnasium.envs.registration import register

from minari.dataset.minari_storage import MinariStorage


file_path = os.path.join(os.path.expanduser("~"), ".minari", "datasets")


def _create_dummy_dataset(file_path):

    os.makedirs(file_path, exist_ok=True)
    
    with h5py.File(os.path.join(file_path,'dummy-test-v0.hdf5'), 'w') as f:

        f.attrs['flatten_observation'] = False
        f.attrs['flatten_action'] = False
        f.attrs['env_spec'] = r"""{"id": "DummyEnv-v0", "entry_point": "dummymodule:dummyenv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": 300, "order_enforce": true, "autoreset": false, "disable_env_checker": false, "apply_api_compatibility": false, "kwargs": {"reward_type": "dense"}, "additional_wrappers": [], "vector_entry_point": null}"""
        f.attrs['total_episodes'] = 100
        f.attrs['total_steps'] = 1000
        f.attrs['dataset_id'] = 'dummy-test-v0'


def test_minari_storage_missing_env_module():
    
    file_path = os.path.join(os.path.expanduser("~"), ".minari", "datasets")
    
    _create_dummy_dataset(file_path)
    
    with pytest.raises(ModuleNotFoundError, match='Install dummymodule for loading DummyEnv-v0 data'):
        MinariStorage(os.path.join(file_path, "dummy-test-v0.hdf5"))

    os.remove(os.path.join(file_path,'dummy-test-v0.hdf5'))
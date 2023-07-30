import copy
from collections import OrderedDict
from typing import Dict
import datetime
import random
from operator import itemgetter 

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
import pickle

import minari
from minari import DataCollectorV0, MinariDataset
from tests.common import (
    register_dummy_envs,
)


NUM_EPISODES = 10000
EPISODE_SAMPLE_COUNT = 10

register_dummy_envs()


def test_generate_dataset_with_collector_env(dataset_id, env_id):
    """Test DataCollectorV0 wrapper and Minari dataset creation."""
    # dataset_id = "cartpole-test-v0"
    # delete the test dataset if it already exists
    local_datasets = minari.list_local_datasets()
    if dataset_id in local_datasets:
        minari.delete_dataset(dataset_id)

    env = gym.make(env_id)

    env = DataCollectorV0(env)

    # Step the environment, DataCollectorV0 wrapper will do the data collection job
    env.reset(seed=42)

    for episode in range(NUM_EPISODES):
        terminated = False
        truncated = False
        while not terminated and not truncated:
            action = env.action_space.sample()  # User-defined policy function
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                assert not env._buffer[-1]
            else:
                assert env._buffer[-1]

        env.reset()

    # Create Minari dataset and store locally
    dataset = minari.create_dataset_from_collector_env(
        dataset_id=dataset_id,
        collector_env=env,
        algorithm_name="random_policy",
        code_permalink="https://github.com/Farama-Foundation/Minari/blob/f095bfe07f8dc6642082599e07779ec1dd9b2667/tutorials/LocalStorage/local_storage.py",
        author="WillDudley",
        author_email="wdudley@farama.org",
    )



def test_generate_dataset_with_external_buffer(dataset_id, env_id):
    """Test create dataset from external buffers without using DataCollectorV0."""
    buffer = []
    # dataset_id = "cartpole-test-v0"


    env = gym.make(env_id)

    observations = []
    actions = []
    rewards = []
    terminations = []
    truncations = []


    observation, info = env.reset(seed=42)

    # Step the environment, DataCollectorV0 wrapper will do the data collection job
    observation, _ = env.reset()
    observations.append(observation)
    for episode in range(NUM_EPISODES):
        terminated = False
        truncated = False

        while not terminated and not truncated:
            action = env.action_space.sample()  # User-defined policy function
            observation, reward, terminated, truncated, _ = env.step(action)
            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            terminations.append(terminated)
            truncations.append(truncated)

        episode_buffer = {
            "observations": copy.deepcopy(observations),
            "actions": copy.deepcopy(actions),
            "rewards": np.asarray(rewards),
            "terminations": np.asarray(terminations),
            "truncations": np.asarray(truncations),
        }
        buffer.append(episode_buffer)

        observations.clear()
        actions.clear()
        rewards.clear()
        terminations.clear()
        truncations.clear()

        observation, _ = env.reset()
        observations.append(observation)

    # Create Minari dataset and store locally
    dataset = minari.create_dataset_from_buffers(
        dataset_id=dataset_id,
        env=env,
        buffer=buffer,
        algorithm_name="random_policy",
        code_permalink="https://github.com/Farama-Foundation/Minari/blob/f095bfe07f8dc6642082599e07779ec1dd9b2667/tutorials/LocalStorage/local_storage.py",
        author="WillDudley",
        author_email="wdudley@farama.org",
    )



def test_generate_dataset_pickle(dataset_id, env_id):
    """Test create dataset from external buffers without using DataCollectorV0."""
    buffer = []
    # dataset_id = "cartpole-test-v0"


    env = gym.make(env_id)

    observations = []
    actions = []
    rewards = []
    terminations = []
    truncations = []


    observation, info = env.reset(seed=42)

    # Step the environment, DataCollectorV0 wrapper will do the data collection job
    observation, _ = env.reset()
    observations.append(observation)
    for episode in range(NUM_EPISODES):
        terminated = False
        truncated = False

        while not terminated and not truncated:
            action = env.action_space.sample()  # User-defined policy function
            observation, reward, terminated, truncated, _ = env.step(action)
            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            terminations.append(terminated)
            truncations.append(truncated)

        episode_buffer = {
            "observations": copy.deepcopy(observations),
            "actions": copy.deepcopy(actions),
            "rewards": np.asarray(rewards),
            "terminations": np.asarray(terminations),
            "truncations": np.asarray(truncations),
        }
        buffer.append(episode_buffer)

        observations.clear()
        actions.clear()
        rewards.clear()
        terminations.clear()
        truncations.clear()

        observation, _ = env.reset()
        observations.append(observation)

    # Create Minari dataset and store locally with pickle
    with open("test.pkl", "wb") as test_file:
        pickle.dump(buffer,test_file)

    #with open("test.pkl", "rb") as test_file:
    #    test = pickle.load(test_file)

    
def test_sample_n_random_episodes_from_minari_dataset(dataset_id):
    dataset = minari.load_dataset(dataset_id)
    episodes = dataset.sample_episodes(EPISODE_SAMPLE_COUNT)
   # print(episodes)

def test_sample_n_random_episodes_from_pickle_dataset():
    with open("test.pkl", "rb") as test_file:
        test = pickle.load(test_file)

    indices = random.sample(range(0,len(test)),EPISODE_SAMPLE_COUNT )

    result = itemgetter(*indices)(test)



def measure(function, args):
    before = datetime.datetime.now()
    function(*args)
    after = datetime.datetime.now()
    return (after-before).total_seconds()    


if __name__ == "__main__":


    environment_list =  [
        ("cartpole-test-v0", "CartPole-v1"),
        ("dummy-dict-test-v0", "DummyDictEnv-v0"),
        ("dummy-tuple-test-v0", "DummyTupleEnv-v0"),
        ("dummy-text-test-v0", "DummyTextEnv-v0"),
        ("dummy-combo-test-v0", "DummyComboEnv-v0"),
        ("dummy-tuple-discrete-box-test-v0", "DummyTupleDiscreteBoxEnv-v0"),
    ]


    measurements = {}



    for dataset_id, env_id in environment_list:

        #dataset_id, env_id = ("cartpole-test-v0", "CartPole-v1")


        # delete the test dataset if it already exists
        local_datasets = minari.list_local_datasets()
        if dataset_id in local_datasets:
            minari.delete_dataset(dataset_id)

        result = measure(test_generate_dataset_with_collector_env, (dataset_id, env_id))
        print(f"Time to generate {NUM_EPISODES} episodes with {env_id} using test_generate_dataset_with_collector_env: {str(result)}")
        
        # delete the test dataset if it already exists
        local_datasets = minari.list_local_datasets()
        if dataset_id in local_datasets:
            minari.delete_dataset(dataset_id)

        
        result = measure(test_generate_dataset_with_external_buffer, (dataset_id, env_id))
        print(f"Time to generate {NUM_EPISODES} episodes with {env_id} using test_generate_dataset_with_external_buffer: {str(result)}")
        

        
        result = measure(test_generate_dataset_pickle, (dataset_id, env_id))
        print(f"Time to generate {NUM_EPISODES} episodes with {env_id} using test_generate_dataset_pickle: {str(result)}")
        
        result = measure(test_sample_n_random_episodes_from_minari_dataset, (dataset_id,))
        print(f"Time to sample {EPISODE_SAMPLE_COUNT} episodes from {env_id} using test_sample_n_random_episodes_from_minari_dataset: {str(result)}")
        
        
        result = measure(test_sample_n_random_episodes_from_pickle_dataset, ())
        print(f"Time to sample {EPISODE_SAMPLE_COUNT} episodes from {env_id} test_sample_n_random_episodes_from_pickle_dataset: {str(result)}")
        
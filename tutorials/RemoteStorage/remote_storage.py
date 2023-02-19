# pyright: basic, reportOptionalMemberAccess=false, reportOptionalSubscript=false

import base64
import json
import os

import gymnasium as gym
import numpy as np
from gymnasium.utils.serialize_spec_stack import serialise_spec_stack

import minari
from minari.dataset import MinariDataset


GCP_DATASET_ADMIN = os.environ["GCP_DATASET_ADMIN"]

credentials_json = base64.b64decode(GCP_DATASET_ADMIN).decode("utf8").replace("'", '"')
with open("credentials.json", "w") as f:
    f.write(credentials_json)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./credentials.json"


def generate_dataset(dataset_name: str):
    num_episodes = 10

    env = gym.make("LunarLander-v2", render_mode="rgb_array")
    environment_stack = serialise_spec_stack(env.spec_stack)
    observation, info = env.reset(seed=42)

    assert env.spec.max_episode_steps is not None, "Max episode steps must be defined"

    replay_buffer = {
        "episode": np.array(
            [[0]] * env.spec.max_episode_steps * num_episodes, dtype=np.int32
        ),
        "observation": np.array(
            [[0] * env.observation_space.shape[0]]
            * env.spec.max_episode_steps
            * num_episodes,
            dtype=np.float32,
        ),
        "action": np.array(
            [[0] * 2] * env.spec.max_episode_steps * num_episodes, dtype=np.float32
        ),
        "reward": np.array(
            [[0]] * env.spec.max_episode_steps * num_episodes, dtype=np.float32
        ),
        "terminated": np.array(
            [[0]] * env.spec.max_episode_steps * num_episodes, dtype=bool
        ),
        "truncated": np.array(
            [[0]] * env.spec.max_episode_steps * num_episodes, dtype=bool
        ),
    }

    total_steps = 0
    for episode in range(num_episodes):
        episode_step = 0
        observation, info = env.reset()
        terminated = False
        truncated = False
        while not terminated and not truncated:
            action = env.action_space.sample()  # User-defined policy function
            observation, reward, terminated, truncated, info = env.step(action)
            replay_buffer["episode"][total_steps] = np.array(episode)
            replay_buffer["observation"][total_steps] = np.array(observation)
            replay_buffer["action"][total_steps] = np.array(action)
            replay_buffer["reward"][total_steps] = np.array(reward)
            replay_buffer["terminated"][total_steps] = np.array(terminated)
            replay_buffer["truncated"][total_steps] = np.array(truncated)
            episode_step += 1
            total_steps += 1

    env.close()

    replay_buffer["episode"] = replay_buffer["episode"][:total_steps]
    replay_buffer["observation"] = replay_buffer["observation"][:total_steps]
    replay_buffer["action"] = replay_buffer["action"][:total_steps]
    replay_buffer["reward"] = replay_buffer["reward"][:total_steps]
    replay_buffer["terminated"] = replay_buffer["terminated"][:total_steps]
    replay_buffer["truncated"] = replay_buffer["truncated"][:total_steps]

    ds = MinariDataset(
        dataset_name=dataset_name,
        algorithm_name="random_policy",
        environment_name="LunarLander-v2",
        environment_stack=json.dumps(environment_stack),
        seed_used=42,  # For the simplicity of this example, we're not actually using a seed.
        code_permalink="https://github.com/Farama-Foundation/Minari/blob/f095bfe07f8dc6642082599e07779ec1dd9b2667/tutorials/LocalStorage/local_storage.py",
        author="WillDudley",
        author_email="wdudley@farama.org",
        observations=replay_buffer["observation"],
        actions=replay_buffer["action"],
        rewards=replay_buffer["reward"],
        terminations=replay_buffer["terminated"],
        truncations=replay_buffer["truncated"],
    )

    print("Dataset generated!")

    return ds


if __name__ == "__main__":
    dataset_name = "LunarLander_v2_remote-test-dataset"

    print("\nGenerate dataset as standard")
    generated_dataset = generate_dataset(dataset_name)

    print("\nSave dataset to local storage")
    generated_dataset.save()

    print(
        "\nUpload dataset to Google Cloud Storage (here naming checks are done, enforcing dataset naming conventions)"
    )
    minari.upload_dataset(dataset_name)

    print("\nList all datasets in remote storage")
    minari.list_remote_datasets()

    print(
        "\nDelete dataset from local storage, list all datasets in local storage to confirm"
    )
    minari.delete_dataset(dataset_name)
    minari.list_local_datasets()

    print("\nDownload dataset from Google Cloud Storage")
    minari.download_dataset(dataset_name)

    print(
        "\nListing datasets in local storage, we should see the dataset we just downloaded"
    )
    minari.list_local_datasets()

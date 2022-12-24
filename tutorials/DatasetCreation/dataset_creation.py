# pyright: basic, reportOptionalMemberAccess=false

import base64
import json
import os

import gymnasium as gym
import numpy as np

import minari
from minari.dataset import MinariDataset

# 1. Get permissions to upload to GCP
GCP_DATASET_ADMIN = os.environ["GCP_DATASET_ADMIN"]

credentials_json = base64.b64decode(GCP_DATASET_ADMIN).decode("utf8").replace("'", '"')
with open("credentials.json", "w") as f:
    f.write(credentials_json)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./credentials.json"


# 2. Standard Gymnasium procedure to collect data into whatever replay buffer you want
env = gym.make("FetchReach-v3")

environment_stack = gym.SpecStack(env).stack_json  # Get the environment specification stack for reproducibility

env.reset()
replay_buffer = {
    "episode": np.array([]),
    "observation": np.array([]),
    "action": np.array([]),
    "reward": np.array([]),
    "terminated": np.array([]),
    "truncated": np.array([]),
}
dataset_name = "FetchReach_v3_example-dataset"

num_episodes = 4


assert env.spec.max_episode_steps is not None, "Max episode steps must be defined"

replay_buffer = {
    "episode": np.array(
        [[0]] * env.spec.max_episode_steps * num_episodes, dtype=np.int32
    ),
    "observation": np.array(
        [[0] * 13] * env.spec.max_episode_steps * num_episodes,
        dtype=np.float32,
    ),
    "action": np.array(
        [[0] * 4] * env.spec.max_episode_steps * num_episodes, dtype=np.float32
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
        replay_buffer["observation"][total_steps] = np.concatenate(
            (
                np.array(observation["observation"]),
                np.array(observation["desired_goal"]),
            )
        )
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


# 3. Convert the replay buffer to a MinariDataset
dataset = MinariDataset(
    dataset_name=dataset_name,
    algorithm_name="random_policy",
    environment_name="FetchReach-v3",
    environment_stack=json.dumps(environment_stack),
    seed_used=42,  # For the simplicity of this example, we're not actually using a seed. Naughty us!
    code_permalink="https://github.com/Farama-Foundation/Kabuki/blob/f095bfe07f8dc6642082599e07779ec1dd9b2667/tutorials/LocalStorage/local_storage.py",
    author="WillDudley",
    author_email="wdudley@farama.org",
    observations=replay_buffer["observation"],
    actions=replay_buffer["action"],
    rewards=replay_buffer["reward"],
    terminations=replay_buffer["terminated"],
    truncations=replay_buffer["truncated"],
)

print("Dataset generated!")

# 4. Save the dataset locally
dataset.save()

# 5. Upload the dataset to GCP
minari.upload_dataset(dataset_name)

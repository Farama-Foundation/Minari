import gymnasium as gym
from kabuki.dataset import KabukiDataset
import numpy as np
import requests
import os
from google.cloud import storage
import kabuki

# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../../../../.config/gcloud/credentials.db"

num_episodes = 2

env = gym.make("LunarLander-v2", render_mode="rgb_array")
observation, info = env.reset(seed=42)

replay_buffer = {
    "episode": np.array([]),
    "observation": np.array([]),
    "action": np.array([]),
    "reward": np.array([]),
    "terminated": np.array([]),
    "truncated": np.array([]),
}

for episode in range(num_episodes):
    observation, info = env.reset()
    terminated = False
    truncated = False
    while not terminated and not truncated:
        action = env.action_space.sample()  # User-defined policy function
        observation, reward, terminated, truncated, info = env.step(action)
        np.append(replay_buffer["episode"], episode)
        np.append(replay_buffer["observation"], observation)
        np.append(replay_buffer["action"], action)
        np.append(replay_buffer["reward"], reward)
        np.append(replay_buffer["terminated"], terminated)
        np.append(replay_buffer["truncated"], truncated)

env.close()

ds = KabukiDataset(
    dataset_name="LunarLander-v2",
    observations=replay_buffer["observation"],
    actions=replay_buffer["action"],
    rewards=replay_buffer["reward"],
    terminations=replay_buffer["terminated"],
    truncations=replay_buffer["truncated"],
)


from google.cloud import storage

ds.dump(
    ".datasets/LunarLander-v2-test_dataset.hdf5"
)  # todo: abstract away parent directory and hdf5 extension

# upload and download datasets
kabuki.upload_dataset("LunarLander-v2-test_dataset")
ds = kabuki.retrieve_dataset("LunarLander-v2-test_dataset")

# list datasets on server
kabuki.list_datasets()

print(ds.dataset_name)
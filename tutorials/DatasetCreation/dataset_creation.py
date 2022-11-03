import base64
import os

import gymnasium as gym
import numpy as np

import kabuki
from kabuki.dataset import KabukiDataset


# 1. Get permissions to upload to GCP
GCP_DATASET_ADMIN = os.environ["GCP_DATASET_ADMIN"]

credentials_json = base64.b64decode(GCP_DATASET_ADMIN).decode("utf8").replace("'", '"')
with open("credentials.json", "w") as f:
    f.write(credentials_json)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./credentials.json"


# 2. Standard Gymnasium procedure to collect data into whatever replay buffer you want
env = gym.make("FetchReach-v3")
env.reset()
replay_buffer = {
    "episode": np.array([]),
    "observation": np.array([]),
    "action": np.array([]),
    "reward": np.array([]),
    "terminated": np.array([]),
    "truncated": np.array([]),
}
dataset_name = "FetchReach-v3-example_dataset"

num_episodes = 4

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

# 3. Convert the replay buffer to a KabukiDataset
dataset = KabukiDataset(
    dataset_name=dataset_name,
    algorithm_name="random_policy",
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
kabuki.upload_dataset(dataset_name)

import gymnasium as gym
from kabuki.dataset import KabukiDataset
import numpy as np
import requests
import os
import base64
import json
from google.cloud import storage
import kabuki

GCP_DATASET_ADMIN = os.environ["GCP_DATASET_ADMIN"]

credentials_json = base64.b64decode(GCP_DATASET_ADMIN).decode("utf8").replace("'", '"')
with open("credentials.json", "w") as f:
    f.write(credentials_json)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./credentials.json"


def generate_dataset(dataset_name: str):
    num_episodes = 10

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
        dataset_name=dataset_name,
        algorithm_name="random_policy",
        observations=replay_buffer["observation"],
        actions=replay_buffer["action"],
        rewards=replay_buffer["reward"],
        terminations=replay_buffer["terminated"],
        truncations=replay_buffer["truncated"],
    )

    print("Dataset generated!")

    return ds


from google.cloud import storage


if __name__ == "__main__":
    dataset_name = "LunarLander-v2-remote_test_dataset"

    print("\nGenerate dataset as standard")
    generated_dataset = generate_dataset(dataset_name)

    print("\nSave dataset to local storage")
    generated_dataset.save()

    print(
        "\nUpload dataset to Google Cloud Storage (here naming checks are done, enforcing dataset naming conventions)"
    )
    kabuki.upload_dataset(dataset_name)

    print("\nList all datasets in remote storage")
    kabuki.list_remote_datasets()

    print(
        "\nDelete dataset from local storage, list all datasets in local storage to confirm"
    )
    kabuki.delete_dataset(dataset_name)
    kabuki.list_local_datasets()

    print("\nDownload dataset from Google Cloud Storage")
    kabuki.download_dataset(dataset_name)

    print(
        "\nListing datasets in local storage, we should see the dataset we just downloaded"
    )
    kabuki.list_local_datasets()

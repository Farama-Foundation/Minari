import gymnasium as gym

import kabuki
from kabuki.dataset import KabukiDataset
import numpy as np


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

    return ds


if __name__ == "__main__":
    dataset_name = "LunarLander-v2-test_dataset"

    print("\nGenerate dataset as standard")
    generated_dataset = generate_dataset(dataset_name)

    print("\nSave dataset to local storage")
    generated_dataset.save()

    print(
        "\nListing datasets in local storage, we should see the dataset we just generated"
    )
    kabuki.list_local_datasets()

    print("\nWe can load the dataset from local storage as follows")
    loaded_dataset = kabuki.load_dataset(dataset_name)

    print("\nWe can delete the dataset from local storage as follows")
    kabuki.delete_dataset(dataset_name)

    print(
        "\nListing datasets in local storage, we should now no longer see the dataset we just generated"
    )
    kabuki.list_local_datasets()

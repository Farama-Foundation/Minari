import gymnasium as gym
from kabuki.dataset import KabukiDataset
import numpy as np

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
    dataset_name="LunarLander-v2-test_dataset",
    observations=replay_buffer["observation"],
    actions=replay_buffer["action"],
    rewards=replay_buffer["reward"],
    terminations=replay_buffer["terminated"],
    truncations=replay_buffer["truncated"],
)

ds.dump("test_dataset.hdf5")

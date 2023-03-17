import gymnasium as gym
import minari
from minari import DataCollectorV0


env = DataCollectorV0(gym.make('LunarLander-v2'), record_infos=True, max_buffer_steps=1000)

total_episodes = 100

for _ in range(total_episodes):
    env.reset()
    while True:
        # random action policy
        action = env.action_space.sample()
        obs, rew, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            break

dataset = minari.create_dataset_from_collector_env(dataset_name="LunarLander-v2-test-v0", collector_env=env)


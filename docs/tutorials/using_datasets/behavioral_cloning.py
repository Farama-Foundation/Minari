# fmt: off
"""
Behavioral cloning with PyTorch
=========================================
"""
# %%%
# We present here how to perform behavioral cloning on a Minari dataset using PyTorch.
# We will start generating the dataset of the expert policy for the `CartPole-v1 <https://gymnasium.farama.org/environments/classic_control/cart_pole/>`_ environment, which is a classic control problem.
# The objective is to balance the pole on the cart, and we receive a reward of +1 for each successful timestep.

# %%
# Policy training
# ~~~~~~~~~~~~~~~~~~~
# To train the expert policy, we use the library rl_zoo3.
# After installing the library with ``pip install rl_zoo3``,
# we train a PPO agent on the environment with the following command:
#
# ``python -m rl_zoo3.train --algo ppo --env CartPole-v1``

# %%
# This will generate a new folder named `log` with the expert policy.

# %%
# Imports
# ~~~~~~~~~~~~~~~~~~~
# Let's import all the required packages and set the random seed for reproducibility:


import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import PPO
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import minari
from minari import DataCollectorV0


torch.manual_seed(42)

# %%
# Dataset generation
# ~~~~~~~~~~~~~~~~~~~
# Now let's generate the dataset using the `DataCollectorV0 <https://minari.farama.org/api/data_collector/>`_ wrapper:
#

env = DataCollectorV0(gym.make('CartPole-v1'))
path = os.path.abspath('') + '/logs/ppo/CartPole-v1_1/best_model'
agent = PPO.load(path)

total_episodes = 1_000
for i in tqdm(range(total_episodes)):
    obs, _ = env.reset(seed=42)
    while True:
        action, _ = agent.predict(obs)
        obs, rew, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

dataset = minari.create_dataset_from_collector_env(dataset_id="CartPole-v1-expert",
                                                   collector_env=env,
                                                   algorithm_name="ExpertPolicy",
                                                   code_permalink="https://minari.farama.org/tutorials/behavioral_cloning",
                                                   author="Farama",
                                                   author_email="contact@farama.org"
                                                   )

# %%
# Once executing the script, the dataset will be saved on your disk. You can display the list of datasets with ``minari list local`` command.

# %%
# Behavioral cloning with PyTorch
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Now we can use PyTorch to learn the policy from the offline dataset.
# Let's define the policy network:


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# %%
# In this scenario, the output dimension will be two, as previously mentioned. As for the input dimension, it will be four, corresponding to the observation space of ``CartPole-v1``.
# Our next step is to load the dataset and set up the training loop. The ``MinariDataset`` is compatible with the PyTorch Dataset API, allowing us to load it directly using PyTorch DataLoader.
# However, since each episode can have a varying length, we need to pad them.
# To achieve this, we can utilize the `collate_fn <https://pytorch.org/docs/stable/data.html#working-with-collate-fn>`_ feature of PyTorch DataLoader. Let's create the ``collate_fn`` function:


def collate_fn(batch):
    return {
        "id": torch.Tensor([x.id for x in batch]),
        "seed": torch.Tensor([x.seed for x in batch]),
        "total_timesteps": torch.Tensor([x.total_timesteps for x in batch]),
        "observations": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.observations) for x in batch],
            batch_first=True
        ),
        "actions": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.actions) for x in batch],
            batch_first=True
        ),
        "rewards": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.rewards) for x in batch],
            batch_first=True
        ),
        "terminations": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.terminations) for x in batch],
            batch_first=True
        ),
        "truncations": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.truncations) for x in batch],
            batch_first=True
        )
    }

# %%
# We can now proceed to load the data and create the training loop.
# To begin, let's initialize the DataLoader, neural network, optimizer, and loss.


minari_dataset = minari.load_dataset("CartPole-v1-expert")
dataloader = DataLoader(minari_dataset, batch_size=256, shuffle=True, collate_fn=collate_fn)

env = minari_dataset.recover_environment()
observation_space = env.observation_space
action_space = env.action_space
assert isinstance(observation_space, spaces.Box)
assert isinstance(action_space, spaces.Discrete)

policy_net = PolicyNetwork(np.prod(observation_space.shape), action_space.n)
optimizer = torch.optim.Adam(policy_net.parameters())
loss_fn = nn.CrossEntropyLoss()

# %%
# We use the cross-entropy loss like a classic classification task, as the action space is discrete.
# We then train the policy to predict the actions:

num_epochs = 32

for epoch in range(num_epochs):
    for batch in dataloader:
        a_pred = policy_net(batch['observations'][:, :-1])
        a_hat = F.one_hot(batch["actions"]).type(torch.float32)
        loss = loss_fn(a_pred, a_hat)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch: {epoch}/{num_epochs}, Loss: {loss.item()}")

# %%
# And now, we can evaluate if the policy learned from the expert!

env = gym.make("CartPole-v1", render_mode="human")
obs, _ = env.reset(seed=42)
done = False
accumulated_rew = 0
while not done:
    action = policy_net(torch.Tensor(obs)).argmax()
    obs, rew, ter, tru, _ = env.step(action.numpy())
    done = ter or tru
    accumulated_rew += rew

env.close()
print("Accumulated rew: ", accumulated_rew)

# %%
# We can visually observe that the learned policy aces this simple control task, and we get the maximum reward 500, as the episode is truncated after 500 steps.
#

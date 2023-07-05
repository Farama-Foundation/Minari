# fmt: off
"""
Offline Q-Learning with PyTorch
=========================================
"""

# %%%
# This tutorial will guide you through the process of utilizing offline data generated with Minari to learn a `Q-value <https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#value-functions>`_ function using PyTorch. 
# The `CartPole-v1 <https://gymnasium.farama.org/environments/classic_control/cart_pole/>`_ environment, which is a classic control problem, will be employed for this demonstration.

# %%
# Dataset generation
# ~~~~~~~~~~~~~~~~~~~~~~
# To begin, we will generate a dataset with a random policy by executing the following script. For a detailed understanding of its functionality, please refer to the documentation of `DataCollectorV0 <https://minari.farama.org/api/data_collector/>`_.

import minari
from minari import DataCollectorV0
import gymnasium as gym

env = DataCollectorV0(gym.make('CartPole-v1'))
env.action_space.seed(42)

total_episodes = 100_000
for i in range(total_episodes):
    obs, _ = env.reset(seed=42)
    while True:
        action = env.action_space.sample()
        obs, rew, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            break

dataset = minari.create_dataset_from_collector_env(dataset_id="CartPole-v1-random", 
                                                   collector_env=env,
                                                   algorithm_name="RandomPolicy",
                                                   code_permalink="https://minari.farama.org/tutorials/q_learning_torch",
                                                   author="Farama",
                                                   author_email="contact@farama.org"
)

# %%%
# Once the script has been executed, the dataset will be saved on your disk. To verify its presence, you can use the command ``minari list local`` in your terminal.
# This command will display a list of local datasets, including the recently generated ``CartPole-v1-random`` dataset.

# %%
# Offline Q-Learning with PyTorch
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Now, we will proceed with using the dataset to train the `Q`-function. 
# Given that the action space in CartPole-v1 is discrete with only two actions, we can leverage this fact to design a neural network that takes the observation as input (excluding the action) and produces two output values, one for each action. 
# To begin, let's import all the necessary libraries and create the neural network that will serve as the `Q`-function:

from gymnasium import spaces
import minari

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

torch.manual_seed(42)

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# %%%
# In this scenario, the output dimension will indeed be two, as previously mentioned. As for the input dimension, it will be four, corresponding to the observation space of ``CartPole-v1``. 
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

# %%%
# We can now proceed to load the data and create the training loop. To begin, let's initialize the DataLoader, neural network, optimizer, and loss.

minari_dataset =  minari.load_dataset("CartPole-v1-random")
dataloader = DataLoader(minari_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

env = minari_dataset.recover_environment()
observation_space = env.observation_space
action_space = env.action_space
assert isinstance(observation_space, spaces.Box)
assert isinstance(action_space, spaces.Discrete)

q_net = QNetwork(np.prod(observation_space.shape), action_space.n)
optimizer = torch.optim.Adam(q_net.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# %%%
# We use the Mean Square Error loss, as a classic regression task. In fact, we want to predict the `Q`-values of each state-action pairs that we take from the trajectories. 
# The training loop will look like this:

num_epochs = 3
gamma = 0.99

for epoch in range(num_epochs):
    for batch in tqdm(dataloader):
        q_values = q_net(batch['observations'])
        q_targets = batch['rewards'].float() + gamma * torch.max(q_values[:, 1:], dim=-1).values
        q_pred = q_values[:, :-1].gather(-1, batch['actions'].unsqueeze(-1)).squeeze()
            
        mask = 1 - torch.cumsum(
            torch.logical_or(batch["terminations"], batch["truncations"]),
            dim=-1
        )
        loss = loss_fn(q_pred * mask, q_targets * mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch: {epoch}/{num_epochs}, Loss: {loss.item()}")

# %%%
# where we compute the ``q_targets`` using the SARSA objective 
# 
# .. math ::
#   r(s_t, a_t) + \gamma \max_{a' \in \mathcal{A}} \hat{Q}_{\theta}(s_{t+1}, a')
#
# Similarly, the loss for a single episode will be: 
#
# .. math ::
#   l(\theta) = \sum_{t=0}^{T-1} \Big( r(s_t, a_t) + \gamma \max_{a' \in \mathcal{A}} \hat{Q}_{\theta}(s_{t+1}, a') - \hat{Q}_{\theta}(s_t, a_t) \Big)^2
# 
# Where `T` is the toal number of step in the episode. However, we have a batch of episodes with different length `T`; we use ``mask`` to zeros the loss after the episode is terminated. 
# 
# Lastly, we can assess the performance of the agent that utilizes the learned `Q`-values:

env = gym.make("CartPole-v1", render_mode="human")
obs, _ = env.reset(seed=42)
done = False
accumulated_rew = 0
while not done:
    action = q_net(torch.Tensor(obs)).argmax()
    obs, rew, ter, tru, _ = env.step(action.numpy())
    done = ter or tru
    accumulated_rew += rew

env.close()
print("Offline Q-Learning accumulated rew: ", accumulated_rew)

# %%%
# We can visually observe that the learned agent acts better than randomly in controlling the pole. 
# In the environment, the reward is +1 for each timestep until the pole falls. With the learned `Q`-values, we achieve an accumulated reward of 247. 
# In contrast, the random policy demonstrates poor performance:

dataset_accumulated = 0
for episode in minari_dataset:
    dataset_accumulated += sum(episode.rewards)
dataset_accumulated /= minari_dataset.total_episodes
print("Dataset mean accumulated rew: ", dataset_accumulated)

# %%%
# as it gets an accumulated reward mean of around 22.
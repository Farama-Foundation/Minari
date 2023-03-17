---
title: Basic Usage
---

# Basic Usage

Minari is a standard dataset hosting interface for Offline Reinforcement Learning applications. Minari is compatible with most of the RL environments that follow the Gymnasium API and facilitates Offline RL dataset handling by providing data collection, dataset hosting, and dataset sampling capabilities.

## Create Minari Dataset

### Collecting Data

Minari can abstract the data collection process. This is achieved by using the [`DataCollectorV0`](/api/data_collector) wrapper which stores the environments stepping data in internal memory buffers before saving the dataset into disk. The [`DataCollectorV0`](/api/data_collector) wrapper can also perform caching by scheduling the amount of episodes or steps that are stored in-memory before saving the data in a temporary [Minari dataset file](/content/dataset_format). This wrapper also computes relevant metadata of the dataset while collecting the data. 

The wrapper is very simple to initialize:

```python
from minari import DataCollectorV0
import gymnasium as gym

env = gym.make('LunarLander-v2')
env = DataCollectorV0(env, record_infos=True, max_buffer_steps=100000)
```

In this example, the [`DataCollectorV0`](/api/data_collector) wraps the `'LunarLander-v2'` environment from Gymnasium. The arguments passed are `record_infos` (when set to `True` the wrapper will also collect the returned `info` dictionaries to create the dataset), and the `max_buffer_steps` argument, which specifies a caching scheduler by giving the number of data steps to store in-memory before moving them to a temporary file on disk. There are more arguments that can be passed to this wrapper, a detailed description of them can be read in the [`DataCollectorV0`](/api/data_collector) documentation.

### Save Dataset

To create a Minari dataset first we need to step the environment with a given policy to allow the [`DataCollectorV0`](/api/data_collector) to record the data that will comprise the dataset. This is as simple as just looping through the Gymansium MDP API. For our example we will loop through `100` episodes with a random policy.

Finally, we need to create the Minari dataset and give it a name id. This is done by calling the [`minari.create_dataset_from_collector_env`](/api/minari_functions) Minari function which will move the temporary data recorded in the `DataCollectorV0` environment to a permanent location in the [local Minari root path](/content/dataset_format) with the Minari dataset standard structure.

Extending the code example for the `'LunarLander-v2'` environment we can create the Minari dataset as follows:

```python
import minari
import gymnasium as gym
from minari import DataCollectorV0

env = gym.make('LunarLander-v2')
env = DataCollectorV0(env, record_infos=True, max_buffer_steps=100000)

total_episodes = 100

for _ in range(total_episodes):
    env.reset(seed=123)
    while True:
        # random action policy
        action = env.action_space.sample()
        obs, rew, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            break

dataset = minari.create_dataset_from_collector_env(dataset_name="LunarLander-v2-test-v0", 
                                                   collector_env=env,
                                                   algorithm_name="Random-Policy",
                                                   code_permalink="https://github.com/Farama-Foundation/Minari",
                                                   author="Farama",
                                                   author_email="contact@farama.org")
```

When creating the Minari dataset additional metadata can be added such as the `algorithm_name` used to compute the actions, a `code_permalink` with a link to the code used to generate the dataset, as well as the `author` and `author_email`.

The [`minari.create_dataset_from_collector_env`](/api/minari_functions) function returns a [`MinariDataset`](/api/minari_dataset) object, `dataset` in the previous code snippet.

Once the dataset has been created we can check if the Minari dataset id appears in the list of local datasets:

```python
>>> import minari
>>> local_datasets = minari.list_local_datasets()
>>> local_datasets.keys()
dict_keys(['LunarLander-v2-test-v0'])
```

The [`minari.list_local_datasets`](/api/minari_functions) function returns a dictionary with keys the local Minari dataset ids and values their metadata.

There is another optional way of creating a Minari dataset and that is by using the [`minari.create_dataset_from_buffers`](/api/minari_functions) function. The data collection is left to the user instead of using the [`DataCollectorV0`](/api/data_collector) wrapper. The user will be responsible for creating their own buffers to store the stepping data, and these buffers must follow a specific structure specified in the [function API docstrings](/api/minari_functions)

### Checkpoint Minari Dataset

When collecting data with the [`DataCollectorV0`](/api/data_collector) wrapper, the recorded data is saved into temporary files and it won't be permanently saved in disk until the [`minari.create_dataset_from_collector_env`](/api/minari_functions) function is called. To avoid losing all of the collected data for large datasets, after creating a [`MinariDataset`](/api/minari_dataset) object, extra data from a [`DataCollectorV0`](/api/data_collector) can be appended to checkpoint the data collection process.

To checkpoint a dataset we can call the [`MinariDataset`](/api/minari_dataset) method `update_dataset_from_collector_env`. Every time the function [`minari.create_dataset_from_collector_env`](/api/minari_functions) or the method `update_dataset_from_collector_env` are called, the buffers from the [`DataCollectorV0`](/api/data_collector) environment are cleared.

Continuing the `'LunarLander-v2'` example we can checkpoint the newly created Minari dataset every 10 episodes as follows:

```python
import minari
import gymnasium as gym
from minari import DataCollectorV0

env = gym.make('LunarLander-v2')
env = DataCollectorV0(env, record_infos=True, max_buffer_steps=100000)

total_episodes = 100
dataset_name = "LunarLander-v2-test-v0"
dataset = None

for episode_id in range(total_episodes):
    env.reset(seed=123)
    while True:
        # random action policy
        action = env.action_space.sample()
        obs, rew, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            break

    if (episode_id + 1) % 10 == 0:
        # Update local Minari dataset every 10 episodes.
        # This works as a checkpoint to not lose the already collected data
        if dataset is None:
            dataset = minari.create_dataset_from_collector_env(dataset_name=dataset_name, 
                                                    collector_env=env,
                                                    algorithm_name="Random-Policy",
                                                    code_permalink="https://github.com/Farama-Foundation/Minari",
                                                    author="Farama",
                                                    author_email="contact@farama.org")
        else:
            assert dataset is not None    
            dataset.update_dataset_from_collector_env(env)
```

## Using Minari Datasets

Minari will only be able to load datasets that are stored in your [local root directory](/content/dataset_format). In order to use any of the dataset sampling features of Minari we first need to load the dataset as a [`MinariDataset`](/api/minari_dataset) object using the [`minari.load_dataset`](/api/minari_functions) function as follows:

```python
>>> import minari
>>> dataset = minari.load_dataset('LunarLander-v2-test-v0')
>>> dataset.name
'LunarLander-v2-test-v0'
```

### Download Remote Datasets

Minari also has a remote storage in a Google Cloud Platform (GCP) bucket which provides access to  standardize Minari datasets. The datasets hosted in the remote Farama server can be listed with [`minari.list_remote_datasets`](/api/minari_functions):

```python
>>> import minari
>>> remote_datasets = minari.list_remote_datasets()
>>> remote_datasets.keys()
dict_keys(['door-expert-v0', 'door-human-v0', 'door-cloned-v0'])
```

Same as the [`minari.list_local_datasets`](/api/minari_functions) function, the [`minari.list_remote_datasets`](/api/minari_functions) function returns a dictionary with keys equal to the remote Minari dataset ids and values their metadata.

To download any of the remote datasets into the local [Minari root path](/content/dataset_format) use the function [`minari.download_dataset`](/api/minari_functions):

```python
>>> import minari
>>> minari.download_dataset(dataset_name="door-cloned-v0")
>>> local_datasets = minari.list_local_datasets()
>>> local_datasets.keys()
dict_keys(['door-cloned-v0'])
```

### Sampling Episodes from Minari Dataset

```{eval-rst}
.. warning:: 
   WIP. Currently MinariDataset doesn't support any sampling features yet, we know this is a priority feature for the users and will soon be available.
```

### Recover Gymnasium Environment

From a `MinariDataset` object we can also recover the Gymnasium environment used to create the dataset, this can be useful for reproducibility or to generate more data for a specific dataset:

```python
import minari

dataset = minari.load_dataset('LunarLander-v2-test-v0')
env = dataset.recover_environment()

env.reset()
for _ in range(100):
    obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        env.reset()
```

### Combine Minari Datasets

Lastly, in the case of having two or more Minari datasets created with the same environment we can combine these datasets into a single one by using the Minari function [`minari.combine_datasets`](/api/minari_functions), i.e. the `'AdroitHandDoor-v1'` environment has two datasets available in the remote Farama servers, `door-human-v0` and `door-expert-v0`, we can combine the episodes in these two datasets into a new Minari dataset `door-all-v0`:

```python
>>> import minari
>>> human_dataset = minari.load_dataset('door-human-v0')
>>> expert_dataset = minari.load_dataset('door-expert-v0')
>>> combine_dataset = minari.combine_datasets(datasets_to_combine=[human_dataset,               expert_dataset], 
                                        new_dataset_name="door-all-v0")
>>> combine_dataset.name
'door-all-v0'
>>> minari.list_local_datasets()
dict_keys(['door-all-v0', 'door-human-v0', 'door-expert-v0'])
```
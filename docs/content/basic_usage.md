---
layout: "contents"
title: Basic Usage
---

# Basic Usage

Minari is a standard dataset hosting interface for Offline Reinforcement Learning applications. Minari is compatible with most of the RL environments that follow the Gymnasium API and facilitates Offline RL dataset handling by providing data collection, dataset hosting, and dataset sampling capabilities.

## Installation

To install the most recent version of the Minari library run this command: `pip install minari`

The beta release is currently under development. If you'd like to start testing or contribute to Minari then please install this project from source with:

```bash
git clone https://github.com/Farama-Foundation/Minari.git
cd Minari
pip install -e .
```

We support Python 3.7, 3.8, 3.9, 3.10 and 3.11 on Linux and macOS.

## Create Minari Dataset

### Collecting Data

```{eval-rst}
Minari can abstract the data collection process. This is achieved by using the :class:`minari.DataCollectorV0` wrapper which stores the environments stepping data in internal memory buffers before saving the dataset into disk. The :class:`minari.DataCollectorV0` wrapper can also perform caching by scheduling the amount of episodes or steps that are stored in-memory before saving the data in a temporary `Minari dataset file </content/dataset_standards>`_ . This wrapper also computes relevant metadata of the dataset while collecting the data. 

The wrapper is very simple to initialize:
```

```python
from minari import DataCollectorV0
import gymnasium as gym

env = gym.make('LunarLander-v2')
env = DataCollectorV0(env, record_infos=True, max_buffer_steps=100000)
```

```{eval-rst}
In this example, the :class:`minari.DataCollectorV0` wraps the `'LunarLander-v2'` environment from Gymnasium. The arguments passed are ``record_infos`` (when set to ``True`` the wrapper will also collect the returned ``info`` dictionaries to create the dataset), and the ``max_buffer_steps`` argument, which specifies a caching scheduler by giving the number of data steps to store in-memory before moving them to a temporary file on disk. There are more arguments that can be passed to this wrapper, a detailed description of them can be read in the :class:`minari.DataCollectorV0` documentation.
```

### Save Dataset

```{eval-rst}
To create a Minari dataset first we need to step the environment with a given policy to allow the :class:`minari.DataCollectorV0` to record the data that will comprise the dataset. This is as simple as just looping through the Gymansium MDP API. For our example we will loop through ``100`` episodes of the ``'LunarLander-v2'`` environment with a random policy.

Finally, we need to create the Minari dataset and give it a name id. This is done by calling the :func:`minari.create_dataset_from_collector_env` Minari function which will move the temporary data recorded in the :class:`minari.DataCollectorV0` environment to a permanent location in the `local Minari root path </content/dataset_standards>`_ with the Minari dataset standard structure.

Extending the code example for the ``'LunarLander-v2'`` environment we can create the Minari dataset as follows:
```

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

dataset = minari.create_dataset_from_collector_env(dataset_id="LunarLander-v2-test-v0", 
                                                   collector_env=env,
                                                   algorithm_name="Random-Policy",
                                                   code_permalink="https://github.com/Farama-Foundation/Minari",
                                                   author="Farama",
                                                   author_email="contact@farama.org")
```

```{eval-rst}
When creating the Minari dataset additional metadata can be added such as the ``algorithm_name`` used to compute the actions, a ``code_permalink`` with a link to the code used to generate the dataset, as well as the ``author`` and ``author_email``.

The :func:`minari.create_dataset_from_collector_env` function returns a :class:`minari.MinariDataset` object, ``dataset`` in the previous code snippet.

Once the dataset has been created we can check if the Minari dataset id appears in the list of local datasets:
```

```python
>>> import minari
>>> local_datasets = minari.list_local_datasets()
>>> local_datasets.keys()
dict_keys(['LunarLander-v2-test-v0'])
```

```{eval-rst}
The :func:`minari.list_local_datasets` function returns a dictionary with keys the local Minari dataset ids and values their metadata.

There is another optional way of creating a Minari dataset and that is by using the :func:`minari.create_dataset_from_buffers` function. The data collection is left to the user instead of using the :class:`minari.DataCollectorV0` wrapper. The user will be responsible for creating their own buffers to store the stepping data, and these buffers must follow a specific structure specified in the function API documentation.
```

### Checkpoint Minari Dataset

```{eval-rst}
When collecting data with the :class:`minari.DataCollectorV0` wrapper, the recorded data is saved into temporary files and it won't be permanently saved in disk until the :func:`minari.create_dataset_from_collector_env` function is called. For large datasets, to avoid losing all of the collected data, extra data from a :class:`minari.DataCollectorV0` can be appended to checkpoint the data collection process.


To checkpoint a dataset we can call the :func:`minari.MinariDataset.update_dataset_from_collector_env` method. Every time the function :func:`minari.create_dataset_from_collector_env` or the method :func:`minari.MinariDataset.update_dataset_from_collector_env` are called, the buffers from the :class:`minari.DataCollectorV0` environment are cleared.

Continuing the ``'LunarLander-v2'`` example we can checkpoint the newly created Minari dataset every 10 episodes as follows:
```

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
            dataset = minari.create_dataset_from_collector_env(dataset_id=dataset_name, 
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

```{eval-rst}
Minari will only be able to load datasets that are stored in your `local root directory  </content/dataset_standards>`_ . In order to use any of the dataset sampling features of Minari we first need to load the dataset as a :class:`minari.MinariDataset` object using the :func:`minari.load_dataset` function as follows:
```

```python
>>> import minari
>>> dataset = minari.load_dataset('LunarLander-v2-test-v0')
>>> dataset.name
'LunarLander-v2-test-v0'
```

### Download Remote Datasets

```{eval-rst}
Minari also has a remote storage in a Google Cloud Platform (GCP) bucket which provides access to  standardize Minari datasets. The datasets hosted in the remote Farama server can be listed with :func:`minari.list_remote_datasets`:
```

```python
>>> import minari
>>> remote_datasets = minari.list_remote_datasets()
>>> remote_datasets.keys()
dict_keys(['door-expert-v0', 'door-human-v0', 'door-cloned-v0'])
```

```{eval-rst}
Same as the :func:`minari.list_local_datasets` function, the :func:`minari.list_remote_datasets` function returns a dictionary with keys equal to the remote Minari dataset ids and values their metadata.

To download any of the remote datasets into the local `Minari root path </content/dataset_standards>`_ use the function :func:`minari.download_dataset`:
```

```python
>>> import minari
>>> minari.download_dataset(dataset_id="door-cloned-v0")
>>> local_datasets = minari.list_local_datasets()
>>> local_datasets.keys()
dict_keys(['door-cloned-v0'])
```

### Sampling Episodes

``` {eval-rst}
Minari can retrieve a certain amount of episode shards from the dataset files as a list of :class:`minari.EpisodeData` objects. The sampling process of the Minari datasets is performed through the method :func:`minari.MinariDataset.sample_episodes`. This method is a generator that randomly samples ``n`` number of :class:`minari.EpisodeData` from the :class:`minari.MinariDataset`. The seed of this generator can be set with :func:`minari.MinariDataset.set_seed`. For example:
```

```python
import minari

dataset = minari.load_dataset("door-cloned-v0")
dataset.set_seed(seed=123)

for i in range(5):
    # sample 5 episodes from the dataset
    episodes = dataset.sample_episodes(n_episodes=5)
    # get id's from the sampled episodes
    ids = list(map(lambda ep: ep.id, episodes))
    print(f"EPISODE ID'S SAMPLE {i}: {ids}")
```

```{eval-rst}
This code will show the following. 
```

```bash
>>> EPISODE ID'S SAMPLE 0: [1, 13, 0, 22, 15]
>>> EPISODE ID'S SAMPLE 1: [3, 10, 23, 7, 18]
>>> EPISODE ID'S SAMPLE 2: [12, 6, 0, 18, 19]
>>> EPISODE ID'S SAMPLE 3: [9, 4, 15, 3, 17]
>>> EPISODE ID'S SAMPLE 4: [19, 4, 12, 17, 21]
```

```{eval-rst}
Notice that in each sample non of the episodes are sampled more than once but the same episode can be retrieved in different :func:`minari.MinariDataset.sample_episodes` calls.

Minari doesn't serve the purpose of creating replay buffers out of the Minari datasets, we leave this task for the user to make for their specific needs.
To create your own buffers and dataloaders, you may need the ability to iterate through an episodes in a deterministic order. This can be achieved with :func:`minari.MinariDataset.iterate_episodes`. This method is a generator that iterates over :class:`minari.EpisodeData` episodes from :class:`minari.MinariDataset`. Specific indices can be also provided. For example:
```

```python
import minari

dataset = minari.load_dataset("door-cloned-v0")
episodes_generator = dataset.iterate_episodes(episode_indices=[1, 2, 0])

for episode in episodes_generator:
    print(f"EPISODE ID {episode.id}")
```

```{eval-rst}
This code will show the following. 
```

```bash
>>> EPISODE ID 1
>>> EPISODE ID 2
>>> EPISODE ID 0
```

```{eval-rst}
In addition, the :class:`minari.MinariDataset` dataset itself is iterable. However, in this case the indices will have to be filtered separately using :func:`minari.MinariDataset.filter_episodes`.
```

```python
import minari

dataset = minari.load_dataset("door-cloned-v0")

for episode in dataset:
    print(f"EPISODE ID {episode.id}")
```


#### Filter Episodes

```{eval-rst}
The episodes in the dataset can be filtered before sampling. This is done with a custom conditional callable passed to :func:`minari.MinariDataset.filter_episodes`. The input to the conditional callable is an episode group in `h5py.Group <https://docs.h5py.org/en/stable/high/group.html>`_ format and the return value must be ``True`` if you want to keep the episode or ``False`` otherwise. The method will return a new :class:`minari.MinariDataset`:
```

```python
import minari

dataset = minari.load_dataset("door-human-v0")

print(f'TOTAL EPISODES ORIGINAL DATASET: {dataset.total_episodes}')

# get episodes with mean reward greater than 2
filter_dataset = dataset.filter_episodes(lambda episode: episode["rewards"].attrs.get("mean") > 2)

print(f'TOTAL EPISODES FILTER DATASET: {filter_dataset.total_episodes}')
```

Some episodes were removed from the dataset:

```bash
>>> TOTAL EPISODES ORIGINAL DATASET: 25
>>> TOTAL EPISODES FILTER DATASET: 18
```

#### Split Dataset

```{eval-rst}
Minari provides another utility function to divide a dataset into multiple datasets, :func:`minari.split_dataset`
```

```python
import minari

dataset = minari.load_dataset("door-human-v0")

split_datasets = minari.split_dataset(dataset, sizes=[20, 5], seed=123)

print(f'TOTAL EPISODES FIRST SPLIT: {split_datasets[0].total_episodes}')
print(f'TOTAL EPISODES SECOND SPLIT: {split_datasets[1].total_episodes}')
```

```bash
>>> TOTAL EPISODES FIRST SPLIT: 20
>>> TOTAL EPISODES SECOND SPLIT: 5
```

### Recover Environment

```{eval-rst}
From a :class:`minari.MinariDataset` object we can also recover the Gymnasium environment used to create the dataset, this can be useful for reproducibility or to generate more data for a specific dataset:
```

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

```{eval-rst}
Lastly, in the case of having two or more Minari datasets created with the same environment we can combine these datasets into a single one by using the Minari function :func:`minari.combine_datasets`, i.e. the ``'AdroitHandDoor-v1'`` environment has two datasets available in the remote Farama servers, ``door-human-v0`` and ``door-expert-v0``, we can combine the episodes in these two datasets into a new Minari dataset ``door-all-v0``:
```

```python
>>> import minari
>>> human_dataset = minari.load_dataset('door-human-v0')
>>> expert_dataset = minari.load_dataset('door-expert-v0')
>>> combine_dataset = minari.combine_datasets(datasets_to_combine=[human_dataset,               expert_dataset], 
                                        new_dataset_id="door-all-v0")
>>> combine_dataset.name
'door-all-v0'
>>> minari.list_local_datasets()
dict_keys(['door-all-v0', 'door-human-v0', 'door-expert-v0'])
```
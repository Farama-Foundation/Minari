---
layout: "contents"
title: Basic Usage
---

# Basic Usage

Minari is a standard dataset hosting interface for Offline Reinforcement Learning applications. Minari is compatible with most of the RL environments that follow the Gymnasium API and facilitates Offline RL dataset handling by providing data collection, dataset hosting, and dataset sampling capabilities.

## Installation

To install the most recent version of the Minari library run this command:

```bash
pip install minari
```

This will install the minimum required dependencies. Additional dependencies will be prompted for installation based on your use case. To install all dependencies at once, use:
```bash
pip install "minari[all]"
```

If you'd like to start testing or contribute to Minari then please install this project from source with:

```
git clone https://github.com/Farama-Foundation/Minari.git
cd Minari
pip install -e ".[all]"
```

We support Python with minimum version 3.8 on Linux and macOS.

## Using Minari Datasets

### Download Datasets

```{eval-rst}
Minari has a remote storage which provides access to a variety of datasets. The datasets hosted in the remote Farama server can be listed running in the terminal:
```

```bash
minari list remote
```

```
                                Minari datasets in Farama server
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Name                           ┃ Total Episodes ┃ Total Steps ┃ Dataset Size ┃ Author          ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ D4RL/antmaze/large-diverse-v1  │           1000 │     1000000 │ 605.2 MB     │ Alex Davey      │
│ D4RL/antmaze/large-play-v1     │           1000 │     1000000 │ 605.2 MB     │ Alex Davey      │
│ D4RL/antmaze/medium-diverse-v1 │           1000 │     1000000 │ 605.2 MB     │ Alex Davey      │
│             ...                │       ...      │     ...     │     ...      │       ...       │
```

To use your own server with Minari, set the `MINARI_REMOTE` environment variable in the format `remote-type://remote-path`. For example, to set up a GCP bucket named `my-datasets`, run the following command:

```bash
export MINARI_REMOTE=gcp://my-datasets
```

Currently, only GCP is supported, but we plan to support other cloud providers in the future.


```{eval-rst}
To download any of the remote datasets into the local storage use the download command:
```

```bash
minari download D4RL/door/human-v2
```

### Load Local Datasets

```{eval-rst}
Minari will only be able to load datasets that are stored in your `local root directory  </content/dataset_standards>`_ . To list the local datasets, use the list command:
```

```bash
minari list local
```

```
                 Local Minari datasets('/Users/farama/.minari/datasets/')
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
┃ Name               ┃ Total Episodes ┃ Total Steps ┃ Dataset Size ┃ Author             ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
│ D4RL/door/human-v2 │             25 │        6729 │ 7.1 MB       │ Rodrigo de Lazcano │
└────────────────────┴────────────────┴─────────────┴──────────────┴────────────────────┘
```

```{eval-rst}
In order to use any of the dataset sampling features of Minari we first need to load the dataset as a :class:`minari.MinariDataset` object using the :func:`minari.load_dataset` Python function as follows:
```

```python
import minari
dataset = minari.load_dataset('D4RL/door/human-v2')
print("Observation space:", dataset.observation_space)
print("Action space:", dataset.action_space)
print("Total episodes:", dataset.total_episodes)
print("Total steps:", dataset.total_steps)
```
```
Observation space: Box(-inf, inf, (39,), float64)
Action space: Box(-1.0, 1.0, (28,), float32)
Total episodes: 25
Total steps: 6729
```

### Sampling Episodes

``` {eval-rst}
Minari can retrieve a certain amount of episode shards from the dataset files as a list of :class:`minari.EpisodeData` objects. The sampling process of the Minari datasets is performed through the method :func:`minari.MinariDataset.sample_episodes`. This method is a generator that randomly samples ``n`` number of :class:`minari.EpisodeData` from the :class:`minari.MinariDataset`. The seed of this generator can be set with :func:`minari.MinariDataset.set_seed`. For example:
```

```python
import minari

dataset = minari.load_dataset("D4RL/door/human-v2")
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

```
EPISODE ID'S SAMPLE 0: [1, 13, 0, 22, 15]
EPISODE ID'S SAMPLE 1: [3, 10, 23, 7, 18]
EPISODE ID'S SAMPLE 2: [12, 6, 0, 18, 19]
EPISODE ID'S SAMPLE 3: [9, 4, 15, 3, 17]
EPISODE ID'S SAMPLE 4: [19, 4, 12, 17, 21]
```

```{eval-rst}
Notice that in each sample non of the episodes are sampled more than once but the same episode can be retrieved in different :func:`minari.MinariDataset.sample_episodes` calls.

Minari doesn't serve the purpose of creating replay buffers out of the Minari datasets, we leave this task for the user to make for their specific needs.
To create your own buffers and dataloaders, you may need the ability to iterate through an episodes in a deterministic order. This can be achieved with :func:`minari.MinariDataset.iterate_episodes`. This method is a generator that iterates over :class:`minari.EpisodeData` episodes from :class:`minari.MinariDataset`. Specific indices can be also provided. For example:
```

```python
import minari

dataset = minari.load_dataset("D4RL/door/human-v2")
episodes_generator = dataset.iterate_episodes(episode_indices=[1, 2, 0])

for episode in episodes_generator:
    print(f"EPISODE ID {episode.id}")
```

```{eval-rst}
This code will show the following.
```

```
EPISODE ID 1
EPISODE ID 2
EPISODE ID 0
```

```{eval-rst}
In addition, the :class:`minari.MinariDataset` dataset itself is iterable:.
```

```python
import minari

dataset = minari.load_dataset("D4RL/door/human-v2")

for episode in dataset:
    print(f"EPISODE ID {episode.id}")
```


#### Filter Episodes

```{eval-rst}
The episodes in the dataset can be filtered before sampling. This is done with a custom conditional callable passed to :func:`minari.MinariDataset.filter_episodes`. The input to the conditional callable is an :class:`minari.EpisodeData` and the return value must be ``True`` if you want to keep the episode or ``False`` otherwise. The method will return a new :class:`minari.MinariDataset`:
```

```python
import minari

dataset = minari.load_dataset("D4RL/door/human-v2")

print(f'TOTAL EPISODES ORIGINAL DATASET: {dataset.total_episodes}')

# get episodes with mean reward greater than 2
filter_dataset = dataset.filter_episodes(lambda episode: episode.rewards.mean() > 2)

print(f'TOTAL EPISODES FILTER DATASET: {filter_dataset.total_episodes}')
```

Some episodes were removed from the dataset:

```
TOTAL EPISODES ORIGINAL DATASET: 25
TOTAL EPISODES FILTER DATASET: 18
```

#### Split Dataset

```{eval-rst}
Minari provides another utility function to divide a dataset into multiple datasets, :func:`minari.split_dataset`
```

```python
import minari

dataset = minari.load_dataset("D4RL/door/human-v2", download=True)

split_datasets = minari.split_dataset(dataset, sizes=[20, 5], seed=123)

print(f'TOTAL EPISODES FIRST SPLIT: {split_datasets[0].total_episodes}')
print(f'TOTAL EPISODES SECOND SPLIT: {split_datasets[1].total_episodes}')
```

```
TOTAL EPISODES FIRST SPLIT: 20
TOTAL EPISODES SECOND SPLIT: 5
```

### Recover Environment

```{eval-rst}
From a :class:`minari.MinariDataset` object we can also recover the Gymnasium environment used to create the dataset, this can be useful for reproducibility or to generate more data for a specific dataset:
```

```python
import minari

dataset = minari.load_dataset('D4RL/door/human-v2')
env = dataset.recover_environment()

env.reset()
for _ in range(100):
    obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        env.reset()
```


```{eval-rst}

.. note::
   There are some datasets that provide a different environment for evaluation purposes than the one used for collecting the data. This environment can be recovered by setting to `True` the `eval_env` argument:

   .. code-block::

        import minari

        dataset = minari.load_dataset('D4RL/door/human-v2')
        eval_env = dataset.recover_environment(eval_env=True)

   If the dataset doesn't have an `eval_env_spec` attribute, the environment used for collecting the data will be retrieved by default.
```

### Combine Minari Datasets

```{eval-rst}
In the case of having two or more Minari datasets created with the same environment we can combine these datasets into a single one by using the Minari function :func:`minari.combine_datasets`, i.e. the ``'AdroitHandDoor-v1'`` environment has two datasets available in the remote Farama servers, ``D4RL/door/human-v2`` and ``D4RL/door/expert-v2``, we can combine the episodes in these two datasets into a new Minari dataset ``D4RL/door/all-v0``:
```

```bash
minari download D4RL/door/expert-v2
minari combine D4RL/door/human-v2 D4RL/door/expert-v2 --dataset-id=D4RL/door/all-v0
minari list local
```
```
                    Local Minari datasets('/Users/farama/.minari/datasets/')
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
┃ Name                ┃ Total Episodes ┃ Total Steps ┃ Dataset Size ┃ Author             ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
│ D4RL/door/all-v0    │           5025 │     1006729 │ 1103.5 MB    │ Rodrigo de Lazcano │
│ D4RL/door/expert-v2 │           5000 │     1000000 │ 1096.4 MB    │ Rodrigo de Lazcano │
│ D4RL/door/human-v2  │             25 │        6729 │ 7.1 MB       │ Rodrigo de Lazcano │
└─────────────────────┴────────────────┴─────────────┴──────────────┴────────────────────┘
```

## Create Minari Dataset

### Collecting Data

```{eval-rst}
Minari can abstract the data collection process. This is achieved by using the :class:`minari.DataCollector` wrapper which stores the environments stepping data in internal memory buffers before saving the dataset into disk. The :class:`minari.DataCollector` wrapper can also perform caching by scheduling the amount of episodes or steps that are stored in-memory before saving the data in a temporary `Minari dataset file </content/dataset_standards>`_ . This wrapper also computes relevant metadata of the dataset while collecting the data.

The wrapper is very simple to initialize:
```

```python
from minari import DataCollector
import gymnasium as gym

env = gym.make('CartPole-v1')
env = DataCollector(env, record_infos=True)
```

```{eval-rst}
In this example, the :class:`minari.DataCollector` wraps the `'CartPole-v1'` environment from Gymnasium. We set ``record_infos=True`` so the wrapper will also collect the returned ``info`` dictionaries to create the dataset. For the full list of arguments, read the :class:`minari.DataCollector` documentation.
```

### Save Dataset

```{eval-rst}
To create a Minari dataset first we need to step the environment with a given policy to allow the :class:`minari.DataCollector` to record the data that will comprise the dataset. This is as simple as just looping through the Gymansium MDP API. For our example we will loop through ``100`` episodes of the ``'CartPole-v1'`` environment with a random policy.

Finally, we need to create the Minari dataset and give it a name id. This is done by calling the :func:`minari.DataCollector.create_dataset` Minari function which will move the temporary data recorded in the :class:`minari.DataCollector` environment to a permanent location in the `local Minari root path </content/dataset_standards>`_ with the Minari dataset standard structure.

Extending the code example for the ``'CartPole-v1'`` environment we can create the Minari dataset as follows:
```

```python
import minari
import gymnasium as gym
from minari import DataCollector

env = gym.make('CartPole-v1')
env = DataCollector(env, record_infos=True)

total_episodes = 100

for _ in range(total_episodes):
    env.reset(seed=123)
    while True:
        # random action policy
        action = env.action_space.sample()
        obs, rew, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

dataset = env.create_dataset(
    dataset_id="cartpole/test-v0",
    algorithm_name="Random-Policy",
    code_permalink="https://github.com/Farama-Foundation/Minari",
    author="Farama",
    author_email="contact@farama.org"
)
```

```{eval-rst}
When creating the Minari dataset additional metadata can be added such as the ``algorithm_name`` used to compute the actions, a ``code_permalink`` with a link to the code used to generate the dataset, as well as the ``author`` and ``author_email``.

The :func:`minari.DataCollector.create_dataset` function returns a :class:`minari.MinariDataset` object, ``dataset`` in the previous code snippet.

Once the dataset has been created we can check if the Minari dataset id appears in the list of local datasets:
```

```bash
minari list local
```
```
        Local Minari datasets('/Users/farama/.minari/datasets/')
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Name             ┃ Total Episodes ┃ Total Steps ┃ Dataset Size ┃  Author  ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ cartpole-test-v0 │            100 │        2059 │ 1.6 MB       │  Farama  │
└──────────────────┴────────────────┴─────────────┴──────────────┴──────────┘
```

```{eval-rst}
The :func:`minari.list_local_datasets` function returns a dictionary with keys the local Minari dataset ids and values their metadata.

There is another optional way of creating a Minari dataset and that is by using the :func:`minari.create_dataset_from_buffers` function. The data collection is left to the user instead of using the :class:`minari.DataCollector` wrapper. The user will be responsible for creating their own buffers to store the stepping data, and these buffers must follow a specific structure specified in the function API documentation.
```

### Checkpoint Minari Dataset

```{eval-rst}
When collecting data with the :class:`minari.DataCollector` wrapper, the recorded data is saved into temporary files and it won't be permanently saved on disk until the :func:`DataCollector.create_dataset` function is called. To prevent losing data for large datasets, it is recommended to create the dataset during data collection and append the data to it using :func:`DataCollector.add_to_dataset`.

Continuing the ``'CartPole-v1'`` example we can checkpoint the newly created Minari dataset every 10 episodes as follows:
```

```python
import minari
import gymnasium as gym
from minari import DataCollector

env = gym.make('CartPole-v1')
env = DataCollector(env, record_infos=True)

total_episodes = 100
dataset_id = "cartpole/test-v0"
dataset = None
if dataset_id in minari.list_local_datasets():
    dataset = minari.load_dataset(dataset_id)

for episode_id in range(total_episodes):
    env.reset()
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
            dataset = env.create_dataset(
                dataset_id=dataset_id,
                algorithm_name="Random-Policy",
                code_permalink="https://github.com/Farama-Foundation/Minari",
                author="Farama",
                author_email="contact@farama.org"
            )
        else:
            env.add_to_dataset(dataset)
```


## Using Namespaces

```{eval-rst}
Namespaces can be used to group together common datasets and provide them with a hierarchical structure. For example, suppose we want to create a series of `Classic Control <https://gymnasium.farama.org/environments/classic_control/>`_ datasets (`cartpole`, `acrobot`, e.t.c.) using the dataset creation code above. Instead of specifying ``dataset_id=cartpole-test-v0``, we can use e.g. ``classic_control/cartpole-test-v0`` when creating the dataset. This, and all other datasets with a ``dataset_id`` that starts with ``classic_control/`` will now be stored together in the ``classic_control`` namespace.

For more flexibility, namespaces can be created and modified directly using the :doc:`Namespace API <../api/namespace/namespace>`.
```

---
layout: "contents"
title: Dataset Standards
---

# Dataset Standards


## Minari Storage

### Minari root

Minari stores the offline datasets under a common root directory. The root directory path for the local datasets is set by default to `~/.minari/datasets/`. However, this path can be modified by setting the environment variable `MINARI_DATASETS_PATH`.

The remote datasets are kept in the public Google Cloud Platform (GCP) bucket [`minari-datasets`](https://console.cloud.google.com/storage/browser/minari-datasets;tab=objects?forceOnBucketsSortingFiltering=false&project=mcmes-345620&prefix=&forceOnObjectsSortingFiltering=false).

The first level of the root directory tree contains the Minari dataset directories, which are named after the datasets `id`. The datasets `id` must follow the syntax `(env_name-)(dataset_name)(-v(version))`, where:

- `env_name`: a string that describes the environment from which the dataset was created. If a dataset comes from the [`AdroitHandDoor`](https://robotics.farama.org/envs/adroit_hand/adroit_door/) environment `ennv_name` can be equal to `door`.
- `dataset_name`: a string describing the content of the dataset. For example, if the dataset for the `AdroitHandDoor` environment was generated from human input we can give the value `human` to `dataset_name`.
- `version`: integer value that represent the number of versions for `door-human-v(version)` dataset, starting from `0`.

In the end, the `id` of the dataset for the initial version of the `AdroitHandDoor` environment example will be `door-human-v0`.

### Data files

Each Minari dataset directory contains another directory named `data` where the files of the collected offline data are stored (more directories are yet to be included for additional information, `_docs` and `policies` are WIP). The `data` directory can contain multiple `.hdf5` files storing the offline data. When using a Minari dataset the offline data is loaded homogeneously from all `.hdf5` as if it were a single file. The names for these files are:

- `main_data.hdf5`: root file that aside from raw data it also contains all the metadata of the global dataset and external links to the data in the other files. Minari will read this file when a dataset is loaded.
- `additional_data_x.hdf5`: these files contain raw data. Each of them is generated after making a checkpoint when collecting the offline data with `MinariDataset.update_datasets(env)`.

The following directory tree of a Minari root path contains three different datasets named `dataset_id-v0`, `dataset_id-v1`, and `other_dataset_id-v0`. The offline data of the `dataset_id-v1` is saved in a single `main_data.hdf5` file, while for `dataset_id-v0` the offline data has been divided into multiple `.hdf5` files.

<div class="only-light">
<ul class="directory-list">
<li class="folder">minari_root
    <ul>
    <li class="folder">dataset_id-v0
        <ul>
        <li class="folder">data
            <ul>
            <li class="file">main_data.hdf5</li>
            <li class="file">additional_data_0.hdf5</li>
            <li class="file">additional_data_1.hdf5</li>
            </ul>
        </li>
        </ul>
    </li>
    <li class="folder">dataset_id-v1
        <ul>
        <li class="folder">data
            <ul>
            <li class="file">main_data.hdf5</li>
            </ul>
        </li>
        </ul>
    </li>
    <li class="folder-closed">other_dataset_id-v0</li>
    </ul>
</li>
</ul>
</div>

<div class="only-dark">
<ul class="directory-list">
<li class="folder-white">minari_root
    <ul class="white">
    <li class="folder-white">dataset_id-v0
        <ul class="white">
        <li class="folder-white">data
            <ul class="white">
            <li class="file-white">main_data.hdf5</li>
            <li class="file-white">additional_data_0.hdf5</li>
            <li class="file-white">additional_data_1.hdf5</li>
            </ul>
        </li>
        </ul>
    </li>
    <li class="folder-white">dataset_id-v1
        <ul class="white">
        <li class="folder-white">data
            <ul class="white">
            <li class="file-white">main_data.hdf5</li>
            </ul>
        </li>
        </ul>
    </li>
    <li class="folder-white-closed">other_dataset_id-v0</li>
    </ul>
</li>
</ul>
</div>

## Dataset File Format

Minari datasets are stored in [`HDF5`](https://www.hdfgroup.org/solutions/hdf5/) file format by using the [`h5py`](https://www.h5py.org/) Python interface. We leverage the hierarchical structure of `HDF5` files of [`group`](https://docs.h5py.org/en/stable/high/group.html) and [`dataset`](https://docs.h5py.org/en/stable/high/dataset.html) elements to clearly divide the recorded step data into episode `groups` and add custom metadata to the whole dataset, to each episode `group`, or to the individual `HDF5` `datasets` that comprise each episode `group`.

More information about the features that the `HDF5` file format support can be read in this [link](https://www.neonscience.org/resources/learning-hub/tutorials/about-hdf5)

### HDF5 file structure

The offline data is organized inside the `main_data.hdf5` file in episode [`groups`](https://docs.h5py.org/en/stable/high/dataset.html) named as `episode_id`. Each episode group contains all the stepping data from a Gymnasium environment until the environment is `terminated` or `truncated`.

The stepping data inside the episode group is divided into some required `datasets` (`StepData`) plus other optional `groups` and nested `sub-groups` such as `infos`. If the action and observation spaces both are simple spaces (not `Tuple` and not `Dict`), then the hierarchical tree of the Minari dataset `HDF5` file will end up looking as follows:

<div class="only-light">
<ul class="directory-list">
<li class="file">main_data.hdf5
    <ul>
        <li class="folder">episode_0
            <ul>
                <li class="dataset">observations</li>
                <li class="dataset">actions</li>
                <li class="dataset">terminations</li>
                <li class="dataset">truncations</li>
                <li class="dataset">rewards</li>
                <li class="folder">infos
                <ul>
                    <li class="dataset">infos_datasets</li>
                    <li class="folder">infos_subgroup
                    <ul>
                        <li class="dataset">more_datasets</li>
                    </ul>
                    </li>
                </ul>
                </li>
                <li class="folder">additional_groups
                    <ul>
                        <li class="dataset">additional_datasets</li>
                    </ul>
                </li>
            </ul>
        </li>
        <li class="folder-closed">episode_1</li>
        <li class="folder-closed">episode_2</li>
        <ul><br></ul>
        <li class="folder-closed">episode_id</li>
    </ul>
</li>
</ul>
</div>

<div class="only-dark">
<ul class="directory-list">
    <li class="file-white" style="color:white">main_data.hdf5
        <ul class="white">
            <li class="folder-white">episode_0
                <ul class="white">
                    <li class="dataset-white">observations</li>
                    <li class="dataset-white">actions</li>
                    <li class="dataset-white">terminations</li>
                    <li class="dataset-white">truncations</li>
                    <li class="dataset-white">rewards</li>
                    <li class="folder-white">infos
                    <ul class="white">
                        <li class="dataset-white">infos_datasets</li>
                        <li class="folder-white">infos_subgroup
                        <ul class="white">
                            <li class="dataset-white">more_datasets</li>
                        </ul>
                        </li>
                    </ul>
                    </li>
                    <li class="folder-white">additional_groups
                        <ul class="white">
                            <li class="dataset-white">additional_datasets</li>
                        </ul>
                    </li>
                </ul>
            </li>
            <li class="folder-white-closed">episode_1</li>
            <li class="folder-white-closed">episode_2</li>
            <ul class="white"><br></ul>
            <li class="folder-white-closed">episode_id</li>
        </ul>
    </li>
</ul>
</div>

In the case where, the observation space is a relatively complex `Dict` space with the following definition:
```
spaces.Dict(
    {
        "component_1": spaces.Box(low=-1, high=1, dtype=np.float32),
        "component_2": spaces.Dict(
            {
                "subcomponent_1": spaces.Box(low=2, high=3, dtype=np.float32),
                "subcomponent_2": spaces.Box(low=4, high=5, dtype=np.float32),
            }
        ),
    }
)
```
and the action space is a `Box` space, the resulting Minari dataset `HDF5` file will end up looking as follows:

<div class="only-light">
<ul class="directory-list">
<li class="file">main_data.hdf5
    <ul>
        <li class="folder">episode_0
            <ul>
                <li class="folder">observations
                <ul>
                    <li class="dataset">component_1</li>
                    <li class="folder">component_2
                    <ul>
                    <li class="dataset"> subcomponent_1 </li>
                    <li class="dataset"> subcomponent_2 </li>
                    </ul>
                    </li>
                </ul>
                </li>
                <li class="dataset">actions</li>
                <li class="dataset">terminations</li>
                <li class="dataset">truncations</li>
                <li class="dataset">rewards</li>
                <li class="folder">infos
                <ul>
                    <li class="dataset">infos_datasets</li>
                    <li class="folder">infos_subgroup
                    <ul>
                        <li class="dataset">more_datasets</li>
                    </ul>
                    </li>
                </ul>
                </li>
                <li class="folder">additional_groups
                    <ul>
                        <li class="dataset">additional_datasets</li>
                    </ul>
                </li>
            </ul>
        </li>
        <li class="folder-closed">episode_1</li>
        <li class="folder-closed">episode_2</li>
        <ul><br></ul>
        <li class="folder-closed">episode_id</li>
    </ul>
</li>
</ul>
</div>

<div class="only-dark">
<ul class="directory-list">
    <li class="file-white" style="color:white">main_data.hdf5
        <ul class="white">
            <li class="folder-white">episode_0
                <ul class="white">
                <li class="folder-white">observations
                <ul>
                    <li class="dataset-white">component_1</li>
                    <li class="folder-white">component_2
                        <ul class="white">
                            <li class="dataset-white"> subcomponent_1 </li>
                            <li class="dataset-white"> subcomponent_2 </li>
                        </ul>
                    </li>
                </ul>
                </li>
                <li class="dataset-white">actions</li>
                <li class="dataset-white">terminations</li>
                <li class="dataset-white">truncations</li>
                <li class="dataset-white">rewards</li>
                <li class="folder-white">infos
                <ul class="white">
                    <li class="dataset-white">infos_datasets</li>
                    <li class="folder-white">infos_subgroup
                    <ul class="white">
                        <li class="dataset-white">more_datasets</li>
                    </ul>
                    </li>
                </ul>
                </li>
                <li class="folder-white">additional_groups
                    <ul class="white">
                        <li class="dataset-white">additional_datasets</li>
                    </ul>
                </li>
            </ul>
        </li>
        <li class="folder-white-closed">episode_1</li>
        <li class="folder-white-closed">episode_2</li>
        <ul class="white"><br></ul>
        <li class="folder-white-closed">episode_id</li>
        </ul>
    </li>
</ul>
</div>

Similarly, consider the case where we have a `Box` space as an observation space and a relatively complex `Tuple` space as an action space with the following definition:
```
spaces.Tuple(
    (
        spaces.Box(low=2, high=3, dtype=np.float32),
        spaces.Tuple(
            (
                spaces.Box(low=2, high=3, dtype=np.float32),
                spaces.Box(low=4, high=5, dtype=np.float32),
            )
        ),
    )
)
```
In this case, the resulting Minari dataset `HDF5` file will end up looking as follows:

<div class="only-light">
<ul class="directory-list">
<li class="file">main_data.hdf5
    <ul>
        <li class="folder">episode_0
            <ul>
                <li class="dataset">observations</li>
                <li class="folder">actions
                <ul>
                    <li class="dataset">_index_0</li>
                    <li class="folder">_index_1
                    <ul>
                    <li class="dataset"> _index_0 </li>
                    <li class="dataset"> _index_1 </li>
                    </ul>
                    </li>
                </ul>
                </li>
                <li class="dataset">terminations</li>
                <li class="dataset">truncations</li>
                <li class="dataset">rewards</li>
                <li class="folder">infos
                <ul>
                    <li class="dataset">infos_datasets</li>
                    <li class="folder">infos_subgroup
                    <ul>
                        <li class="dataset">more_datasets</li>
                    </ul>
                    </li>
                </ul>
                </li>
                <li class="folder">additional_groups
                    <ul>
                        <li class="dataset">additional_datasets</li>
                    </ul>
                </li>
            </ul>
        </li>
        <li class="folder-closed">episode_1</li>
        <li class="folder-closed">episode_2</li>
        <ul><br></ul>
        <li class="folder-closed">episode_id</li>
    </ul>
</li>
</ul>
</div>

<div class="only-dark">
<ul class="directory-list">
    <li class="file-white" style="color:white">main_data.hdf5
        <ul class="white">
            <li class="folder-white">episode_0
                <ul class="white">
                    <li class="dataset-white">observations</li>
                    <li class="folder-white">actions
                        <ul>
                            <li class="dataset-white">_index_0</li>
                            <li class="folder-white">_index_1
                            <ul>
                            <li class="dataset-white"> _index_0 </li>
                            <li class="dataset-white"> _index_1 </li>
                            </ul>
                            </li>
                        </ul>
                    </li>
                    <li class="dataset-white">terminations</li>
                    <li class="dataset-white">truncations</li>
                    <li class="dataset-white">rewards</li>
                    <li class="folder-white">infos
                    <ul class="white">
                        <li class="dataset-white">infos_datasets</li>
                        <li class="folder-white">infos_subgroup
                        <ul class="white">
                            <li class="dataset-white">more_datasets</li>
                        </ul>
                        </li>
                    </ul>
                    </li>
                    <li class="folder-white">additional_groups
                        <ul class="white">
                            <li class="dataset-white">additional_datasets</li>
                        </ul>
                    </li>
                </ul>
            </li>
            <li class="folder-white-closed">episode_1</li>
            <li class="folder-white-closed">episode_2</li>
            <ul class="white"><br></ul>
            <li class="folder-white-closed">episode_id</li>
        </ul>
    </li>
</ul>
</div>

Note how the `Tuple` space elements are assigned corresponding keys of the format `f"_index_{i}"` were `i` is their index in the `Tuple` space.


The required `datasets` found in the episode groups correspond to the data involved in every Gymnasium step call: `obs, rew, terminated, truncated, info = env.step(action)`: `observations`, `actions`, `rewards`, `terminations`, and `truncations`. These datasets are `np.ndarray` or nested groups of `np.ndarray` and other groups, depending on the observation and action spaces, and the space of all datasets under each required top-level episode key is equal to:

- `actions`: `shape=(num_steps, action_space_component_shape)`. If the action or observation space is `Dict` or a `Tuple`, then the corresponding entry will be a group instead of a dataset. Within this group, there will be nested groups and datasets, as specified by the action and observation spaces. `Dict` and `Tuple` spaces are represented as groups, and `Box` and `Discrete` spaces are represented as datasets. All datasets at any level under the top-level key `actions` will have the same `num_steps`, but will vary in `action_space_component_shape` on for each particular action space component. For example, a `Dict` space may contain two `Box` spaces with different shapes.
- `observations`: `shape=(num_steps + 1, observation_space_component_shape)`. Observations nest in the same way as actions if the top level space is a `Tuple` or `Dict` space. The value of `num_steps + 1` is the same for datasets at any level under `observations`. These datasets have an additional element because the initial observation of the environment when calling `obs, info = env.reset()` is also saved. `observation_space_component_shape` will vary between datasets, depending on the shapes of the simple spaces specified in the observation space.
- `rewards`: `shape=(num_steps, 1)`, stores the returned reward in each step.
- `terminations`: `shape=(num_steps, 1)`, the `dtype` is `np.bool` and the last element value will be `True` if the episode finished due to  a `terminated` step return.
- `truncations`: `shape=(num_steps, 1)`, the `dtype` is `np.bool` and the last element value will be `True` if the episode finished due to a `truncated` step return.

The `dtype` of the numpy array datasets can be of any type compatible with [`h5py`](https://docs.h5py.org/en/latest/faq.html#what-datatypes-are-supported).

The `info` dictionary returned in `env.step()` and `env.reset()` can be optionally saved in the dataset as a `sub-group`. The option to save the `info` data can be set in the `DataCollector` wrapper with the  `record_infos` argument.

Also, additional `datasets` and nested `sub-groups` can be saved in each episode. This can be the case of environment data that doesn't participate in each `env.step()` or `env.reset()` call in the Gymnasium API, such as the full environment state in each step. This can be achieved by creating a custom `StepDataCallback` that returns extra keys and nested dictionaries in the `StepData` dictionary return.

For example, the `Adroit Hand` environments in the `Gymnasium-Robotics` project need to store the full state of the MuJoCo simulation since this information is not present in the `observations` dataset and the environments are reset by setting an initial state in the simulation.

The following code snippet creates a custom `StepDataCallbak` and adds a new key, `state`, to the returned `StepData` dictionary. `state` is a nested dictionary with `np.ndarray` values and the keys are relevant MuJoCo data that represent the state of the simulation: `qpos`, `qvel`, and some other body positions.

```python
from minari import StepDataCallback
class AdroitStepDataCallback(StepDataCallback):
    def __call__(self, env, **kwargs):
        step_data = super().__call__(env, **kwargs)
        step_data['state'] = env.get_env_state()
        return step_data
```

The episode groups in the `HDF5` file will then have the following structure:

<div class="only-light">
<ul class="directory-list">
<li class="folder">episode_id
    <ul>
        <li class="dataset">observations</li>
        <li class="dataset">actions</li>
        <li class="dataset">terminations</li>
        <li class="dataset">truncations</li>
        <li class="dataset">rewards</li>
        <li class="folder-closed">infos
        <li class="folder">state
        <ul>
            <li class="dataset">qpos</li>
            <li class="dataset">qvel</li>
            <li class="dataset">object_body_pos</li>
        </ul>
    </ul>
</li>
</ul>
</div>

<div class="only-dark">
<ul class="directory-list">
<li class="folder-white">episode_id
    <ul class="white">
        <li class="dataset-white">observations</li>
        <li class="dataset-white">actions</li>
        <li class="dataset-white">terminations</li>
        <li class="dataset-white">truncations</li>
        <li class="dataset-white">rewards</li>
        <li class="folder-white-closed">infos
        <li class="folder-white">state
        <ul class="white">
            <li class="dataset-white">qpos</li>
            <li class="dataset-white">qvel</li>
            <li class="dataset-white">object_body_pos</li>
        </ul>
    </ul>
</li>
</ul>
</div>

### Default dataset metadata
`HDF5` files can have metadata attached to `objects` as [`attributes`](https://docs.h5py.org/en/stable/high/attr.html). Minari uses these `attributes` to add metadata to the global dataset file, to each episode group, as well as to the individual datasets inside each episode. This                                                                                 metadata can be added by the user by overriding the `EpisodeMetadataCallback` in the `DataCollector` wrapper. However, there is also some metadata added by default to every dataset.

When creating a Minari dataset with the `DataCollector` wrapper the default global metadata will be the following:

| Attribute               | Type       | Description |
| ----------------------- | ---------- | ----------- |
| `total_episodes`        | `np.int64` | Number of episodes in the Minari dataset. |
| `total_steps`           | `np.int64` | Number of steps in the Minari dataset. |
| `env_spec`              | `str`      | json string of the Gymnasium environment spec.|
| `dataset_id`            | `str`      | Identifier of the Minari dataset. |
| `code_permalink`        | `str`      | Link to a repository with the code used to generate the dataset.|
| `author`                | `str`      | Author's name that created the dataset. |
| `author_email`          | `str`      | Email of the author that created the dataset.|
| `algorithm_name`        | `str`      | Name of the expert policy used to create the dataset. |
| `action_space`          | `str`      | Serialized Gymnasium action space describing actions in dataset. |
| `observation_space`     | `str`      | Serialized Gymnasium observation space describing observations in dataset. |
| `minari_version`        | `str`      | Version specifier of Minari versions compatible with the dataset. |



For each episode group the default metadata `attributes` are:

| Attribute     | Type       | Description                             |
| ------------- | ---------- | --------------------------------------- |
| `id`          | `np.int64` | ID of the episode, `episode_id`. |
| `total_steps` | `np.int64` | Number of steps in the episode.         |
| `seed`        | `np.int64` | Seed used to reset the episode.         |

Statistical metrics are also computed as metadata for the individual datasets in each episode (for now only computed for `rewards` dataset)

- `rewards` dataset:

    | Metric | Type         | Description                                |
    | ------ | ------------ | ------------------------------------------ |
    | `max`  | `np.float64` | Maximum reward value in the episode.       |
    | `min`  | `np.float64` | Minimum reward value in the episode.       |
    | `mean` | `np.float64` | Mean value of the episode rewards.         |
    | `std`  | `np.float64` | Standard deviation of the episode rewards. |
    | `sum`  | `np.float64` | Total undiscounted return of the episode.  |

## Observation and Action Spaces
The Minari storage format supports the following observation and action spaces:

### Supported Spaces

| Space                                                                                 | Description                                                                                              |
| ------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| [Discrete](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/spaces/discrete.py) |Describes a discrete space where `{0, 1, ..., n-1}` are the possible values our observation can take. An optional argument can be used to shift the values to `{a, a+1, ..., a+n-1}`.|
| [Box](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/spaces/box.py)           |An n-dimensional continuous space. The `upper` and `lower` arguments can be used to define bounded spaces.|
| [Tuple](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/spaces/tuple.py)       |Represents a tuple of spaces.                                                                             |
| [Dict](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/spaces/dict.py)         |Represents a dictionary of spaces.                                                                        |
| [Text](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/spaces/text.py)         |The elements of this space are bounded strings from a charset. Note: at the moment, we don't guarantee support for all surrogate pairs.                                                                        |                                                                       |

#### Space Serialization
Spaces are serialized to a JSON format when saving to disk. This serialization supports all space types supported by Minari, and aims to be both human, and machine readable. The serialized action and observation spaces for the episodes in the dataset are saved as strings in the global HDF5 group metadata in `main_data.hdf5` for a particular dataset as `action_space` and `observation_space` respectively. All episodes in `main_data.hdf5` must have observations and actions that comply with these action and observation spaces.

## Minari Data Structures

A Minari dataset is encapsulated in the `MinariDataset` class which allows for iterating and sampling through episodes which are defined as `EpisodeData` data class.

### EpisodeData Structure

Episodes can be accessed from a Minari dataset through iteration, random sampling, or even filtering episodes from a dataset through an arbitrary condition via the `filter_episodes` method. Take the following example where we load the `door-human-v0` dataset and randomly sample 10 episodes:

```python
import minari
dataset = minari.load_dataset("door-human-v1", download=True)
sampled_episodes = dataset.sample_episodes(10)
```

The `sampled_episodes` variable will be a list of 10 `EpisodeData` elements, each containing episode data. An `EpisodeData` element is a data class consisting of the following fields:

| Field             | Type                                 | Description                                                   |
| ----------------- | ------------------------------------ | ------------------------------------------------------------- |
| `id`              | `np.int64`                           | ID of the episode.                                            |
| `seed`            | `np.int64`                           | Seed used to reset the episode.                               |
| `total_timesteps` | `np.int64`                           | Number of timesteps in the episode.                           |
| `observations`    | `np.ndarray`, `list`, `tuple`, `dict` | Observations for each timestep including initial observation. |
| `actions`         | `np.ndarray`, `list`, `tuple`, `dict` | Actions for each timestep.                                    |
| `rewards`         | `np.ndarray`                         | Rewards for each timestep.                                    |
| `terminations`    | `np.ndarray`                         | Terminations for each timestep.                               |
| `truncations`     | `np.ndarray`                         | Truncations for each timestep.                                |

As mentioned in the `Supported Spaces` section, many different observation and action spaces are supported so the data type for these fields are dependent on the environment being used.

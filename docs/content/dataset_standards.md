---
layout: "contents"
title: Dataset Standards
---

# Dataset Standards


## Minari Storage

### Minari root

Minari stores the offline datasets under a common root directory. The root directory path for the local datasets is set by default to `~/.minari/datasets/`. However, this path can be modified by setting the environment variable `MINARI_DATASETS_PATH`.

The remote datasets are kept in the public Google Cloud Platform (GCP) bucket [`minari-datasets`](https://console.cloud.google.com/storage/browser/minari-datasets;tab=objects?forceOnBucketsSortingFiltering=false&project=mcmes-345620&prefix=&forceOnObjectsSortingFiltering=false).

The first level of the root directory tree contains the Minari dataset directories, which are names as the datasets `id` name. The datasets `id` must follow the syntax `(env_name-)(dataset_name)(-v(version))`, where:

- `env_name`: a string that describes the environment from which the dataset was created. If a dataset comes from the [`AdroitHandDoor`](https://robotics.farama.org/envs/adroit_hand/adroit_door/) environment `ennv_name` can be equal to `door`.
- `dataset_name`: a string describing the content of the dataset. For example, if the dataset for the `AdroitHandDoor` environment was generated from human input we can give the value `human` to `dataset_name`.
- `version`: integer value that represent the number of versions for `door-human-v(version)` dataset, starting from `0`.

In the end the first version for the example `AdroitHandDoor` dataset will be `door-human-v0`.

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

The stepping data inside the episode group is divided into some required `datasets` (`StepData`) plus other optional `groups` and nested `sub-groups` such as `infos`. The hierarchical tree of the Minari dataset `HDF5` file will end up looking as follows:

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

The required `datasets` found in the episode groups correspond to the data involved in every Gymnasium step call `obs, rew, terminated, truncated, info = env.step(action)`: `observations`, `actions`, `rewards`, `terminations`, and `truncations`. These datasets are `np.ndarray` and their shape is equal to:

- `actions`: `shape=(number_of_steps, action_space_shape)`. If the action space is a `Dictionary` or a `Tuple` each step action is flatten before creating the `actions` dataset (currently `Sequence` and `Graph` action spaces are not supported). If using the `DataCollectorv0` wrapper to create the Mianri datasets, the saved actions will be automatically flattened by the `StepDataCallback`.
- `observations`: `shape=(number_of_steps + 1, observation_space_shape)`. The observations are also flattened if the observation space of the environment is of types `Dictionary` or `Tuple`. The size of the first axis of the `observations` dataset has an additional element because the initial observation of the environment when calling `obs, info = env.reset()` is also saved. You can get a transition of the form `(o_t, a_t, o_t+1)` from the datasets in the episode group, where `o_t` is the current observation, `o_t+1` is the next observation after taking action `a`, and `t` is the discrete transition index
; as follows:

    ```python
    next_observations = observations[1:]
    observations = observations[:-1]

    # get transition at timestep t
    observation = observations[t]             # o_t
    action = actions[t]                       # a_t
    next_observation = next_observations[t]   # o_t+1
    reward = rewards[t]                       # r_t
    terminated = terminations[t]
    truncated = truncations[t]
    ```

- `rewards`: `shape=(number_of_steps, 1)`, stores the returned reward in each step.
- `terminations`: `shape=(number_of_steps, 1)`, the `dtype` is `np.bool` and the last element value will be `True` if the episode finished due to  a `terminated` step return.
- `truncations`: `shape=(number_of_steps, 1)`, the `dtype` is `np.bool` and the last element value will be `True` if the episode finished due to a `truncated` step return.  

The `dtype` of the numpy array datasets can be of any type compatible with [`h5py`](https://docs.h5py.org/en/latest/faq.html#what-datatypes-are-supported).

The `info` dictionary returned in `env.step()` and `env.reset()` can be optionally saved in the dataset as a `sub-group`. The option to save the `info` data can be set in the `DataCollectorv0` wrapper with the  `record_infos` argument.

Also, additional `datasets` and nested `sub-groups` can be saved in each episode. This can be the case of environment data that doesn't participate in each `env.step()` or `env.reset()` call in the Gymnasium API, such as the full environment state in each step. This can be achieved by creating a custom `StepDataCallback` that returns extra keys and nested dictionaries in the `StepData` dictionary return.

For example, the `Adroit Hand` environments in the `Gymnasium-Robotics` project need to store the full state of the MuJoCo simulation since this information is not present in the `observations` dataset and the environments are reset by setting an initial state in the simulation.

The following code snippet creates a custom `StepDataCallbak` and adds a new key, `state`, to the returned `StepData` dictionary. `state` is a nested dictionary with `np.ndarray` values and the keys are relevant MuJoCo data that represent the state of the simulation: `qpos`, `qvel`, and some other body positions.

```python
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
`HDF5` files can have metadata attached to `objects` as [`attributes`](https://docs.h5py.org/en/stable/high/attr.html). Minari uses these `attributes` to add metadata to the global dataset file, to each episode group, as well as to the individual datasets inside each episode. This                                                                                 metadata can be added by the user by overriding the `EpisodeMetadataCallback` in the `DataCollectorV0` wrapper. However, there is also some metadata added by default to every dataset.

When creating a Minari dataset with the `DataCollectorV0` wrapper the default global metadata will be the following:

| Attribute               | Type       | Description |
| ----------------------- | ---------- | ----------- |
| `total_episodes`        | `np.int64` | Number of episodes in the Minari dataset. |
| `total_steps`           | `np.int64` | Number of steps in the Minari dataset. |
| `flattened_observation` | `np.bool`  | If the observation space had to be flattened. Usually for `Dictionary` and `Tuple` spaces. |
| `flattened_action`      | `np.bool`  | If the action space had to be flattened. Usually for `Dictionary` and `Tuple` spaces.|
| `env_spec`              | `str`      | json string of the Gymnasium environment spec.|
| `dataset_name`          | `str`      | Name tag of the Minari dataset. |
| `code_permalink`        | `str`      | Link to a repository with the code used to generate the dataset.|
| `author`                | `str`      | Author's name that created the dataset. |
| `author_email`          | `str`      | Email of the author that created the dataset.|
| `algorithm_name`        | `str`      | Name of the expert policy used to create the dataset. |

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

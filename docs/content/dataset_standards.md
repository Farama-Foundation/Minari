---
layout: "contents"
title: Dataset Standards
---

# Dataset Standards

## Minari Dataset Directory

Minari stores the offline datasets under a common root directory. The root directory path for the local datasets is set by default to `~/.minari/datasets/`. However, this path can be modified by setting the environment variable `MINARI_DATASETS_PATH`.

The remote datasets are kept in the public Google Cloud Platform (GCP) bucket [`minari-datasets`](https://console.cloud.google.com/storage/browser/minari-datasets;tab=objects?forceOnBucketsSortingFiltering=false&project=mcmes-345620&prefix=&forceOnObjectsSortingFiltering=false).

The first level of the root directory tree contains the Minari dataset directories, which are named after the datasets `id`. The datasets `id` must follow the syntax `(env_name-)(dataset_name)(-v(version))`, where:

- `env_name`: a string that describes the environment from which the dataset was created. For example, if a dataset comes from the [`AdroitHandDoor`](https://robotics.farama.org/envs/adroit_hand/adroit_door/) environment `env_name` can be equal to `door`.
- `dataset_name`: a string describing the content of the dataset. For example, if the dataset for the `AdroitHandDoor` environment was generated from human input we can give the value `human` to `dataset_name`.
- `version`: integer value that represent the number of versions for `door-human-v(version)` dataset, starting from `0`.

In the end, the `id` of the dataset for the initial version of the `AdroitHandDoor` environment example will be `door-human-v0`.

```{eval-rst}
Each Minari dataset directory contains another directory named `data` where the data files are stored. We currently support two files format: Arrow and HDF5. You can choose the file format when you create the dataset (see :class:`minari.DataCollector` and :func:`minari.create_dataset_from_buffers`).
```

## Dataset Metadata
Datasets can have metadata attached to them, as well as metadata for each episode. The dataset metadata can be specified by the user during dataset creation. On the other hand, the metadata for each episode can be added by the user by overriding the `EpisodeMetadataCallback` function in the `DataCollector` wrapper.

When creating a Minari dataset the default global metadata will be the following:

| Attribute               | Type       | Description |
| ----------------------- | ---------- | ----------- |
| `dataset_id`            | `str`      | Identifier of the Minari dataset. |
| `total_episodes`        | `int` | Number of episodes in the Minari dataset. |
| `total_steps`           | `int` | Number of steps in the Minari dataset. |
| `action_space`          | `gymnasium.Space`      | Gymnasium action space describing actions in dataset. |
| `observation_space`     | `gymnasium.Space`      | Gymnasium observation space describing observations in dataset. |
| `env_spec`              | `str`      | JSON string of the Gymnasium environment spec.|
| `code_permalink`        | `str`      | Link to a repository with the code used to generate the dataset.|
| `author`                | `str`      | Author's name that created the dataset. |
| `author_email`          | `str`      | Email of the author that created the dataset.|
| `algorithm_name`        | `str`      | Name of the expert policy used to create the dataset. |
| `minari_version`        | `str`      | Version specifier of Minari versions compatible with the dataset. |


where only `dataset_id`, `total_episodes`, and `total_steps` are mandatory (with the latter two computed automatically).

For each episode group the default metadata attributes are:

| Metric | Type         | Description                                |
| ------ | ------------ | ------------------------------------------ |
| `rewards_max`  | `float` | Maximum reward value in the episode.       |
| `rewards_min`  | `float` | Minimum reward value in the episode.       |
| `rewards_mean` | `float` | Mean value of the episode rewards.         |
| `rewards_std`  | `float` | Standard deviation of the episode rewards. |
| `rewards_sum`  | `float` | Total undiscounted return of the episode.  |


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

Spaces are serialized to a JSON format when saving to disk. This serialization supports all space types supported by Minari, and aims to be both human, and machine readable.

## EpisodeData Structure

A Minari dataset is encapsulated in the `MinariDataset` class which allows for iterating and sampling through episodes which are defined as `EpisodeData` data class. Take the following example where we load the `door-human-v2` dataset and randomly sample 10 episodes:

```python
import minari
dataset = minari.load_dataset("door-human-v2", download=True)
sampled_episodes = dataset.sample_episodes(10)
```

The `sampled_episodes` variable will be a list of 10 `EpisodeData` elements, each containing episode data. An `EpisodeData` element is a data class consisting of the following fields:

| Field             | Type                                 | Description                                                   |
| ----------------- | ------------------------------------ | ------------------------------------------------------------- |
| `id`              | `int`                           | ID of the episode.                                            |
| `seed`            | `int`                           | Seed used to reset the episode.                               |
| `total_steps`     | `int`                           | Number of steps in the episode.                               |
| `observations`    | `np.ndarray`, `list`, `tuple`, `dict` | Stacked observations for each step including initial observation.    |
| `actions`         | `np.ndarray`, `list`, `tuple`, `dict` | Stacked actions for each step.                                       |
| `rewards`         | `np.ndarray`                         | Rewards for each step.                                        |
| `terminations`    | `np.ndarray`                         | Terminations for each step.                                   |
| `truncations`     | `np.ndarray`                         | Truncations for each step.                                    |
| `infos`           | `dict`                               | A dictionary containing additional information returned by the environment             |

As mentioned in the `Supported Spaces` section, many different observation and action spaces are supported so the data type for these fields are dependent on the environment being used.

Moreover, when creating a dataset with `DataCollector`, if the `DataCollector` is initialized with `record_infos=True`, an info dict must be provided from every call to the environment's `step` and `reset` function. The structure of the info dictionary must be the same across steps. Given that it is not guaranteed that all Gymnasium environments provide infos at every step, we provide the `StepDataCallback` which can modify the infos from a non-compliant environment so they have the same structure at every step.

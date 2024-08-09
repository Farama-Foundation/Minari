---
layout: "contents"
title: Dataset Standards
---

# Dataset Standards

## Minari Dataset Directory

Minari stores the offline datasets under a common root directory. The root directory path for the local datasets is set by default to `~/.minari/datasets/`. However, this path can be modified by setting the environment variable `MINARI_DATASETS_PATH`.

The remote datasets are kept in the public Google Cloud Platform (GCP) bucket [`minari-remote`](https://console.cloud.google.com/storage/browser/minari-remote).

Minari dataset directories are named after the datasets `id`. The datasets `id` must follow the syntax `(namespace/)(env_name/)dataset_name(-v(version))`, where:

- `namespace`: an optional string that allows grouping of several datasets under a common subdirectory. If no namespace is specified, the dataset is stored in the top level directory and the dataset `id` takes the form `(env_name/)dataset_name(-v(version))`, with no leading forward slash. Namespaces can be arbitrarily nested.
- `env_name`: an optional string that describes the environment from which the dataset was created. For example, if a dataset comes from the [`AdroitHandDoor`](https://robotics.farama.org/envs/adroit_hand/adroit_door/) environment `env_name` can be equal to `door`. The `env_name` is a namespace group we encourage to use when the dataset is generated from an environment.
- `dataset_name`: a string describing the content of the dataset. For example, if the dataset for the `AdroitHandDoor` environment was generated from human input we can give the value `human` to `dataset_name`.
- `version`: integer value that represent the number of versions for `door-human-v(version)` dataset, starting from `0`.

In the end, the `id` of the dataset for the initial version of the `AdroitHandDoor` environment example, with no namespace, will be `door/human-v0`. If the dataset was created with a namespace, for example `D4RL`, the dataset `id` would instead be `D4RL/door/human-v0`.

Minari dataset directories are stored in the root directory (`~/.minari/datasets/` by default) or in a subdirectory corresponding to the namespace of the dataset if a namespace is specified.


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
| `author`                | `set of str`      | Name of the authors that created the dataset. |
| `author_email`          | `set of str`      | Email of the authors that created the dataset.|
| `algorithm_name`        | `str`      | Name of the expert policy used to create the dataset. |
| `minari_version`        | `str`      | Version of Minari that generated the dataset. |
| `requirements`          | `list of str`      | List of requirements in pip-style to load the environment. |


where only `dataset_id`, `total_episodes`, and `total_steps` are mandatory (with the latter two computed automatically).

For each episode group the default metadata attributes are:

| Metric | Type         | Description                                |
| ------ | ------------ | ------------------------------------------ |
| `rewards_max`  | `float` | Maximum reward value in the episode.       |
| `rewards_min`  | `float` | Minimum reward value in the episode.       |
| `rewards_mean` | `float` | Mean value of the episode rewards.         |
| `rewards_std`  | `float` | Standard deviation of the episode rewards. |
| `rewards_sum`  | `float` | Total undiscounted return of the episode.  |

## Namespace metadata

Namespaces can have metadata associated to them, to describe the group of datasets that they contain. Arbitrary JSON-serializable metadata is supported, and is stored in the file `namespace_metadata.json` in the appropriate namespace directory.


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

A Minari dataset is encapsulated in the `MinariDataset` class which allows for iterating and sampling through episodes which are defined as `EpisodeData` data class. Take the following example where we load the `D4RL/door/human-v2` dataset and randomly sample 10 episodes:

```python
import minari
dataset = minari.load_dataset("D4RL/door/human-v2", download=True)
sampled_episodes = dataset.sample_episodes(10)
```

The `sampled_episodes` variable will be a list of 10 `EpisodeData` elements, each containing episode data. An `EpisodeData` element is a data class consisting of the following fields:

| Field             | Type                                 | Description                                                   |
| ----------------- | ------------------------------------ | ------------------------------------------------------------- |
| `id`              | `int`                           | ID of the episode.                                            |
| `observations`    | `np.ndarray`, `list`, `tuple`, `dict` | Stacked observations for each step including initial observation.    |
| `actions`         | `np.ndarray`, `list`, `tuple`, `dict` | Stacked actions for each step.                                       |
| `rewards`         | `np.ndarray`                         | Rewards for each step.                                        |
| `terminations`    | `np.ndarray`                         | Terminations for each step.                                   |
| `truncations`     | `np.ndarray`                         | Truncations for each step.                                    |
| `infos`           | `dict`                               | A dictionary containing additional information returned by the environment             |

As mentioned in the `Supported Spaces` section, many different observation and action spaces are supported so the data type for these fields are dependent on the environment being used.

Moreover, when creating a dataset with `DataCollector`, if the `DataCollector` is initialized with `record_infos=True`, an info dict must be provided from every call to the environment's `step` and `reset` function. The structure of the info dictionary must be the same across steps. Given that it is not guaranteed that all Gymnasium environments provide infos at every step, we provide the `StepDataCallback` which can modify the infos from a non-compliant environment so they have the same structure at every step.

Other optional attributes of the episode, such as reset `seed` and `options`, can be found in the episode metadata using `MinariStorage.get_episode_metadata`.

import io
import json
import warnings
from collections import OrderedDict
from typing import Union

import gymnasium as gym
import numpy as np
from datasets import Dataset, DatasetInfo, load_dataset
from huggingface_hub import hf_hub_download, upload_file, whoami

import minari
from minari import MinariDataset
from minari.serialization import deserialize_space, serialize_space


def _reconstuct_obs_or_action_at_index_recursive(
    data: Union[dict, tuple, np.ndarray], index: int
) -> Union[np.ndarray, dict]:
    if isinstance(data, dict):
        return {
            key: _reconstuct_obs_or_action_at_index_recursive(data[key], index)
            for key in data.keys()
        }
    elif isinstance(data, tuple):
        return {
            f"_index_{str(index)}": _reconstuct_obs_or_action_at_index_recursive(
                entry, index
            )
            for index, entry in enumerate(data)
        }

    elif isinstance(data, np.ndarray):
        return data[index]
    else:
        assert False, f"error, invalid observation or action structure{data}"


def convert_minari_dataset_to_hugging_face_dataset(dataset: MinariDataset):
    """Converts a MinariDataset into a HuggingFace datasets dataset."""
    episodes = [episode for episode in dataset.iterate_episodes()]
    episodes_dict = {
        "observations": [],
        "actions": [],
        "rewards": [],
        "truncations": [],
        "terminations": [],
        "episode_ids": [],
    }
    for episode in episodes:
        episodes_dict["observations"].extend(
            [
                _reconstuct_obs_or_action_at_index_recursive(episode.observations, i)
                for i in range(episode.total_timesteps + 1)
            ]
        )
        episodes_dict["actions"].extend(
            [
                _reconstuct_obs_or_action_at_index_recursive(episode.actions, i)
                for i in range(episode.total_timesteps)
            ]
            + [
                None,
            ]
        )
        episodes_dict["rewards"].extend(
            list(episode.rewards)
            + [
                None,
            ]
        )
        episodes_dict["truncations"].extend(
            list(episode.truncations)
            + [
                None,
            ]
        )
        episodes_dict["terminations"].extend(
            list(episode.terminations)
            + [
                None,
            ]
        )
        episodes_dict["episode_ids"].extend(
            [episode.id for i in range(episode.total_timesteps + 1)]
        )

    description_json_str = json.dumps(
        {
            "dataset_id": dataset.spec.dataset_id,
            "env_name": dataset.spec.env_spec.id,
            "action_space": serialize_space(dataset.spec.action_space),
            "observation_space": serialize_space(dataset.spec.observation_space),
        }
    )

    hugging_face_dataset = Dataset.from_dict(
        episodes_dict, info=DatasetInfo(description=description_json_str)
    )
    return hugging_face_dataset


def _cast_to_numpy_recursive(
    space: gym.spaces.Space, entry: Union[tuple, dict, list]
) -> Union[OrderedDict, np.ndarray, tuple]:
    """Recurses on an observation or action space, and mirrors the recursion on an observation or action, casting all components to numpy arrays."""
    if isinstance(space, gym.spaces.Dict) and isinstance(entry, dict):
        result = OrderedDict()
        for key in space.spaces.keys():
            result[key] = _cast_to_numpy_recursive(space.spaces[key], entry[key])
        return result
    elif isinstance(space, gym.spaces.Tuple) and isinstance(
        entry, dict
    ):  # we substitute tuples with dicts in the hugging face dataset
        result = []  # with keys corresponding to the elements index in the tuple.
        for i in range(len(space.spaces)):
            result.append(
                _cast_to_numpy_recursive(space.spaces[i], entry[f"_index_{str(i)}"])
            )
        return tuple(result)
    elif isinstance(space, gym.spaces.Discrete):
        return np.asarray(entry, dtype=space.dtype)
    elif isinstance(space, gym.spaces.Box):
        return np.asarray(entry, dtype=space.dtype)
    else:
        raise TypeError(
            f"{type(space)} is not supported. or there is type mismatch with the entry type {type(entry)}"
        )


def convert_hugging_face_dataset_to_minari_dataset(dataset: Dataset):

    description_data = json.loads(dataset.info.description)

    action_space = deserialize_space(description_data["action_space"])
    observation_space = deserialize_space(description_data["observation_space"])
    env_name = description_data["env_name"]
    dataset_id = description_data["dataset_id"]

    episode_ids = dataset.unique("episode_ids")

    buffer = []

    for episode_id in episode_ids:
        episode_rows = dataset.filter(
            lambda row: row["episode_ids"] == episode_id
        ).to_dict()
        del episode_rows["episode_ids"]
        episode_rows["actions"] = episode_rows["actions"][:-1]
        episode_rows["rewards"] = np.asarray(episode_rows["rewards"][:-1])
        episode_rows["terminations"] = np.asarray(episode_rows["terminations"][:-1])
        episode_rows["truncations"] = np.asarray(episode_rows["truncations"][:-1])

        episode_rows["actions"] = [
            _cast_to_numpy_recursive(action_space, action)
            for action in episode_rows["actions"]
        ]
        episode_rows["observations"] = [
            _cast_to_numpy_recursive(observation_space, observation)
            for observation in episode_rows["observations"]
        ]

        buffer.append(OrderedDict(episode_rows))

    # Create Minari dataset and store locally
    dataset = minari.create_dataset_from_buffers(
        dataset_id=dataset_id,
        env=gym.make(env_name),
        buffer=buffer,
        algorithm_name="random_policy",
        code_permalink="https://github.com/Farama-Foundation/Minari/blob/f095bfe07f8dc6642082599e07779ec1dd9b2667/tutorials/LocalStorage/local_storage.py",
        author="WillDudley",
        author_email="wdudley@farama.org",
        action_space=action_space,
        observation_space=observation_space,
    )
    return dataset


def push_dataset_to_hugging_face(dataset: Dataset, path: str, private: bool = True):
    """Pushes a huggingface dataset to the HuggingFace repository at the specified path."""
    try:
        whoami()
    except OSError:
        warnings.warn(
            "Please log in using the huggingface-hub cli in order to push to a remote dataset."
        )
        return

    metadata_file = io.BytesIO(dataset.info.description.encode("utf-8"))

    upload_file(
        path_or_fileobj=metadata_file,
        path_in_repo="metadata.json",
        repo_id=path,
        repo_type="dataset",
    )

    dataset.push_to_hub(path, private=private)


def pull_dataset_from_hugging_face(path: str) -> Dataset:
    """Pulls a hugging face dataset from the HuggingFace repository at the specified path."""
    hugging_face_dataset = load_dataset(path)

    with open(
        hf_hub_download(filename="metadata.json", repo_id=path, repo_type="dataset"),
    ) as metadata_file:
        metadata_str = metadata_file.read()
    hugging_face_dataset["train"].info.description = metadata_str

    return hugging_face_dataset["train"]

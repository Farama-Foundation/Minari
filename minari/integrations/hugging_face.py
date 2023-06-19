import warnings
from typing import Union

import numpy as np
from datasets import Dataset
from huggingface_hub import whoami

from minari import MinariDataset
from minari.serialization import serialize_space


def _reconstuct_obs_or_action_at_index_recursive(
    data: Union[dict, tuple, np.ndarray], index: int
) -> Union[np.ndarray, dict, tuple]:
    if isinstance(data, dict):
        return {
            key: _reconstuct_obs_or_action_at_index_recursive(data[key], index)
            for key in data.keys()
        }
    elif isinstance(data, tuple):
        return tuple(
            [
                _reconstuct_obs_or_action_at_index_recursive(entry, index)
                for entry in data
            ]
        )

    elif isinstance(data, np.ndarray):
        return data[index]
    else:
        assert False, f"error, invalid observation or action structure{data}"


def convert_minari_dataset_to_hugging_face_dataset(dataset: MinariDataset):
    """Converts a MinariDataset into a HuggingFace datasets dataset."""
    episodes = [episode for episode in dataset.iterate_episodes()]
    episodes_dict ={
        "observation":[],
        "action":[],
        "reward":[],
        "truncation":[],
        "termination":[],
        "episode_id":[],
    }
    for episode in episodes:
        episodes_dict["observation"].extend([_reconstuct_obs_or_action_at_index_recursive(episode.observations, i)
                for i in range(episode.total_timesteps + 1)])
        episodes_dict["action"].extend([_reconstuct_obs_or_action_at_index_recursive(episode.actions, i)
                for i in range(episode.total_timesteps )] + [None,])
        episodes_dict["reward"].extend(list(episode.rewards) + [None,])
        episodes_dict["truncation"].extend(list(episode.truncations) + [None,])
        episodes_dict["termination"].extend(list(episode.terminations) + [None,])
        episodes_dict["episode_id"].extend([episode.id for i in range(episode.total_timesteps + 1)])

           
    hugging_face_dataset = Dataset.from_dict(episodes_dict)
    print(dir(hugging_face_dataset))
    print(hugging_face_dataset.features)
    #hugging_face_dataset.metadata["action_space"] = serialize_space(dataset.spec.action_space)
    #hugging_face_dataset.metadata["observation_space"] = serialize_space(dataset.spec.observation_space)
    # push_dataset_to_hugging_face(dataset, "???/minari_test")
    return dataset


def convert_hugging_face_dataset_to_minari_dataset(dataset: Dataset):
    episodes = dataset.unique("episode_id")
    
    for episode in episodes:
        pass


    


def push_dataset_to_hugging_face(dataset: Dataset, path: str, private: bool = True):
    """Pushes a huggingface dataset to the HuggingFace repository at the specified path."""
    try:
        whoami()
    except OSError:
        warnings.warn(
            "Please log in using the huggingface-hub cli in order to push to a remote dataset."
        )
        return
    dataset.push_to_hub(path, private=private)

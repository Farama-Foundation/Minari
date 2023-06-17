import warnings

from typing import Union

import numpy as np
from datasets import Dataset
from huggingface_hub import whoami


from minari import MinariDataset


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
    episodes =  [episode for episode in dataset.iterate_episodes()]
    episode = episodes[0]
    episode_dict = {"observation": [_reconstuct_obs_or_action_at_index_recursive(episode.observations,i) for i in range(episode.total_timesteps + 1)],
            "action": [_reconstuct_obs_or_action_at_index_recursive(episode.actions,i) for i in range(episode.total_timesteps)] + [None,],
            "reward": list(episode.rewards) + [None,],
            "truncation": list(episode.truncations) + [None,],
            "termination": list(episode.terminations) + [None,]
    }
    dataset = Dataset.from_dict(episode_dict)
    #push_dataset_to_hugging_face(dataset, "???/minari_test")
    return dataset


def convert_hugging_face_dataset_to_minari_dataset(dataset: Dataset):
    pass


def push_dataset_to_hugging_face(dataset: Dataset, path: str, private:bool = True):
    """Pushes a huggingface dataset to the HuggingFace repository at the specified path."""
    try:
        whoami()
    except EnvironmentError:
        warnings.warn("Please log in using the huggingface-hub cli in order to push to a remote dataset.")
        return 
    dataset.push_to_hub(path, private = private)

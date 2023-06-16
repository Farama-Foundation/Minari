import warnings


from minari import MinariDataset
from datasets import Dataset
from huggingface_hub import whoami


def convert_minari_dataset_to_hugging_face_dataset(dataset: MinariDataset):
    """Converts a MinariDataset into a HuggingFace datasets dataset."""
    episodes =  [episode for episode in dataset.iterate_episodes()]
    episode = episodes[0]
    episode_dict = {"observation": episode.observations,
            "action": list(episode.actions) + [None,],
            "reward": list(episode.rewards) + [None,],
            "truncation": list(episode.truncations) + [None,],
            "termination": list(episode.terminations) + [None,]
    }
    dataset = Dataset.from_dict(episode_dict)
    push_dataset_to_hugging_face(dataset, "???/minari_test")
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

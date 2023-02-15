import os
import shutil

from minari.minari_dataset import MinariDataset
from minari.storage.datasets_root_dir import get_dataset_path


def load_dataset(dataset_name: str):
    """Retrieve Minari dataset from local database.

    Args:
        dataset_name (str): name id of Minari dataset

    Returns:
        MinariDataset
    """
    file_path = get_dataset_path(dataset_name)
    data_path = os.path.join(file_path, "data", "main_data.hdf5")
    return MinariDataset(data_path)


def list_local_datasets(verbose=True):
    """Get a list of all the Minari dataset names in the local database.

    Args:
        verbose (bool, optional): If True the dataset names will be shown in the command line. Defaults to True.

    Returns:
       list[str]: List of local Minari dataset name id's
    """
    datasets_path = get_dataset_path("")
    datasets = [dir_name for dir_name in os.listdir(datasets_path)]

    if verbose:
        print("Datasets found locally:")
        for dataset in datasets:
            print(dataset)

    return datasets


def delete_dataset(dataset_name: str):
    """Delete a Minari dataset from the local Minari database.

    Args:
        dataset_name (str): name id of the Minari dataset
    """
    dataset_path = get_dataset_path(dataset_name)
    shutil.rmtree(dataset_path)
    print(f"Dataset {dataset_name} deleted!")

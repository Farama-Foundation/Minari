import os
import shutil
from typing import Dict, Union

import h5py

from minari.dataset.minari_dataset import MinariDataset
from minari.storage.datasets_root_dir import get_dataset_path


def load_dataset(dataset_id: str):
    """Retrieve Minari dataset from local database.

    Args:
        dataset_id (str): name id of Minari dataset

    Returns:
        MinariDataset
    """
    file_path = get_dataset_path(dataset_id)
    data_path = os.path.join(file_path, "data", "main_data.hdf5")
    return MinariDataset(data_path)


def list_local_datasets() -> Dict[str, Dict[str, Union[str, int, bool]]]:
    """Get the ids and metadata of all the Minari datasets in the local database.

    Returns:
       Dict[str, Dict[str, str]]: keys the names of the Minari datasets and values the metadata
    """
    datasets_path = get_dataset_path("")
    dataset_ids = sorted(
        [
            dir_name
            for dir_name in os.listdir(datasets_path)
            if not dir_name.startswith(".")
        ]
    )

    local_datasets = {}
    for dst_id in dataset_ids:
        if "data" not in os.listdir(os.path.join(datasets_path, dst_id)):
            # Minari datasets must contain the data directory.
            continue

        main_file_path = os.path.join(datasets_path, dst_id, "data/main_data.hdf5")

        with h5py.File(main_file_path, "r") as f:
            metadata = dict(f.attrs.items())
            local_datasets[dst_id] = metadata

    return local_datasets


def delete_dataset(dataset_id: str):
    """Delete a Minari dataset from the local Minari database.

    Args:
        dataset_id (str): name id of the Minari dataset
    """
    dataset_path = get_dataset_path(dataset_id)
    shutil.rmtree(dataset_path)
    print(f"Dataset {dataset_id} deleted!")

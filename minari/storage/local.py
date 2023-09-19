import importlib.metadata
import os
import shutil
from typing import Dict, Union

from packaging.specifiers import SpecifierSet

from minari.dataset.minari_dataset import MinariDataset, parse_dataset_id
from minari.dataset.minari_storage import MinariStorage
from minari.storage import hosting
from minari.storage.datasets_root_dir import get_dataset_path


# Use importlib due to circular import when: "from minari import __version__"
__version__ = importlib.metadata.version("minari")


def load_dataset(dataset_id: str, download: bool = False):
    """Retrieve Minari dataset from local database.

    Args:
        dataset_id (str): name id of Minari dataset
        download (bool): if `True` download the dataset if it is not found locally. Default to `False`.

    Returns:
        MinariDataset
    """
    file_path = get_dataset_path(dataset_id)
    data_path = os.path.join(file_path, "data")

    if not os.path.exists(data_path):
        if not download:
            raise FileNotFoundError(
                f"Dataset {dataset_id} not found locally at {file_path}. Use download=True to download the dataset."
            )

        hosting.download_dataset(dataset_id)

    return MinariDataset(data_path)


def list_local_datasets(
    latest_version: bool = False,
    compatible_minari_version: bool = False,
) -> Dict[str, Dict[str, Union[str, int, bool]]]:
    """Get the ids and metadata of all the Minari datasets in the local database.

    Args:
        latest_version (bool): if `True` only the latest version of the datasets are returned i.e. from ['door-human-v0', 'door-human-v1`], only the metadata for v1 is returned. Default to `False`.
        compatible_minari_version (bool): if `True` only the datasets compatible with the current Minari version are returned. Default to `False`.

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

        data_path = os.path.join(datasets_path, dst_id, "data")
        metadata = MinariStorage(data_path).metadata
        if ("minari_version" not in metadata) or (
            compatible_minari_version
            and __version__ not in SpecifierSet(metadata["minari_version"])
        ):
            continue
        env_name, dataset_name, version = parse_dataset_id(dst_id)
        dataset = f"{env_name}-{dataset_name}"
        if latest_version:
            if dataset not in local_datasets or version > local_datasets[dataset][0]:
                local_datasets[dataset] = (version, metadata)
        else:
            local_datasets[dst_id] = metadata

    if latest_version:
        # Return dict = {'dataset_id': metadata}
        return dict(
            map(lambda x: (f"{x[0]}-v{x[1][0]}", x[1][1]), local_datasets.items())
        )
    else:
        return local_datasets


def delete_dataset(dataset_id: str):
    """Delete a Minari dataset from the local Minari database.

    Args:
        dataset_id (str): name id of the Minari dataset
    """
    dataset_path = get_dataset_path(dataset_id)
    shutil.rmtree(dataset_path)
    print(f"Dataset {dataset_id} deleted!")

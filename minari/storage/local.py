import importlib.metadata
import os
import pathlib
import shutil
import tempfile
import warnings
from typing import Dict, Iterable, Optional, Tuple, Union

from minari.dataset.minari_dataset import (
    MinariDataset,
    gen_dataset_id,
    parse_dataset_id,
)
from minari.dataset.minari_storage import MinariStorage
from minari.storage import hosting
from minari.storage.datasets_root_dir import get_dataset_path


# Use importlib due to circular import when: "from minari import __version__"
__version__ = importlib.metadata.version("minari")


def list_non_hidden_dirs(path: pathlib.Path) -> Iterable[str]:
    """List all non-hidden subdirectories."""
    for d in path.iterdir():
        if d.is_dir() and (not d.name.startswith(".")):
            yield d.name


def dataset_id_sort_key(dataset_id: str) -> Tuple[str, str, int]:
    """Key for sorting dataset ids first by namespace, and then alphabetically."""
    namespace, dataset_name, version = parse_dataset_id(dataset_id)
    namespace = "" if namespace is None else namespace
    return (namespace, dataset_name, version)


def load_dataset(dataset_id: str, download: bool = False):
    """Retrieve Minari dataset from local database.

    Args:
        dataset_id (str): name id of Minari dataset
        download (bool): if `True` download the dataset if it is not found locally, or download it again
            if the local copy cannot be loaded, e.g. when files are missing or the metadata is unreadable
            after an interrupted download. Damaged data files are only detected when episodes are read.
            Default to `False`.

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

    # Only storage-level errors mean the local copy is incomplete or corrupted, e.g. after an
    # interrupted download. Errors raised by MinariDataset (such as an unsupported dataset version)
    # are not fixed by downloading again, so they propagate unchanged.
    try:
        storage = MinariStorage.read(data_path)
    except ValueError as e:
        if not download:
            raise ValueError(
                f"Dataset {dataset_id} found locally at {file_path} could not be loaded, it may be incomplete or corrupted: {e}. "
                f"Use download=True or minari.download_dataset('{dataset_id}', force_download=True) to download it again, "
                f"or minari.delete_dataset('{dataset_id}') to remove it."
            ) from e
        warnings.warn(
            f"Dataset {dataset_id} found locally at {file_path} could not be loaded ({e}). Downloading it again."
        )
        _redownload_dataset(dataset_id, file_path)
        storage = MinariStorage.read(data_path)

    return MinariDataset(storage)


def _redownload_dataset(dataset_id: str, file_path: pathlib.Path):
    """Replace a broken local dataset with a fresh download.

    The broken copy is moved aside, not forced over, so that the normal compatibility checks of
    `download_dataset` run. If the download fails, e.g. because the dataset only exists locally
    or the remote is unreachable, the broken copy is put back.
    """
    backup_dir = pathlib.Path(
        tempfile.mkdtemp(prefix=f".{file_path.name}-backup-", dir=file_path.parent)
    )
    backup_path = backup_dir / file_path.name
    shutil.move(file_path, backup_path)
    try:
        hosting.download_dataset(dataset_id)
    except BaseException:
        if file_path.exists():
            shutil.rmtree(file_path)
        shutil.move(backup_path, file_path)
        shutil.rmtree(backup_dir)
        raise
    shutil.rmtree(backup_dir)


def list_local_datasets(
    latest_version: bool = False,
    compatible_minari_version: bool = False,
    prefix: Optional[str] = None,
) -> Dict[str, Dict[str, Union[str, int, bool]]]:
    """Get the ids and metadata of all the Minari datasets in the local database.

    Args:
        latest_version (bool): if `True` only the latest version of the datasets are returned i.e. from ['D4RL/door/human-v0', 'D4RL/door/human-v1`], only the metadata for v1 is returned. Default to `False`.
        compatible_minari_version (bool): if `True` only the datasets compatible with the current Minari version are returned. Default to `False`.

    Returns:
       Dict[str, Dict[str, str]]: keys the names of the Minari datasets and values the metadata
    """
    from minari import supported_dataset_versions

    datasets_path = get_dataset_path()
    dataset_ids = []

    def recurse_directories(base_path: pathlib.Path, namespace):
        parent_dir = base_path.joinpath(namespace)
        if not parent_dir.exists():
            return
        # TODO: metadata.json of HDF5 should stay in dataset root directory not /data
        # Then we can use metadata.json to check if it is a dataset
        if parent_dir.joinpath("data").exists():
            dataset_ids.append(namespace)
            return

        for dir_name in list_non_hidden_dirs(parent_dir):
            namespaced_dir_name = pathlib.Path(namespace, dir_name).as_posix()
            recurse_directories(base_path, namespaced_dir_name)

    recurse_directories(datasets_path, prefix or "")

    dataset_ids = sorted(dataset_ids, key=dataset_id_sort_key)

    local_datasets = {}
    for dst_id in dataset_ids:
        data_path = pathlib.Path(datasets_path, dst_id, "data").as_posix()
        try:
            metadata = MinariStorage.read_raw_metadata(data_path)
            metadata_id = metadata["dataset_id"]

            if dst_id != metadata_id:
                raise ValueError(
                    f"Namespace location '{dst_id}' does not match id '{metadata_id}'."
                )
        except Exception as e:
            warnings.warn(f"Misconfigured dataset named {dst_id}: {e}")
            continue

        if (
            compatible_minari_version
            and metadata["minari_version"] not in supported_dataset_versions
        ):
            continue

        namespace, dataset_name, version = parse_dataset_id(dst_id)
        dataset = gen_dataset_id(namespace, dataset_name)

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

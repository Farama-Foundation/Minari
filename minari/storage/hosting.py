from __future__ import annotations

import importlib.metadata
import os
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

from minari.dataset.minari_dataset import gen_dataset_id, parse_dataset_id
from minari.dataset.minari_storage import MinariStorage
from minari.storage.datasets_root_dir import get_dataset_path
from minari.storage.local import load_dataset
from minari.storage.remotes import get_cloud_storage


# Use importlib due to circular import when: "from minari import __version__"
__version__ = importlib.metadata.version("minari")


def upload_dataset(dataset_id: str, token: str):
    """Upload a Minari dataset to the remote Farama server.

    If you would like to upload a dataset please first get in touch with the Farama team at contact@farama.org.

    Args:
        dataset_id (str): name id of the local Minari dataset
        token (str): token used for authenticating to the remote storage.
            Notice, that for GCP, this is the path to the service account key file, while for Hugging Face, this is the API token.
    """
    # Avoid circular import
    from minari.namespace import list_remote_namespaces, upload_namespace

    remote_datasets = list_remote_datasets()
    if dataset_id in remote_datasets.keys():
        warnings.warn(
            f"Upload aborted. {dataset_id} is already in remote.", UserWarning
        )
        return

    cloud_storage = get_cloud_storage(token=token)
    namespace, _, _ = parse_dataset_id(dataset_id)
    if namespace is not None and namespace not in list_remote_namespaces():
        upload_namespace(namespace, token)

    datasets_to_upload = [dataset_id]
    while len(datasets_to_upload):
        dataset_id = datasets_to_upload.pop()

        print(f"Uploading dataset {dataset_id}")
        cloud_storage.upload_dataset(dataset_id)

        dataset = load_dataset(dataset_id)
        combined_datasets = dataset.spec.combined_datasets
        datasets_to_upload.extend(combined_datasets)


def download_dataset(dataset_id: str, force_download: bool = False):
    """Download dataset from remote Farama server.

    An error will be raised if the dataset version is not compatible with the local installed version of Minari.
    This error can be skipped and the download continued with the `force_download` argument. Also, with `force_download`,
    any local datasets that match the id of the downloading dataset will be overridden.

    Args:
        dataset_id (str): name id of the Minari dataset. It can also be a complete remote path, e.g. `hf://farama-minari/D4RL/door/human-v2`.
        force_download (bool): boolean flag for force downloading the dataset. Default Value = False
    """
    from minari import supported_dataset_versions
    from minari.namespace import (
        download_namespace_metadata,
        list_local_namespaces,
        namespace_hierarchy,
    )

    remote_path = None
    if "://" in dataset_id:
        remote_type, remote_path = dataset_id.split("://", maxsplit=1)
        remote_path, dataset_id = remote_path.split("/", maxsplit=1)
        remote_path = f"{remote_type}://{remote_path}"

    (
        namespace,
        dataset_name,
        dataset_version,
    ) = parse_dataset_id(dataset_id)
    dataset_versionless = gen_dataset_id(namespace, dataset_name)

    all_dataset_versions = list_remote_datasets(
        remote_path=remote_path, prefix=dataset_versionless
    )

    # 1. Check if there are any remote versions of the dataset at all
    if not all_dataset_versions:
        raise ValueError(
            f"Couldn't find any version for dataset {dataset_versionless} in the remote Farama server."
        )

    # 2. Check if there are any remote compatible versions with the local installed Minari version
    compatible_dataset_versions = [
        parse_dataset_id(ds_id)[-1]
        for ds_id, ds_metadata in all_dataset_versions.items()
        if ds_metadata.get("minari_version") in supported_dataset_versions
    ]

    if not compatible_dataset_versions:
        message = f"Couldn't find any compatible version of dataset {dataset_versionless} with the local installed version of Minari, {__version__}."
        if not force_download:
            raise ValueError(message)
        else:
            warnings.warn(message)

    # 3. Check that the dataset version exists
    if dataset_id not in all_dataset_versions:
        e_msg = f"The dataset version, {dataset_id}, doesn't exist in the remote Farama server."
        if not compatible_dataset_versions:
            raise ValueError(
                e_msg
                + f", and no other compatible versions with the local installed Minari version {__version__} were found."
            )
        else:
            raise ValueError(
                e_msg
                + f" We suggest you download the latest compatible version of this dataset: {gen_dataset_id(namespace, dataset_name, max(compatible_dataset_versions))}"
            )

    # 4. Check that the dataset version is compatible with the local installed Minari version
    if dataset_version not in compatible_dataset_versions:
        e_msg = (
            f"The version you are trying to download for dataset, {dataset_id}, is not compatible with "
            f"your local installed version of Minari, {__version__}."
        )
        if not force_download:
            raise ValueError(
                e_msg
                + f" You can download the latest compatible version of this dataset: {gen_dataset_id(namespace, dataset_name, max(compatible_dataset_versions))}."
            )
        # Only a warning and force download incompatible dataset
        elif compatible_dataset_versions:
            warnings.warn(
                e_msg
                + f" {dataset_id} will be FORCE download but you can download the latest compatible version of this dataset: {gen_dataset_id(namespace, dataset_name, max(compatible_dataset_versions))}."
            )

    # 5. Warning to recommend downloading the latest compatible version of the dataset
    elif dataset_version < max(compatible_dataset_versions):
        warnings.warn(
            f"We recommend you install a higher dataset version available and compatible with your local installed Minari version: {dataset_versionless}-v{max(compatible_dataset_versions)}."
        )

    file_path = get_dataset_path(dataset_id)
    if os.path.exists(file_path):
        if not force_download:
            warnings.warn(
                f"Skipping Download. Dataset {dataset_id} found locally at {file_path}, Use force_download=True to download the dataset again.\n"
            )
            return

    if namespace is not None and namespace not in list_local_namespaces():
        for parent_namespace in namespace_hierarchy(namespace):
            download_namespace_metadata(parent_namespace)

    print(f"\nDownloading {dataset_id} from Farama servers...")
    datasets_path = get_dataset_path()
    cloud_storage = get_cloud_storage(remote_path=remote_path)
    cloud_storage.download_dataset(dataset_id, datasets_path)
    print(f"\nDataset {dataset_id} downloaded to {file_path}")

    # Skip a force download of an incompatible dataset version
    if dataset_version in compatible_dataset_versions:
        data_path = file_path.joinpath("data")
        metadata = MinariStorage.read_raw_metadata(data_path)
        combined_datasets = metadata.get("combined_datasets", [])

        # If the dataset is a combination of other datasets download the subdatasets recursively
        if len(combined_datasets) > 0:
            print(
                f"\nDataset {dataset_id} is formed by a combination of the following datasets: "
            )
            for name in combined_datasets:
                print(f" * {name}")
            print("\nDownloading extra datasets ...")
            for dataset in combined_datasets:
                download_dataset(dataset_id=dataset)


def list_remote_datasets(
    remote_path: Optional[str] = None,
    prefix: Optional[str] = None,
    latest_version: bool = False,
    compatible_minari_version: bool = False,
) -> Dict[str, Dict[str, str]]:
    """Get the names and metadata of all the Minari datasets in the remote Farama server.

    Args:
        remote_path (str): path to the remote storage. If not specified, the Farama default storage will be used.
            You can also define this in the environment variable `MINARI_REMOTE`.
        prefix (str): prefix to filter the datasets. Default to None.
        latest_version (bool): if `True` only the latest version of the datasets are returned i.e. from ['D4RL/door/human-v0', 'D4RL/door/human-v1`], only the metadata for v1 is returned. Default to `False`.
        compatible_minari_version (bool): if `True` only the datasets compatible with the current Minari version are returned. Default to `False`.

    Returns:
       Dict[str, Dict[str, str]]: keys the names of the Minari datasets and values the metadata
    """
    from minari import supported_dataset_versions

    cloud_storage = get_cloud_storage(remote_path=remote_path)
    dataset_ids = cloud_storage.list_datasets(prefix=prefix)

    with ThreadPoolExecutor(max_workers=10) as executor:
        remote_metadatas = executor.map(cloud_storage.get_dataset_metadata, dataset_ids)

    remote_datasets = {}
    max_version = defaultdict(dict)
    for metadata in remote_metadatas:
        supported_dataset = metadata.get("minari_version") in supported_dataset_versions
        if compatible_minari_version and not supported_dataset:
            continue

        dataset_id = metadata["dataset_id"]
        remote_datasets[dataset_id] = metadata

        if latest_version:
            namespace, dataset_name, version = parse_dataset_id(dataset_id)
            old_version = max_version[namespace].get(dataset_name, version)
            max_version[namespace][dataset_name] = max(old_version, version)
            if old_version != max_version[namespace][dataset_name]:
                min_id = gen_dataset_id(
                    namespace, dataset_name, min(old_version, version)
                )
                del remote_datasets[min_id]

    return remote_datasets

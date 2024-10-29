from __future__ import annotations

import importlib.metadata
import json
import os
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

from minari.dataset.minari_dataset import gen_dataset_id, parse_dataset_id
from minari.dataset.minari_storage import METADATA_FILE_NAME, MinariStorage
from minari.storage.datasets_root_dir import get_dataset_path
from minari.storage.local import load_dataset
from minari.storage.remotes import get_cloud_storage


# Use importlib due to circular import when: "from minari import __version__"
__version__ = importlib.metadata.version("minari")


def upload_dataset(dataset_id: str, key_path: str):
    """Upload a Minari dataset to the remote Farama server.

    If you would like to upload a dataset please first get in touch with the Farama team at contact@farama.org.

    Args:
        dataset_id (str): name id of the local Minari dataset
        key_path (str): path to the credentials file.
    """
    # Avoid circular import
    from minari.namespace import list_remote_namespaces, upload_namespace

    remote_datasets = list_remote_datasets()
    if dataset_id in remote_datasets.keys():
        warnings.warn(
            f"Upload aborted. {dataset_id} is already in remote.", UserWarning
        )
        return

    cloud_storage = get_cloud_storage(key_path=key_path)

    datasets_to_upload = [dataset_id]
    while len(datasets_to_upload):
        dataset_id = datasets_to_upload.pop()
        namespace, _, _ = parse_dataset_id(dataset_id)

        if namespace is not None and namespace not in list_remote_namespaces():
            upload_namespace(namespace, key_path)

        print(f"Uploading dataset {dataset_id}")
        path = get_dataset_path(dataset_id)
        cloud_storage.upload_directory(path, dataset_id)

        dataset = load_dataset(dataset_id)
        combined_datasets = dataset.spec.combined_datasets
        datasets_to_upload.extend(combined_datasets)


def download_dataset(dataset_id: str, force_download: bool = False):
    """Download dataset from remote Farama server.

    An error will be raised if the dataset version is not compatible with the local installed version of Minari.
    This error can be skipped and the download continued with the `force_download` argument. Also, with `force_download`,
    any local datasets that match the id of the downloading dataset will be overridden.

    Args:
        dataset_id (str): name id of the Minari dataset
        force_download (bool): boolean flag for force downloading the dataset. Default Value = False
    """
    # Avoid circular import
    from minari.namespace import create_namespace, list_local_namespaces

    (
        namespace,
        dataset_name,
        dataset_version,
    ) = parse_dataset_id(dataset_id)

    all_dataset_versions = get_remote_dataset_versions(namespace, dataset_name)
    dataset_versionless = gen_dataset_id(namespace, dataset_name)

    # 1. Check if there are any remote versions of the dataset at all
    if not all_dataset_versions:
        raise ValueError(
            f"Couldn't find any version for dataset {dataset_versionless} in the remote Farama server."
        )

    # 2. Check if there are any remote compatible versions with the local installed Minari version
    compatible_dataset_versions = get_remote_dataset_versions(
        namespace,
        dataset_name,
        latest_version=False,
        compatible_minari_version=True,
    )

    if not compatible_dataset_versions:
        message = f"Couldn't find any compatible version of dataset {dataset_versionless} with the local installed version of Minari, {__version__}."
        if not force_download:
            raise ValueError(message)
        else:
            warnings.warn(message)

    # 3. Check that the dataset version exists
    if dataset_version not in all_dataset_versions:
        e_msg = f"The dataset version, {dataset_id}, doesn't exist in the remote Farama server."
        if not compatible_dataset_versions:
            raise ValueError(
                e_msg
                + f", and no other compatible versions with the local installed Minari version {__version__} were found."
            )
        else:
            raise ValueError(
                e_msg
                + f" We suggest you download the latest compatible version of this dataset: {dataset_versionless}-v{max(compatible_dataset_versions)}"
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
                + f" You can download the latest compatible version of this dataset: {dataset_versionless}-v{max(compatible_dataset_versions)}."
            )
        # Only a warning and force download incompatible dataset
        elif compatible_dataset_versions:
            warnings.warn(
                e_msg
                + f" {dataset_id} will be FORCE download but you can download the latest compatible version of this dataset: {dataset_versionless}-v{max(all_dataset_versions)}."
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
        create_namespace(namespace)

    print(f"\nDownloading {dataset_id} from Farama servers...")
    datasets_path = get_dataset_path("")
    cloud_storage = get_cloud_storage()
    blobs = cloud_storage.list_blobs(prefix=dataset_id)

    for blob in blobs:
        print(f"\n * Downloading data file '{blob.name}' ...\n")
        blob_dir, file_name = os.path.split(blob.name)
        if (
            file_name == ""
        ):  # If the object blob path is a directory continue searching for files
            continue
        blob_local_dir = os.path.join(datasets_path, blob_dir)
        if not os.path.exists(blob_local_dir):
            os.makedirs(blob_local_dir)
        cloud_storage.download_blob(blob, os.path.join(blob_local_dir, file_name))

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
    latest_version: bool = False,
    compatible_minari_version: bool = False,
) -> Dict[str, Dict[str, str]]:
    """Get the names and metadata of all the Minari datasets in the remote Farama server.

    Args:
        latest_version (bool): if `True` only the latest version of the datasets are returned i.e. from ['D4RL/door/human-v0', 'D4RL/door/human-v1`], only the metadata for v1 is returned. Default to `False`.
        compatible_minari_version (bool): if `True` only the datasets compatible with the current Minari version are returned. Default to `False`.

    Returns:
       Dict[str, Dict[str, str]]: keys the names of the Minari datasets and values the metadata
    """
    from minari import supported_dataset_versions

    def blob_to_metadata(blob):
        try:
            if os.path.basename(blob.name) == METADATA_FILE_NAME:
                metadata = json.loads(blob.download_as_bytes(client=None))
                if (
                    compatible_minari_version
                    and metadata["minari_version"] not in supported_dataset_versions
                ):
                    return
                return metadata
        except Exception:
            warnings.warn(f"Misconfigured dataset named {blob.name} on remote")

    cloud_storage = get_cloud_storage()
    blobs = cloud_storage.list_blobs()
    with ThreadPoolExecutor(max_workers=10) as executor:
        remote_metadatas = executor.map(blob_to_metadata, blobs)

    remote_datasets = {}
    max_version = defaultdict(dict)
    for metadata in remote_metadatas:
        if metadata is None:
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


def get_remote_dataset_versions(
    namespace: str | None,
    dataset_name: str,
    latest_version: bool = False,
    compatible_minari_version: bool = False,
) -> List[int]:
    """Finds all registered versions in the remote Farama server of the dataset given.

    Args:
        namespace (str | None): identifier for remote namespace. Defaults to None.
        dataset_name (str): name of the dataset.
        latest_version (bool): if `True` only the latest version of the datasets is returned. Default to `False`.
        compatible_minari_version: only return highest version among the datasets compatible with the local installed version of Minari. Default to `False`
    Returns:
        A list of integer versions of the dataset with the specified requirements.
    """
    versions: list[int] = []

    for dataset_id in list_remote_datasets(
        latest_version, compatible_minari_version
    ).keys():
        (
            remote_namespace,
            remote_dataset_name,
            remote_version,
        ) = parse_dataset_id(dataset_id)

        if remote_namespace == namespace and remote_dataset_name == dataset_name:
            versions.append(remote_version)

    return versions

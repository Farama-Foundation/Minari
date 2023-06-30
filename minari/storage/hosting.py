from __future__ import annotations

import glob
import importlib.metadata
import os
import warnings
from typing import Dict

import h5py
from google.cloud import storage  # pyright: ignore [reportGeneralTypeIssues]
from gymnasium import logger
from packaging.specifiers import SpecifierSet
from tqdm.auto import tqdm  # pyright: ignore [reportMissingModuleSource]

from minari.dataset.minari_dataset import parse_dataset_id
from minari.storage.datasets_root_dir import get_dataset_path
from minari.storage.local import load_dataset


# Use importlib due to circular import when: "from minari import __version__"
__version__ = importlib.metadata.version("minari")


def upload_dataset(dataset_id: str, path_to_private_key: str):
    """Upload a Minari dataset to the remote Farama server.

    If you would like to upload a dataset please first get in touch with the Farama team at contact@farama.org.

    Args:
        dataset_id (str): name id of the local Minari dataset
        path_to_private_key (str): path to the GCP bucket json credentials. Reach out to the Farama team.
    """

    def _upload_local_directory_to_gcs(local_path, bucket, gcs_path):
        assert os.path.isdir(local_path)
        for local_file in glob.glob(local_path + "/**"):
            if not os.path.isfile(local_file):
                _upload_local_directory_to_gcs(
                    local_file, bucket, gcs_path + "/" + os.path.basename(local_file)
                )
            else:
                remote_path = os.path.join(gcs_path, local_file[1 + len(local_path) :])
                blob = bucket.blob(remote_path)
                # add metadata to main data file of dataset
                if blob.name.endswith("main_data.hdf5"):
                    blob.metadata = metadata
                blob.upload_from_filename(local_file)

    file_path = get_dataset_path(dataset_id)
    remote_datasets = list_remote_datasets()
    if dataset_id not in remote_datasets.keys():
        storage_client = storage.Client.from_service_account_json(
            json_credentials_path=path_to_private_key
        )
        bucket = storage.Bucket(storage_client, "minari-datasets")

        dataset = load_dataset(dataset_id)

        with h5py.File(dataset.spec.data_path, "r") as f:
            metadata = dict(f.attrs.items())

        # See https://github.com/googleapis/python-storage/issues/27 for discussion on progress bars
        _upload_local_directory_to_gcs(str(file_path), bucket, dataset_id)

        print(f"Dataset {dataset_id} uploaded!")

        combined_datasets = dataset.spec.combined_datasets

        if len(combined_datasets) > 0:
            print(
                f"Dataset {dataset_id} is formed by a combination of the following datasets:"
            )
            for name in combined_datasets:
                print(f"\t{name}")
            for dataset in combined_datasets:
                print(f"Uploading dataset {dataset}")
                upload_dataset(
                    dataset_id=dataset, path_to_private_key=path_to_private_key
                )
    else:
        print(
            f"Stopped upload of dataset {dataset_id}. {dataset_id} is already in the Farama servers."
        )


def download_dataset(dataset_id: str, force_download: bool = False):
    """Download dataset from remote Farama server.

    An error will be raised if the dataset version is not compatible with the local installed version of Minari.
    This error can be skipped and the download continued with the `force_download` argument. Also, with `force_download`,
    any local datasets that match the id of the downloading dataset will be overridden.

    Args:
        dataset_id (str): name id of the Minari dataset
        force_download (bool): boolean flag for force downloading the dataset. Default Value = False
    """
    # Check if minari version is compatible with dataset
    download_env_name, download_dataset_name, download_version = parse_dataset_id(
        dataset_id
    )
    all_remote_dataset_ids = list_remote_datasets().keys()
    compatible_remote_dataset_ids = list_remote_datasets(
        compatible_minari_version=True
    ).keys()

    if dataset_id not in all_remote_dataset_ids:
        raise ValueError(
            f"Dataset {dataset_id} doesn't exist in the remote Farama server."
        )
    # If dataset_id not in remote, version is not compatible or dataset doesn't exist.
    # Check and suggest other compatible versions.
    if dataset_id not in compatible_remote_dataset_ids:
        available_datasets = []
        for id in compatible_remote_dataset_ids:
            env_name, dataset_name, version = parse_dataset_id(id)
            if download_env_name == env_name and download_dataset_name == dataset_name:
                available_datasets.append(f"{env_name}-{dataset_name}-v{version}")

        if available_datasets:
            error_msg = f"The version you are trying to download for dataset, {dataset_id}, is not compatible with\
                                    your local installed version of Minari, {__version__}."
            if not force_download:
                raise ValueError(
                    error_msg
                    + f" We found other dataset versions that are compatible: {', '.join(available_datasets)}."
                )
            else:
                logger.warn(
                    error_msg
                    + f" We found other dataset versions that are compatible: {', '.join(available_datasets)}."
                )
        else:
            raise ValueError(error_msg)

    higher_version = find_highest_remote_version(
        download_env_name, download_dataset_name, True
    )

    if higher_version > download_version:
        logger.warn(
            f"We recommend you install a higher dataset version available and compatible with your local installed Minari version: {download_env_name}-{download_dataset_name}-v{higher_version}."
        )

    file_path = get_dataset_path(dataset_id)
    if os.path.exists(file_path):
        if not force_download:
            logger.warn(
                f"Skipping Download. Dataset {dataset_id} found locally at {file_path}, Use force_download=True to download the dataset again.\n"
            )
            return

    print(f"\nDownloading {dataset_id} from Farama servers...")
    storage_client = storage.Client.create_anonymous_client()

    bucket = storage_client.bucket(bucket_name="minari-datasets")
    # Construct a client side representation of a blob.
    # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
    # any content from Google Cloud Storage. As we don't need additional data,
    # using `Bucket.blob` is preferred here.
    blobs = bucket.list_blobs(prefix=dataset_id)  # Get list of files
    for blob in blobs:
        print(f"\n * Downloading data file '{blob.name}' ...\n")
        blob_dir, file_name = os.path.split(blob.name)
        # If the object blob path is a directory continue searching for files
        if file_name == "":
            continue
        blob_local_dir = os.path.join(os.path.dirname(file_path), blob_dir)
        if not os.path.exists(blob_local_dir):
            os.makedirs(blob_local_dir)
        # Download progress bar:
        # https://stackoverflow.com/questions/62811608/how-to-show-progress-bar-when-we-are-downloading-a-file-from-cloud-bucket-using
        with open(os.path.join(blob_local_dir, file_name), "wb") as f:
            with tqdm.wrapattr(f, "write", total=blob.size) as file_obj:
                storage_client.download_blob_to_file(blob, file_obj)

    print(f"\nDataset {dataset_id} downloaded to {file_path}")

    combined_datasets = load_dataset(dataset_id).spec.combined_datasets

    # If the dataset is a combination of other datasets download the subdatasets recursively
    if len(combined_datasets) > 0:
        print(
            f"\nDataset {dataset_id} is formed by a combination of the following datasets:"
        )
        for name in combined_datasets:
            print(f"  * {name}")
        print("\nDownloading extra datasets ...")
        for dataset in combined_datasets:
            download_dataset(dataset_id=dataset)


def list_remote_datasets(
    compatible_minari_version: bool = False,
) -> Dict[str, Dict[str, str]]:
    """Get the names and metadata of all the Minari dataset in the remote Farama server.

    Args:
        compatible_minari_version (bool): if `True` only the datasets compatible with the current Minari version are returned. Default to `False`.

    Returns:
       Dict[str, Dict[str, str]]: keys the names of the Minari datasets and values the metadata
    """
    storage_client = storage.Client.create_anonymous_client()

    blobs = storage_client.list_blobs(bucket_or_name="minari-datasets")

    remote_datasets = {}
    for blob in blobs:
        try:
            if blob.name.endswith("main_data.hdf5"):
                metadata = blob.metadata
                if compatible_minari_version and __version__ not in SpecifierSet(
                    metadata["minari_version"]
                ):
                    continue
                remote_datasets[blob.metadata["dataset_id"]] = metadata
        except Exception:
            warnings.warn(f"Misconfigured dataset named {blob.name} on remote")

    return remote_datasets


def find_highest_remote_version(
    env_name: str | None, dataset_name: str, compatible_minari_version: bool = False
) -> int:
    """Finds the highest registered version in the remote Farama server of the dataset given.

    Args:
        env_name: name to identigy the environment of the dataset
        dataset_name: name of the dataset within the ``env_name``
        compatible_minari_version: only return highest version among the datasets compatible with the local installed version of Minari. Default to `False`
    Returns:
        The highest version of a dataset with matching environment name and name, otherwise ``None`` is returned.
    """
    version: list[int] = []

    for dataset_id in list_remote_datasets(compatible_minari_version).keys():
        remote_env_name, remote_dataset_name, remote_version = parse_dataset_id(
            dataset_id
        )

        if (
            remote_env_name == env_name
            and remote_dataset_name == dataset_name
            and remote_version is not None
        ):
            version.append(remote_version)

    if not version:
        raise ValueError(
            f"Couldn't find any version for dataset {env_name}-{dataset_name}."
        )

    return max(version)

from __future__ import annotations

import glob
import importlib.metadata
import os
import warnings
from typing import Dict, List

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
                    with h5py.File(
                        local_file, "r"
                    ) as file:  # TODO: remove h5py when migrating to JSON metadata
                        blob.metadata = file.attrs
                blob.upload_from_filename(local_file)

    file_path = get_dataset_path(dataset_id)
    remote_datasets = list_remote_datasets()
    if dataset_id not in remote_datasets.keys():
        storage_client = storage.Client.from_service_account_json(
            json_credentials_path=path_to_private_key
        )
        bucket = storage.Bucket(storage_client, "minari-datasets")

        dataset = load_dataset(dataset_id)

        # See https://github.com/googleapis/python-storage/issues/27 for discussion on progress bars
        _upload_local_directory_to_gcs(str(file_path), bucket, dataset_id)

        print(f"Dataset {dataset_id} uploaded!")

        combined_datasets = dataset.spec.combined_datasets

        if len(combined_datasets) > 0:
            print(
                f"Dataset {dataset_id} is formed by a combination of the following datasets: "
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
    download_env_name, download_dataset_name, download_version = parse_dataset_id(
        dataset_id
    )

    all_dataset_versions = get_remote_dataset_versions(
        download_env_name, download_dataset_name
    )

    # 1. Check if there are any remote versions of the dataset at all
    if not all_dataset_versions:
        raise ValueError(
            f"Couldn't find any version for dataset {download_env_name}-{download_dataset_name} in the remote Farama server."
        )

    # 2. Check if there are any remote compatible versions with the local installed Minari version
    max_version = get_remote_dataset_versions(
        download_env_name, download_dataset_name, True, True
    )
    if not max_version:
        if not force_download:
            raise ValueError(
                f"Couldn't find any compatible version of dataset {download_env_name}-{download_dataset_name} with the local installed version of Minari, {__version__}."
            )
        else:
            logger.warn(
                f"Couldn't find any compatible version of dataset {download_env_name}-{download_dataset_name} with the local installed version of Minari, {__version__}."
            )

    # 3. Check that the dataset version exists
    if download_version not in all_dataset_versions:
        e_msg = f"The dataset version, {dataset_id}, doesn't exist in the remote Farama server."
        if not max_version:
            raise ValueError(
                e_msg
                + f", and no other compatible versions with the local installed Minari version {__version__} were found."
            )
        else:
            raise ValueError(
                e_msg
                + f" We suggest you download the latest compatible version of this dataset: {download_env_name}-{download_dataset_name}-v{max_version[0]}"
            )

    # 4. Check that the dataset version is compatible with the local installed Minari version
    compatible_dataset_versions = get_remote_dataset_versions(
        download_env_name, download_dataset_name, True
    )
    if download_version not in compatible_dataset_versions:
        e_msg = f"The version you are trying to download for dataset, {dataset_id}, is not compatible with\
                                    your local installed version of Minari, {__version__}."
        if not force_download:
            raise ValueError(
                e_msg
                + f" You can download the latest compatible version of this dataset: {download_env_name}-{download_dataset_name}-v{max_version[0]}."
            )
        # Only a warning and force download incompatible dataset
        else:
            logger.warn(
                e_msg
                + f" {dataset_id} will be FORCE download but you can download the latest compatible version of this dataset: {download_env_name}-{download_dataset_name}-v{max_version}."
            )

    # 5. Warning to recommend downloading the latest compatible version of the dataset
    elif max_version[0] is not None and download_version < max_version[0]:
        logger.warn(
            f"We recommend you install a higher dataset version available and compatible with your local installed Minari version: {download_env_name}-{download_dataset_name}-v{max_version}."
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

    # Skip a force download of an incompatible dataset version
    if download_dataset in compatible_dataset_versions:
        combined_datasets = load_dataset(dataset_id).spec.combined_datasets

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
        latest_version (bool): if `True` only the latest version of the datasets are returned i.e. from ['door-human-v0', 'door-human-v1`], only the metadata for v1 is returned. Default to `False`.
        compatible_minari_version (bool): if `True` only the datasets compatible with the current Minari version are returned. Default to `False`.

    Returns:
       Dict[str, Dict[str, str]]: keys the names of the Minari datasets and values the metadata
    """
    client = storage.Client.create_anonymous_client()
    blobs = client.list_blobs(bucket_or_name="minari-datasets")

    # Generate dict = {'env_name-dataset_name': (version, metadata)}
    remote_datasets = {}
    for blob in blobs:
        try:
            if blob.name.endswith("main_data.hdf5"):
                metadata = blob.metadata
                if compatible_minari_version and __version__ not in SpecifierSet(
                    metadata["minari_version"]
                ):
                    continue
                dataset_id = metadata["dataset_id"]
                env_name, dataset_name, version = parse_dataset_id(dataset_id)
                dataset = f"{env_name}-{dataset_name}"
                if latest_version:
                    if (
                        dataset not in remote_datasets
                        or version > remote_datasets[dataset][0]
                    ):
                        remote_datasets[dataset] = (version, metadata)
                else:
                    remote_datasets[dataset_id] = metadata
        except Exception:
            warnings.warn(f"Misconfigured dataset named {blob.name} on remote")

    if latest_version:
        # Return dict = {'dataset_id': metadata}
        return dict(
            map(lambda x: (f"{x[0]}-v{x[1][0]}", x[1][1]), remote_datasets.items())
        )
    else:
        return remote_datasets


def get_remote_dataset_versions(
    env_name: str | None,
    dataset_name: str,
    latest_version: bool = False,
    compatible_minari_version: bool = False,
) -> List[int]:
    """Finds all registered versions in the remote Farama server of the dataset given.

    Args:
        env_name (str): name to identigy the environment of the dataset
        dataset_name (str): name of the dataset within the ``env_name``
        latest_version (bool): if `True` only the latest version of the datasets is returned. Default to `False`.
        compatible_minari_version: only return highest version among the datasets compatible with the local installed version of Minari. Default to `False`
    Returns:
        A list of integer versions of the dataset with the specified requirements.
    """
    versions: list[int] = []

    for dataset_id in list_remote_datasets(
        latest_version, compatible_minari_version
    ).keys():
        remote_env_name, remote_dataset_name, remote_version = parse_dataset_id(
            dataset_id
        )

        if remote_env_name == env_name and remote_dataset_name == dataset_name:
            versions.append(remote_version)

    return versions

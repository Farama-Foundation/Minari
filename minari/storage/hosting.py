import glob
import os
from typing import Dict

import h5py
from google.cloud import storage  # pyright: ignore [reportGeneralTypeIssues]
from gymnasium import logger
from tqdm.auto import tqdm  # pyright: ignore [reportMissingModuleSource]

from minari.storage.datasets_root_dir import get_dataset_path
from minari.storage.local import load_dataset


def upload_dataset(dataset_name: str, path_to_private_key: str):
    """Upload a Minari dataset to the remote Farama server.

    If you would like to upload a dataset please first get in touch with the Farama team at contact@farama.org.

    Args:
        dataset_name (str): name id of the local Minari dataset
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

    file_path = get_dataset_path(dataset_name)
    remote_datasets = list_remote_datasets()
    if dataset_name not in remote_datasets.keys():
        storage_client = storage.Client.from_service_account_json(
            json_credentials_path=path_to_private_key
        )
        bucket = storage.Bucket(storage_client, "minari-datasets")

        dataset = load_dataset(dataset_name)

        with h5py.File(dataset.spec.data_path, "r") as f:
            metadata = dict(f.attrs.items())

        # See https://github.com/googleapis/python-storage/issues/27 for discussion on progress bars
        _upload_local_directory_to_gcs(str(file_path), bucket, dataset_name)

        print(f"Dataset {dataset_name} uploaded!")

        combined_datasets = dataset.spec.combined_datasets

        if len(combined_datasets) > 0:
            print(
                f"Dataset {dataset_name} is formed by a combination of the following datasets:"
            )
            for name in combined_datasets:
                print(f"\t{name}")
            for dataset in combined_datasets:
                print(f"Uploading dataset {dataset}")
                upload_dataset(
                    dataset_name=dataset, path_to_private_key=path_to_private_key
                )
    else:
        print(
            f"Stopped upload of dataset {dataset_name}. {dataset_name} is already in the Farama servers."
        )


def download_dataset(dataset_name: str):
    """Download dataset from remote Farama server.

    Args:
        dataset_name (str): name id of the Minari dataset
    """
    file_path = get_dataset_path(dataset_name)
    if os.path.exists(file_path):
        logger.warn(
            f"Dataset {dataset_name} found locally at {file_path} and its content will be overridden with the remote dataset.\n"
        )

    print(f"\nDownloading {dataset_name} from Farama servers...")
    storage_client = storage.Client.create_anonymous_client()

    bucket = storage_client.bucket(bucket_name="minari-datasets")
    # Construct a client side representation of a blob.
    # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
    # any content from Google Cloud Storage. As we don't need additional data,
    # using `Bucket.blob` is preferred here.
    blobs = bucket.list_blobs(prefix=dataset_name)  # Get list of files

    for blob in blobs:
        print(f"\n * Downloading data file '{blob.name}' ...\n")
        blob_dir, file_name = os.path.split(blob.name)
        blob_local_dir = os.path.join(os.path.dirname(file_path), blob_dir)
        if not os.path.exists(blob_local_dir):
            os.makedirs(blob_local_dir)
        # Download progress bar:
        # https://stackoverflow.com/questions/62811608/how-to-show-progress-bar-when-we-are-downloading-a-file-from-cloud-bucket-using
        with open(os.path.join(blob_local_dir, file_name), "wb") as f:
            with tqdm.wrapattr(f, "write", total=blob.size) as file_obj:
                storage_client.download_blob_to_file(blob, file_obj)

    print(f"\nDataset {dataset_name} downloaded to {file_path}")

    combined_datasets = load_dataset(dataset_name).spec.combined_datasets

    # If the dataset is a combination of other datasets download the subdatasets recursively
    if len(combined_datasets) > 0:
        print(
            f"\nDataset {dataset_name} is formed by a combination of the following datasets:"
        )
        for name in combined_datasets:
            print(f"  * {name}")
        print("\nDownloading extra datasets ...")
        for dataset in combined_datasets:
            download_dataset(dataset_name=dataset)


def list_remote_datasets() -> Dict[str, Dict[str, str]]:
    """Get the names and metadata of all the Minari dataset in the remote Farama server.

    Returns:
       Dict[str, Dict[str, str]]: keys the names of the Minari datasets and values the metadata
    """
    storage_client = storage.Client.create_anonymous_client()

    blobs = storage_client.list_blobs(bucket_or_name="minari-datasets")

    remote_datasets_metadata = list(
        map(
            lambda x: x.metadata,
            filter(lambda x: x.name.endswith("main_data.hdf5"), blobs),
        )
    )
    remote_datasets = {}
    for metadata in remote_datasets_metadata:
        remote_datasets[metadata["dataset_name"]] = metadata

    return remote_datasets

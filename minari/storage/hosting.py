import os.path

from google.cloud import storage

from .. import dataset
from ..utils.assert_name_spec import test_and_return_name
from .datasets_root_dir import get_file_path


def upload_dataset(dataset_name: str):
    project_id = "dogwood-envoy-367012"
    bucket_name = "minari"
    test_and_return_name(dataset_name)
    file_path = get_file_path(dataset_name)

    storage_client = storage.Client(project_id)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(f"{dataset_name}.hdf5")

    blob.upload_from_filename(
        file_path
    )  # See https://github.com/googleapis/python-storage/issues/27 for discussion on progress bars

    print(f"Dataset {dataset_name} uploaded!")


def download_dataset(dataset_name: str):
    test_and_return_name(dataset_name)
    file_path = get_file_path(dataset_name)

    if os.path.isfile(file_path):
        print(f"Dataset {dataset_name} found locally at {file_path}")
    else:
        print(
            f"Dataset not found locally. Downloading {dataset_name} from Farama servers..."
        )
        project_id = "dogwood-envoy-367012"
        bucket_name = "minari"
        storage_client = storage.Client(project=project_id)

        bucket = storage_client.bucket(bucket_name)

        # Construct a client side representation of a blob.
        # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
        # any content from Google Cloud Storage. As we don't need additional data,
        # using `Bucket.blob` is preferred here.
        blob = bucket.blob(f"{dataset_name}.hdf5")

        blob.download_to_filename(
            f"{file_path}"
        )  # See https://github.com/googleapis/python-storage/issues/27 for discussion on progress bars
        print(f"Dataset {dataset_name} downloaded to {file_path}")

    return dataset.MinariDataset.load(file_path)


def list_remote_datasets():
    project_id = "dogwood-envoy-367012"
    bucket_name = "minari"
    storage_client = storage.Client(project=project_id)

    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs()

    print("Datasets available to download:")
    for blob in blobs:
        print(blob.name)

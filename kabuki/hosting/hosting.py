import os.path

from google.cloud import storage
from ..utils.assert_name_spec import test_and_return_name


def upload_dataset(dataset_path: str):
    project_id = "dogwood-envoy-367012"
    bucket_name = "kabuki-datasets"
    filename = test_and_return_name(dataset_path)

    storage_client = storage.Client(project_id)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(dataset_path)

    blob.upload_from_filename(
        f"{filename}.hdf5"
    )  # See https://github.com/googleapis/python-storage/issues/27 for discussion on progress bars

    print(f"File {filename}.hdf5 uploaded!")


def retrieve_dataset(source_blob_name: str, return_dataset=True):
    filename = test_and_return_name(source_blob_name)

    if os.path.isfile(source_blob_name):
        print(f"Dataset {source_blob_name} found locally.")
        return source_blob_name
    else:
        print(
            f"Dataset not found locally. Downloading {filename} from Farama servers..."
        )
        project_id = "dogwood-envoy-367012"
        bucket_name = "kabuki-datasets"
        storage_client = storage.Client(project=project_id)

        bucket = storage_client.bucket(bucket_name)

        # Construct a client side representation of a blob.
        # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
        # any content from Google Cloud Storage. As we don't need additional data,
        # using `Bucket.blob` is preferred here.
        blob = bucket.blob(f"{filename}.hdf5")

        blob.download_to_filename(
            source_blob_name
        )  # See https://github.com/googleapis/python-storage/issues/27 for discussion on progress bars
        print(f"Dataset {source_blob_name} downloaded!")

        if return_dataset:
            from .. import dataset

            return dataset.MDPDataset.load(source_blob_name)


def list_datasets():
    project_id = "dogwood-envoy-367012"
    bucket_name = "kabuki-datasets"
    storage_client = storage.Client(project=project_id)

    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs()

    for blob in blobs:
        print(blob.name)

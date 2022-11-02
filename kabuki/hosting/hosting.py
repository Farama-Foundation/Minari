import os.path

from google.cloud import storage
from ..utils.assert_name_spec import test_and_return_name
from .. import dataset


def upload_dataset(dataset_path: str, root_dir: str = ".datasets"):
    project_id = "dogwood-envoy-367012"
    bucket_name = "kabuki-datasets"
    filename = test_and_return_name(dataset_path)
    path = os.path.join(root_dir, f"{filename}.hdf5")

    storage_client = storage.Client(project_id)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(f"{dataset_path}.hdf5")

    blob.upload_from_filename(
        path
    )  # See https://github.com/googleapis/python-storage/issues/27 for discussion on progress bars

    print(f"Dataset {filename} uploaded!")


def retrieve_dataset(dataset_name: str, root_dir: str = ".datasets"):
    filename = test_and_return_name(dataset_name)
    path = os.path.join(root_dir, filename)

    if not os.path.isdir(root_dir):
        os.mkdir(root_dir)

    if os.path.isfile(path):
        print(f"Dataset {dataset_name} found locally in {path}/")
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
            f"{path}.hdf5"
        )  # See https://github.com/googleapis/python-storage/issues/27 for discussion on progress bars
        print(f"Dataset {dataset_name} downloaded to {path}/")

    return dataset.KabukiDataset.load(f"{path}.hdf5")


def list_datasets():
    project_id = "dogwood-envoy-367012"
    bucket_name = "kabuki-datasets"
    storage_client = storage.Client(project=project_id)

    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs()

    print(f"Found datasets:")
    for blob in blobs:
        print(blob.name)

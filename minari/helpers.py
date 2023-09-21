import os

import numpy as np
from google.cloud import storage  # pyright: ignore [reportGeneralTypeIssues]

from minari.storage.datasets_root_dir import get_dataset_path


def get_dataset_size(dataset_id: str):
    """Returns the dataset size in MB.

    Args:
        dataset_id (str) : name id of Minari Dataset

    Returns:
        datasize (float): size of the dataset in MB
    """
    file_path = get_dataset_path(dataset_id)
    data_path = os.path.join(file_path, "data")
    datasize_list = []
    if os.path.exists(data_path):

        for filename in os.listdir(data_path):
            if ".hdf5" in filename:
                datasize = os.path.getsize(os.path.join(data_path, filename))
                datasize_list.append(datasize)

    else:
        storage_client = storage.Client.create_anonymous_client()
        bucket = storage_client.bucket(bucket_name="minari-datasets")

        blobs = bucket.list_blobs(prefix=dataset_id)
        for blob in blobs:
            if ".hdf5" in blob.name:
                datasize_list.append(bucket.get_blob(blob.name).size)

    datasize = int(np.sum(datasize_list) / 1000000)

    return datasize

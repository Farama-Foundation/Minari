from . import dataset
from kabuki.storage.hosting import (
    upload_dataset,
    download_dataset,
    list_remote_datasets,
)
from kabuki.storage.local import load_dataset, list_local_datasets, delete_dataset

__version__ = "0.0.1"

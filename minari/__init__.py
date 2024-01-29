from minari.data_collector import DataCollector
from minari.data_collector.callbacks import EpisodeMetadataCallback, StepDataCallback
from minari.dataset.minari_dataset import EpisodeData, MinariDataset
from minari.storage.hosting import (
    download_dataset,
    list_remote_datasets,
    upload_dataset,
)
from minari.storage.local import delete_dataset, list_local_datasets, load_dataset
from minari.utils import (
    combine_datasets,
    create_dataset_from_buffers,
    create_dataset_from_collector_env,
    get_normalized_score,
    split_dataset,
)


__all__ = [
    # Minari Dataset
    "MinariDataset",
    "EpisodeData",
    # Data collection
    "DataCollector",
    "EpisodeMetadataCallback",
    "StepDataCallback",
    # Dataset Functions
    "download_dataset",
    "list_remote_datasets",
    "upload_dataset",
    "delete_dataset",
    "list_local_datasets",
    "load_dataset",
    "combine_datasets",
    "create_dataset_from_buffers",
    "create_dataset_from_collector_env",
    "split_dataset",
    "get_normalized_score",
]

__version__ = "0.4.3"


def __getattr__(name):
    if name == "DataCollectorV0":
        from minari.data_collector import DataCollectorV0
        return DataCollectorV0
    else:
        raise ImportError(f"cannot import name '{name}' from '{__name__}' ({__file__})")

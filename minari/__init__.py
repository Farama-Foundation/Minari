from minari.data_collector import DataCollector
from minari.data_collector.callbacks import EpisodeMetadataCallback, StepDataCallback
from minari.dataset.episode_data import EpisodeData
from minari.dataset.minari_dataset import MinariDataset
from minari.dataset.step_data import StepData
from minari.storage.hosting import (
    download_dataset,
    list_remote_datasets,
    upload_dataset,
)
from minari.storage.local import delete_dataset, list_local_datasets, load_dataset
from minari.utils import (
    combine_datasets,
    create_dataset_from_buffers,
    get_normalized_score,
    split_dataset,
)


__all__ = [
    # Minari Dataset
    "MinariDataset",
    "EpisodeData",
    "StepData",
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
    "split_dataset",
    "get_normalized_score",
]

__version__ = "0.5.1"
supported_dataset_versions = {
    "0.4.0",
    "0.4.1",
    "0.4.2",
    "0.4.3",
    "0.5.0",
    "0.5.1",
}

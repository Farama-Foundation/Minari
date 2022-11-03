import os

from kabuki.dataset import KabukiDataset
from kabuki.storage.datasets_root_dir import get_file_path


def load_dataset(dataset_name: str):
    file_path = get_file_path(dataset_name)

    return KabukiDataset.load(file_path)

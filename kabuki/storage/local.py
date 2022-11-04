import os

from kabuki.dataset import KabukiDataset
from kabuki.storage.datasets_root_dir import get_file_path


def load_dataset(dataset_name: str):
    file_path = get_file_path(dataset_name)

    return KabukiDataset.load(file_path)


def list_local_datasets():
    datasets_path = get_file_path("").parent
    datasets = [
        f[:-5]
        for f in os.listdir(datasets_path)
        if os.path.isfile(os.path.join(datasets_path, f))
    ]

    print("Datasets found locally:")
    for dataset in datasets:
        print(dataset)


def delete_dataset(dataset_name: str):
    file_path = get_file_path(dataset_name)
    os.remove(file_path)
    print("Dataset {dataset_name} deleted!")

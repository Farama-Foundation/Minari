import os
import shutil
from minari.minari_dataset import MinariDataset
from minari.storage.datasets_root_dir import get_dataset_path


def load_dataset(dataset_name: str):
    file_path = get_dataset_path(dataset_name)
    data_path = os.path.join(file_path, "data", "main_data.hdf5")
    return MinariDataset(data_path)


def list_local_datasets(verbose=True):
    datasets_path = get_dataset_path("")
    datasets = [dir_name for dir_name in os.listdir(datasets_path)]

    if verbose:
        print("Datasets found locally:")
        for dataset in datasets:
            print(dataset)

    return datasets

def delete_dataset(dataset_name: str):
    dataset_path = get_dataset_path(dataset_name)
    shutil.rmtree(dataset_path)
    print(f"Dataset {dataset_name} deleted!")

import os
from pathlib import Path


def get_dataset_path(dataset_name):
    """Get the path to a dataset file."""
    # IF YOU EDIT THIS PLEASE ALSO EDIT LINES ~540 IN dataset.pyx
    datasets_path = os.environ.get("MINARI_DATASETS_PATH")
    if datasets_path is not None:
        file_path = os.path.join(datasets_path, dataset_name)
    else:
        datasets_path = os.path.join(os.path.expanduser("~"), ".minari", "datasets")
        file_path = os.path.join(datasets_path, dataset_name)

    os.makedirs(datasets_path, exist_ok=True)
    return Path(file_path)

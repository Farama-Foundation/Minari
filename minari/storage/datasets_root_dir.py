import os
from pathlib import Path


def get_dataset_path(dataset_id: str) -> Path:
    """Get the path to a dataset main directory."""
    datasets_path = os.environ.get("MINARI_DATASETS_PATH")
    if datasets_path is not None:
        file_path = os.path.join(datasets_path, dataset_id)
    else:
        datasets_path = os.path.join(os.path.expanduser("~"), ".minari", "datasets")
        file_path = os.path.join(datasets_path, dataset_id)

    os.makedirs(datasets_path, exist_ok=True)
    return Path(file_path)

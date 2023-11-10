import os
from pathlib import Path


def test_create_leftover():
    """Create a leftover dataset to simulate a failed test, to verify test isolation."""
    dataset_dir = os.environ.get(
        "MINARI_DATASETS_PATH",
        os.path.join(os.path.expanduser("~"), ".minari/datasets/"),
    )
    new_dataset_path = Path(dataset_dir) / "leftover_dataset/data"
    new_dataset_path.mkdir(parents=True)

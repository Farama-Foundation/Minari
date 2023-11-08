from pathlib import Path

def test_create_leftover(tmp_dataset_dir):
    """Create a leftover dataset to simulate a failed test, to verify test isolation."""
    leftover_dataset_path = Path(tmp_dataset_dir) / "leftover_dataset/data"
    leftover_dataset_path.mkdir(parents=True)

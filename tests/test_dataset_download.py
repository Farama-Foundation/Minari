import pytest

import minari
from minari import MinariDataset


@pytest.mark.parametrize(
    "dataset_name",
    ["pen-human-v0", "door-human-v0", "hammer-human-v0", "relocate-human-v0"],
)
def test_download_dataset_from_farama_server(dataset_name: str):
    """Test downloading Minari datasets from remote server.

    Use 'human' adroit test since they are not excessively heavy.

    Args:
        dataset_name (str): name of the remote Minari dataset.
    """
    remote_datasets = minari.list_remote_datasets()
    assert dataset_name in remote_datasets

    minari.download_dataset(dataset_name)
    local_datasets = minari.list_local_datasets()
    assert dataset_name in local_datasets

    dataset = minari.load_dataset(dataset_name)
    assert isinstance(dataset, MinariDataset)

    minari.delete_dataset(dataset_name)
    local_datasets = minari.list_local_datasets()
    assert dataset_name not in local_datasets

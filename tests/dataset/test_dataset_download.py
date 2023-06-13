import pytest

import minari
from minari import MinariDataset


@pytest.mark.parametrize(
    "dataset_id",
    ["pen-human-v0", "door-human-v0", "hammer-human-v0", "relocate-human-v0"],
)
def test_download_dataset_from_farama_server(dataset_id: str):
    """Test downloading Minari datasets from remote server.

    Use 'human' adroit test since they are not excessively heavy.

    Args:
        dataset_id (str): name of the remote Minari dataset.
    """
    remote_datasets = minari.list_remote_datasets()
    assert dataset_id in remote_datasets

    minari.download_dataset(dataset_id)
    local_datasets = minari.list_local_datasets()
    assert dataset_id in local_datasets

    dataset = minari.load_dataset(dataset_id)
    assert isinstance(dataset, MinariDataset)

    minari.delete_dataset(dataset_id)
    local_datasets = minari.list_local_datasets()
    assert dataset_id not in local_datasets

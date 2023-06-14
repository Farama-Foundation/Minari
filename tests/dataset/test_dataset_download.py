import pytest

import minari
from minari import MinariDataset
from minari.storage.datasets_root_dir import get_dataset_path
from tests.common import check_data_integrity


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

    minari.download_dataset(dataset_id, force_download=True)
    local_datasets = minari.list_local_datasets()
    assert dataset_id in local_datasets

    file_path = get_dataset_path(dataset_id)

    with pytest.warns(
        UserWarning,
        match=f"Skipping Download. Dataset {dataset_id} found locally at {file_path}, Use force_download=True to download the dataset again.\n",
    ):
        minari.download_dataset(dataset_id)

    download_dataset_output = minari.download_dataset(dataset_id)
    assert download_dataset_output is None

    dataset = minari.load_dataset(dataset_id)
    assert isinstance(dataset, MinariDataset)

    check_data_integrity(dataset._data, dataset.episode_indices)

    minari.delete_dataset(dataset_id)
    local_datasets = minari.list_local_datasets()
    assert dataset_id not in local_datasets

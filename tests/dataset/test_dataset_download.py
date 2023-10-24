import pytest

import minari
from minari import MinariDataset
from minari.storage.datasets_root_dir import get_dataset_path
from minari.storage.hosting import get_remote_dataset_versions
from tests.common import check_data_integrity


env_names = ["pen", "door", "hammer", "relocate"]


def get_latest_compatible_dataset_id(env_name, dataset_name):
    latest_compatible_version = get_remote_dataset_versions(
        dataset_name=dataset_name,
        env_name=env_name,
        latest_version=True,
        compatible_minari_version=True,
    )[0]
    return f"{env_name}-{dataset_name}-v{latest_compatible_version}"


@pytest.mark.parametrize(
    "dataset_id",
    [
        get_latest_compatible_dataset_id(env_name=env_name, dataset_name="human")
        for env_name in env_names
    ],
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

    check_data_integrity(dataset.storage, dataset.episode_indices)

    minari.delete_dataset(dataset_id)
    local_datasets = minari.list_local_datasets()
    assert dataset_id not in local_datasets


@pytest.mark.parametrize(
    "dataset_id",
    [
        get_latest_compatible_dataset_id(env_name=env_name, dataset_name="human")
        for env_name in env_names
    ],
)
def test_load_dataset_with_download(dataset_id: str):
    """Test load dataset with and without download."""
    with pytest.raises(FileNotFoundError):
        dataset = minari.load_dataset(dataset_id)

    dataset = minari.load_dataset(dataset_id, download=True)
    assert isinstance(dataset, MinariDataset)

    minari.delete_dataset(dataset_id)

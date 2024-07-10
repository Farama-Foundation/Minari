import pytest
from pytest import MonkeyPatch

import minari
from minari import MinariDataset
from minari.namespace import (
    create_namespace,
    delete_namespace,
    download_namespace_metadata,
    get_namespace_metadata,
    list_local_namespaces,
    list_remote_namespaces,
)
from minari.storage.datasets_root_dir import get_dataset_path
from tests.common import check_data_integrity, get_latest_compatible_dataset_id


MINARI_TEST_REMOTE = "gcp://minari-datasets-test"


@pytest.fixture()
def use_test_server():
    with MonkeyPatch.context() as mp:
        mp.setenv("MINARI_REMOTE", MINARI_TEST_REMOTE)
        yield


def test_download_namespace_dataset(use_test_server):
    dataset_id = get_latest_compatible_dataset_id("test_namespace", "door", "human")
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
        assert minari.download_dataset(dataset_id) is None

    dataset = minari.load_dataset(dataset_id)
    assert isinstance(dataset, MinariDataset)

    check_data_integrity(dataset.storage, dataset.episode_indices)

    minari.delete_dataset(dataset_id)
    local_datasets = minari.list_local_datasets()
    assert dataset_id not in local_datasets


def test_pull_namespace_metadata(use_test_server):
    test_namespaces = {
        "test_namespace",
        "test_namespace/nested",
        "test_namespace/nested_2",
    }
    example_metadata = {"key": [1, 2, 3], "1": {"2": 3}, "description": "a description"}

    assert len(list_local_namespaces()) == 0
    assert test_namespaces.issubset(list_remote_namespaces())

    download_namespace_metadata("test_namespace/nested")
    assert list_local_namespaces() == ["test_namespace/nested"]
    assert get_namespace_metadata("test_namespace/nested") == example_metadata

    download_namespace_metadata("test_namespace/nested_2")
    assert list_local_namespaces() == [
        "test_namespace/nested",
        "test_namespace/nested_2",
    ]
    assert get_namespace_metadata("test_namespace/nested_2") is None

    # Pulling again is a no-op, unless there is a metadata conflict
    download_namespace_metadata("test_namespace/nested_2")
    assert get_namespace_metadata("test_namespace/nested") == example_metadata
    assert get_namespace_metadata("test_namespace/nested_2") is None

    delete_namespace("test_namespace/nested_2")
    create_namespace("test_namespace/nested_2", description="Conflicting description")

    with pytest.warns(UserWarning, match="Skipping update for namespace"):
        download_namespace_metadata("test_namespace/nested_2")

    assert get_namespace_metadata("test_namespace/nested_2") == {
        "description": "Conflicting description"
    }

    download_namespace_metadata("test_namespace/nested_2", overwrite=True)
    assert get_namespace_metadata("test_namespace/nested_2") is None

    with pytest.raises(ValueError, match="doesn't exist in the remote Farama server."):
        download_namespace_metadata("test_namespace/nested_3")

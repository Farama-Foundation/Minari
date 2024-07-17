import pytest
from pytest import MonkeyPatch

import minari
from minari import (
    MinariDataset,
    download_dataset,
    list_local_datasets,
    list_remote_datasets,
)
from minari.namespace import (
    create_namespace,
    delete_namespace,
    download_namespace_metadata,
    get_namespace_metadata,
    list_local_namespaces,
    list_remote_namespaces,
)
from tests.common import check_data_integrity, get_latest_compatible_dataset_id


MINARI_TEST_REMOTE = "gcp://minari-datasets-test"


@pytest.fixture()
def use_test_server():
    with MonkeyPatch.context() as mp:
        mp.setenv("MINARI_REMOTE", MINARI_TEST_REMOTE)
        yield


def test_download_namespace_dataset(use_test_server):
    namespace = "test_namespace"
    door_id = get_latest_compatible_dataset_id(namespace, "door", "human")
    pen_id = get_latest_compatible_dataset_id(namespace, "pen", "human")

    remote_datasets = list_remote_datasets()
    assert door_id in remote_datasets

    download_dataset(door_id)
    assert list(list_local_datasets().keys()) == [door_id]
    assert list_local_namespaces() == [namespace]
    assert get_namespace_metadata(namespace) is None

    download_dataset(pen_id)
    assert list(list_local_datasets().keys()) == [door_id, pen_id]
    assert list_local_namespaces() == [namespace]
    assert get_namespace_metadata(namespace) is None

    with pytest.warns(UserWarning, match="Skipping Download."):
        download_dataset(door_id)

    # Check door dataset downloaded correctly
    dataset = minari.load_dataset(door_id)
    assert isinstance(dataset, MinariDataset)

    check_data_integrity(dataset.storage, dataset.episode_indices)

    minari.delete_dataset(door_id)
    assert list(list_local_datasets().keys()) == [pen_id]


def test_download_namespace_metadata(use_test_server):
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

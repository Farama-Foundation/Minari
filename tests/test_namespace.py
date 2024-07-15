import gymnasium as gym
import pytest

from minari import list_local_datasets
from minari.data_collector.data_collector import DataCollector
from minari.namespace import (
    create_namespace,
    delete_namespace,
    get_namespace_metadata,
    list_local_namespaces,
    update_namespace_metadata,
)
from tests.common import (
    check_load_and_delete_dataset,
    create_dummy_dataset_with_collecter_env_helper,
)


@pytest.mark.parametrize(
    "namespace",
    [
        "example_-123",
        "nested/nested/nested/namespace",
    ],
)
@pytest.mark.parametrize(
    "description, metadata, combined_metadata",
    [
        (None, {}, None),
        ("my_desc", {}, {"description": "my_desc"}),
        (None, {"key": [1]}, {"description": None, "key": [1]}),
        ("my_desc", {"key": [1]}, {"description": "my_desc", "key": [1]}),
    ],
)
def test_create_namespace(namespace, description, metadata, combined_metadata):
    create_namespace(namespace, description, **metadata)
    assert namespace in list_local_namespaces()
    assert get_namespace_metadata(namespace) == combined_metadata
    delete_namespace(namespace)


@pytest.mark.parametrize("namespace", ["test_namespace"])
def test_namespace_update(namespace):
    create_namespace(namespace)

    with pytest.raises(ValueError, match="Namespace 'test_namespace' already exists"):
        create_namespace(namespace)

    update_namespace_metadata(namespace, description="a new definition")
    assert get_namespace_metadata(namespace) == {"description": "a new definition"}


@pytest.mark.parametrize("namespace", ["test_namespace"])
def test_create_nested_namespaces(namespace):
    create_namespace(namespace, description="my description")
    assert list_local_namespaces() == [namespace]
    assert get_namespace_metadata(namespace) == {"description": "my description"}

    nested_namespace = f"{namespace}/nested"
    create_namespace(nested_namespace, description="is nested")
    assert list_local_namespaces() == [namespace, nested_namespace]
    assert get_namespace_metadata(nested_namespace) == {"description": "is nested"}


def test_nonexistent_namespaces():
    with pytest.raises(ValueError, match="does not exist"):
        update_namespace_metadata("does_not/exist")

    with pytest.raises(ValueError, match="does not exist"):
        get_namespace_metadata("does_not/exist")


@pytest.mark.parametrize("namespace", ["/", "./", "../", "/namespace", "namespace/"])
def test_create_invalid_namespace(namespace):
    with pytest.raises(ValueError, match="Malformed namespace"):
        create_namespace(namespace)


@pytest.mark.parametrize("namespace", ["nested/namespace"])
def test_create_namespaced_datasets(namespace):
    env = gym.make("CartPole-v1")
    env = DataCollector(env)

    dataset_id_1 = f"{namespace}/cartpole-test-v1"
    create_dummy_dataset_with_collecter_env_helper(dataset_id_1, env)
    assert list_local_namespaces() == [namespace]
    assert get_namespace_metadata(namespace) is None

    update_namespace_metadata(namespace, description="new description")
    assert get_namespace_metadata(namespace) == {"description": "new description"}

    # Creating a new dataset in the same namespace doesn't change the namespace metadata
    dataset_id_2 = f"{namespace}/cartpole-test-v2"
    create_dummy_dataset_with_collecter_env_helper(dataset_id_2, env)
    assert list_local_namespaces() == [namespace]
    assert get_namespace_metadata(namespace) == {"description": "new description"}

    assert list(list_local_datasets().keys()) == [dataset_id_1, dataset_id_2]

    # Check that can only delete when empty
    with pytest.raises(ValueError, match="is not empty"):
        delete_namespace(namespace)

    check_load_and_delete_dataset(dataset_id_1)
    check_load_and_delete_dataset(dataset_id_2)
    delete_namespace(namespace)

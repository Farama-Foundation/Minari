from typing import List

import pytest
from huggingface_hub.errors import HfHubHTTPError

import minari
from minari import MinariDataset
from minari.storage.datasets_root_dir import get_dataset_path
from tests.common import (
    check_data_integrity,
    get_latest_compatible_dataset_id,
    skip_if_error,
)


env_names = ["pen", "door", "hammer", "relocate"]


@pytest.mark.parametrize(
    "dataset_id",
    [
        get_latest_compatible_dataset_id(
            namespace=f"D4RL/{env_name}", dataset_name="human"
        )
        for env_name in env_names
    ],
)
@skip_if_error(HfHubHTTPError)
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
        download_dataset_output = minari.download_dataset(dataset_id)

    assert download_dataset_output is None

    dataset = minari.load_dataset(dataset_id)
    assert isinstance(dataset, MinariDataset)

    check_data_integrity(dataset, list(dataset.episode_indices))

    minari.delete_dataset(dataset_id)
    local_datasets = minari.list_local_datasets()
    assert dataset_id not in local_datasets


@pytest.mark.parametrize(
    "dataset_id",
    [
        get_latest_compatible_dataset_id(
            namespace=f"D4RL/{env_name}", dataset_name="human"
        )
        for env_name in env_names
    ],
)
@skip_if_error(HfHubHTTPError)
def test_load_dataset_with_download(dataset_id: str):
    """Test load dataset with and without download."""
    with pytest.raises(FileNotFoundError):
        dataset = minari.load_dataset(dataset_id)

    dataset = minari.load_dataset(dataset_id, download=True)
    assert isinstance(dataset, MinariDataset)

    minari.delete_dataset(dataset_id)


@skip_if_error(HfHubHTTPError)
def test_download_error_messages(monkeypatch):
    # 1. Check if there are any remote versions of the dataset at all
    with pytest.raises(ValueError, match="Couldn't find any version for dataset"):
        minari.download_dataset("non-existent-dataset-v0")

    with pytest.raises(ValueError, match="Couldn't find any version for dataset"):
        minari.download_dataset("non-existent-dataset-v0", force_download=True)

    # 2. Check if there are any remote compatible versions with the local installed Minari version
    with monkeypatch.context() as mp:
        mp.setattr("minari.supported_dataset_versions", set())

        with pytest.raises(
            ValueError, match="Couldn't find any compatible version of dataset"
        ):
            minari.download_dataset("D4RL/door/human-v2")

        with pytest.warns(match="Couldn't find any compatible version of dataset"):
            minari.download_dataset("D4RL/door/human-v2", force_download=True)
        minari.delete_dataset("D4RL/door/human-v2")

    # 3. Check that the dataset version exists
    with pytest.raises(ValueError, match="doesn't exist in the remote Farama server."):
        minari.download_dataset("D4RL/door/human-v999")

    with pytest.raises(ValueError, match="doesn't exist in the remote Farama server."):
        minari.download_dataset("D4RL/door/human-v999", force_download=True)

    # 4. Check that the dataset version is compatible with the local installed Minari version
    def patch_get_remote_dataset(compatible_v: List[int], not_compatible_v: List[int]):
        compatible_metadata = {"minari_version": minari.__version__}
        not_compatible_metadata = {"minari_version": "not-compatible-version"}

        def patched_list_remote(*args, **kwargs):
            ds_list = {
                f"D4RL/door/human-v{v}": compatible_metadata for v in compatible_v
            }
            ds_list.update(
                {
                    f"D4RL/door/human-v{v}": not_compatible_metadata
                    for v in not_compatible_v
                }
            )
            return ds_list

        return patched_list_remote

    # Pretend that D4RL/door/human-v1 is compatible but D4RL/door/human-v2 is not
    with monkeypatch.context() as mp:
        mp.setattr(
            "minari.storage.hosting.list_remote_datasets",
            patch_get_remote_dataset([1], [2]),
        )

        with pytest.raises(
            ValueError,
            match="D4RL/door/human-v2, is not compatible with your local installed version of Minari",
        ):
            minari.download_dataset("D4RL/door/human-v2")

        with pytest.warns(
            match="will be FORCE download but you can download the latest compatible version of this dataset:"
        ):
            minari.download_dataset("D4RL/door/human-v2", force_download=True)
        minari.delete_dataset("D4RL/door/human-v2")

    # 5. Warning to recommend downloading the latest compatible version of the dataset
    # Pretend that D4RL/door/human-v3 exists and try to download D4RL/door/human-v2
    with monkeypatch.context() as mp:
        mp.setattr(
            "minari.storage.hosting.list_remote_datasets",
            patch_get_remote_dataset([2, 3], []),
        )

        with pytest.warns(
            match="We recommend you install a higher dataset version available and compatible"
        ):
            minari.download_dataset("D4RL/door/human-v2")
        minari.delete_dataset("D4RL/door/human-v2")

    # Skip datasets that exist locally
    latest_door_human_id = get_latest_compatible_dataset_id(
        namespace="D4RL/door", dataset_name="human"
    )
    minari.download_dataset(latest_door_human_id)

    with pytest.warns(
        match=f"Skipping Download. Dataset {latest_door_human_id} found locally at"
    ):
        minari.download_dataset(latest_door_human_id)

    minari.download_dataset(latest_door_human_id, force_download=True)
    minari.delete_dataset(latest_door_human_id)

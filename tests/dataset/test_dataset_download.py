import shutil
from pathlib import Path
from typing import Iterable, List, Optional

import gymnasium as gym
import pytest
from huggingface_hub.errors import HfHubHTTPError

import minari
from minari import DataCollector, MinariDataset
from minari.dataset.minari_storage import MinariStorage
from minari.storage import hosting
from minari.storage.datasets_root_dir import get_dataset_path
from tests.common import (
    check_data_integrity,
    create_dummy_dataset_with_collecter_env_helper,
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


class _LocalDirCloudStorage:
    """Fake remote serving datasets from a local directory, so no network is used."""

    def __init__(self, remote_dir: Path):
        self.remote_dir = remote_dir
        self.downloads: List[str] = []

    def list_datasets(self, prefix: Optional[str] = None) -> Iterable[str]:
        for metadata_path in self.remote_dir.glob("**/data/metadata.json"):
            dataset_id = metadata_path.parent.parent.relative_to(self.remote_dir)
            dataset_id = dataset_id.as_posix()
            if prefix is None or dataset_id.startswith(prefix):
                yield dataset_id

    def get_dataset_metadata(self, dataset_id: str) -> dict:
        return MinariStorage.read_raw_metadata(self.remote_dir / dataset_id / "data")

    def download_dataset(self, dataset_id: str, path: Path) -> None:
        self.downloads.append(dataset_id)
        shutil.copytree(
            self.remote_dir / dataset_id, path / dataset_id, dirs_exist_ok=True
        )


@pytest.fixture
def partially_downloaded_dataset(monkeypatch, tmp_path):
    """Create a dataset, publish it on a fake remote and break the local copy."""
    dataset_id = "cartpole-partial-v0"
    num_episodes = 3
    env = DataCollector(gym.make("CartPole-v1"), data_format="hdf5")
    create_dummy_dataset_with_collecter_env_helper(
        dataset_id, env, num_episodes=num_episodes
    )
    env.close()

    remote_dir = tmp_path / "remote"
    shutil.copytree(get_dataset_path(dataset_id), remote_dir / dataset_id)
    remote = _LocalDirCloudStorage(remote_dir)
    monkeypatch.setattr(hosting, "get_cloud_storage", lambda **_: remote)

    return dataset_id, num_episodes, remote


@pytest.mark.parametrize(
    "missing_files",
    [["main_data.hdf5"], ["metadata.json"], ["main_data.hdf5", "metadata.json"]],
)
def test_load_dataset_recovers_from_partial_download(
    partially_downloaded_dataset, missing_files
):
    """`load_dataset(download=True)` downloads again an incomplete local dataset."""
    dataset_id, num_episodes, remote = partially_downloaded_dataset
    data_path = get_dataset_path(dataset_id) / "data"
    for file_name in missing_files:
        (data_path / file_name).unlink()

    with pytest.warns(UserWarning, match="could not be loaded"):
        dataset = minari.load_dataset(dataset_id, download=True)

    assert remote.downloads == [dataset_id]
    assert isinstance(dataset, MinariDataset)
    assert dataset.total_episodes == num_episodes


def test_load_dataset_incomplete_error_message(partially_downloaded_dataset):
    """Without download, the error explains how to recover an incomplete dataset."""
    dataset_id, _, remote = partially_downloaded_dataset
    (get_dataset_path(dataset_id) / "data" / "main_data.hdf5").unlink()

    with pytest.raises(ValueError, match="force_download=True") as exc_info:
        minari.load_dataset(dataset_id)

    assert "minari.delete_dataset" in str(exc_info.value)
    assert remote.downloads == []

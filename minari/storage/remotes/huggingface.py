from pathlib import Path
from typing import Any, Iterable, Optional
from minari.storage.remotes.cloud_storage import CloudStorage
from huggingface_hub import HfApi, snapshot_download


class HuggingFaceStorage(CloudStorage):

    def __init__(self, name: str, key_path: Optional[str] = None) -> None:
        self.name = name
        self._api = HfApi()


    def upload_directory(self, path: Path, remote_dir_path: str) -> None:
        pass

    def upload_file(self, local_path: Path, remote_path: str) -> None:
        pass

    def list_datasets(self, prefix: Optional[str] = None) -> Iterable[str]:
        for dataset in self._api.list_datasets(author=self.name):
            yield dataset.id

    def download_dataset(self, dataset_id: Any, path: Path) -> None:
        snapshot_download(repo_id=dataset_id, cache_dir=path, repo_type="dataset")
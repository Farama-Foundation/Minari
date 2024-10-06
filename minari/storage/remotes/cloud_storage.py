from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable, Optional


class CloudStorage(ABC):
    @abstractmethod
    def __init__(self, name: str, key_path: Optional[str]) -> None:
        ...

    @abstractmethod
    def upload_directory(self, path: Path, remote_dir_path: str) -> None:
        ...

    @abstractmethod
    def upload_file(self, local_path: Path, remote_path: str) -> None:
        ...

    @abstractmethod
    def list_datasets(self, prefix: Optional[str] = None) -> Iterable[str]:
        ...
    
    @abstractmethod
    def get_dataset_metadata(self, dataset_id: str) -> dict:
        ...

    @abstractmethod
    def download_dataset(self, dataset_id: str, path: Path) -> None:
        ...

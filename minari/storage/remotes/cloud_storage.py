from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional


class CloudStorage(ABC):
    @abstractmethod
    def __init__(self, name: str, key_path: Optional[str]) -> None:
        ...

    @abstractmethod
    def upload_path(self, path: Path, dataset_id: str) -> None:
        ...

    @abstractmethod
    def list_blobs(self, prefix: Optional[str] = None) -> list:
        ...

    @abstractmethod
    def download_blob(self, blob: Any, file_path: str) -> None:
        ...

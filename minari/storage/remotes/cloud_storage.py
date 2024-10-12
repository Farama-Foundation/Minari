from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable, Optional


class CloudStorage(ABC):
    @abstractmethod
    def __init__(self, name: str, key_path: Optional[str]) -> None: ...

    @abstractmethod
    def upload_directory(self, path: Path, remote_dir_path: str) -> None: ...

    @abstractmethod
    def upload_file(self, local_path: Path, remote_path: str) -> None: ...

    @abstractmethod
    def list_blobs(self, prefix: Optional[str] = None) -> Iterable: ...

    @abstractmethod
    def download_blob(self, blob: Any, file_path: str) -> None: ...

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Optional


class CloudStorage(ABC):
    # TODO: add docs
    @abstractmethod
    def __init__(self, name: str, token: Optional[str]) -> None: ...

    @abstractmethod
    def upload_dataset(self, dataset_id: str) -> None: ...

    @abstractmethod
    def list_datasets(self, prefix: Optional[str] = None) -> Iterable[str]: ...

    @abstractmethod
    def get_dataset_metadata(self, dataset_id: str) -> dict: ...

    @abstractmethod
    def download_dataset(self, dataset_id: str, path: Path) -> None: ...

    @abstractmethod
    def list_namespaces(self) -> Iterable[str]: ...

    @abstractmethod
    def download_namespace_metadata(self, namespace: str, path: Path) -> None: ...

    @abstractmethod
    def upload_namespace(self, namespace: str) -> None: ...

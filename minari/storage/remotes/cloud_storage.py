from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Optional


class CloudStorage(ABC):
    """Abstract interface for cloud storage.

    This class is intended for internal use to interact with different remote storages.
    """

    @abstractmethod
    def __init__(self, name: str, token: Optional[str]) -> None:
        """Initialize the cloud storage.

        Args:
            name (str): name of the cloud storage.
            token (str): authentication token for the cloud storage.
                Notice, in case of GCP, this is the path to the service account key file.
        """
        ...

    @abstractmethod
    def upload_dataset(self, dataset_id: str) -> None:
        """Upload a local dataset to the remote server.

        Args:
            dataset_id (str): identifier for the local dataset.
        """
        ...

    @abstractmethod
    def list_datasets(self, prefix: Optional[str] = None) -> Iterable[str]:
        """List all datasets in the remote server.

        Args:
            prefix (str): filter datasets by prefix. Defaults to None.

        Returns:
            Iterable[str]: list of all datasets in the remote server.
        """
        ...

    @abstractmethod
    def get_dataset_metadata(self, dataset_id: str) -> dict:
        """Get metadata for a dataset in the remote server.

        Args:
            dataset_id (str): identifier for the remote dataset.

        Returns:
            dict: metadata for the remote dataset.
        """
        ...

    @abstractmethod
    def download_dataset(self, dataset_id: str, path: Path) -> None:
        """Download a remote dataset to local storage.

        Args:
            dataset_id (str): identifier for the remote dataset.
            path (Path): local path to download the dataset.
        """
        ...

    @abstractmethod
    def list_namespaces(self) -> Iterable[str]:
        """List all namespaces in the remote server.

        Returns:
            Iterable[str]: list of all namespaces in the remote server.
        """
        ...

    @abstractmethod
    def download_namespace_metadata(self, namespace: str, path: Path) -> None:
        """Download a remote namespace to local storage.

        Args:
            namespace (str): identifier for the remote namespace.
            path (Path): local path to download the namespace.
        """
        ...

    @abstractmethod
    def upload_namespace(self, namespace: str) -> None:
        """Upload a local namespace to the remote server.

        Args:
            namespace (str): identifier for the local namespace.
        """
        ...

import json
import os
from pathlib import Path
from typing import Iterable, Optional

from minari.dataset.minari_storage import METADATA_FILE_NAME
from minari.storage.datasets_root_dir import get_dataset_path
from minari.storage.remotes.cloud_storage import CloudStorage


try:
    import google.cloud.storage as gcp_storage
    from tqdm import tqdm
except ImportError:
    raise ImportError(
        'google-cloud-storage or tqdm are not installed. Please install it using `pip install "minari[gcs]"`'
    )


_NAMESPACE_METADATA_FILENAME = "namespace_metadata.json"


class GCPStorage(CloudStorage):
    def __init__(self, name: str, token: Optional[str] = None) -> None:
        if token is None:
            self.storage_client = gcp_storage.Client.create_anonymous_client()
        else:
            self.storage_client = gcp_storage.Client.from_service_account_json(
                json_credentials_path=token
            )
        self.bucket = gcp_storage.Bucket(self.storage_client, name)

    def upload_dataset(self, dataset_id: str) -> None:
        path = get_dataset_path(dataset_id)
        self._upload_directory(path, dataset_id)

    def _upload_directory(self, path: Path, remote_dir_path: str) -> None:
        for local_file in path.glob("*"):
            if local_file.is_dir():
                self._upload_directory(
                    local_file, remote_dir_path + "/" + local_file.name
                )
            else:
                remote_path = f"{remote_dir_path}/{local_file.name}"
                blob = self.bucket.blob(remote_path)
                blob.upload_from_filename(local_file)

    def list_datasets(self, prefix: Optional[str] = None) -> Iterable[str]:
        for blob in self.bucket.list_blobs(prefix=prefix):
            if os.path.basename(blob.name) == METADATA_FILE_NAME:
                yield os.path.dirname(os.path.dirname(blob.name))

    def get_dataset_metadata(self, dataset_id: str) -> dict:
        metadata_blob = os.path.join(dataset_id, "data", METADATA_FILE_NAME)
        metadata_blob = self.bucket.blob(metadata_blob)
        metadata = json.loads(
            metadata_blob.download_as_bytes(client=self.storage_client)
        )
        return metadata

    def download_dataset(self, dataset_id: str, path: Path) -> None:
        blobs = self.bucket.list_blobs(prefix=dataset_id)
        for blob in blobs:
            blob_path = Path(blob.name)
            file_path = path.joinpath(blob_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            print(f"\n * Downloading data file '{blob.name}' ...\n")
            with open(file_path, "wb") as f:
                with tqdm.wrapattr(f, "write", total=blob.size) as file_obj:
                    self.storage_client.download_blob_to_file(blob, file_obj)

    def list_namespaces(self) -> Iterable[str]:
        for blob in self.bucket.list_blobs():
            if os.path.basename(blob.name) == _NAMESPACE_METADATA_FILENAME:
                namespace = os.path.dirname(blob.name)
                yield namespace

    def download_namespace_metadata(self, namespace: str, path: Path) -> None:
        metadata_blob = os.path.join(namespace, _NAMESPACE_METADATA_FILENAME)
        metadata_blob = self.bucket.blob(metadata_blob)
        local_filename = path / namespace / _NAMESPACE_METADATA_FILENAME
        with open(local_filename, "wb") as f:
            self.storage_client.download_blob_to_file(metadata_blob, f)

    def upload_namespace(self, namespace: str) -> None:
        local_filepath = get_dataset_path(namespace) / _NAMESPACE_METADATA_FILENAME
        remote_filepath = f"{namespace}/{_NAMESPACE_METADATA_FILENAME}"
        blob = self.bucket.blob(remote_filepath)
        blob.upload_from_filename(local_filepath)

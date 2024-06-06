import os
from pathlib import Path
from typing import Any, Optional

from google.cloud import storage
from tqdm.auto import tqdm

from minari.storage.remotes.cloud_storage import CloudStorage


class GCPStorage(CloudStorage):
    def __init__(self, name: str, key_path: Optional[str] = None) -> None:
        if key_path is None:
            self.storage_client = storage.Client.create_anonymous_client()
        else:
            self.storage_client = storage.Client.from_service_account_json(
                json_credentials_path=key_path
            )
        self.bucket = storage.Bucket(self.storage_client, name)

    def upload_path(self, path: Path, dataset_id: str) -> None:
        # See https://github.com/googleapis/python-storage/issues/27 for discussion on progress bars
        for local_file in path.glob("**"):
            if not os.path.isfile(local_file):
                self.upload_path(
                    local_file, dataset_id + "/" + os.path.basename(local_file)
                )
            else:
                remote_path = os.path.join(dataset_id, local_file.name)
                blob = self.bucket.blob(remote_path)
                blob.upload_from_filename(local_file)

    def list_blobs(self, prefix: Optional[str] = None) -> list:
        return self.bucket.list_blobs(prefix=prefix)

    def download_blob(self, blob: Any, file_path: Path) -> None:
        with open(file_path, "wb") as f:
            with tqdm.wrapattr(f, "write", total=blob.size) as file_obj:
                self.storage_client.download_blob_to_file(blob, file_obj)

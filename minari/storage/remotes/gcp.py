import os
from pathlib import Path
from typing import Any, Iterable, Optional

from minari.storage.remotes.cloud_storage import CloudStorage


try:
    import google.cloud.storage as gcp_storage
    from tqdm import tqdm
except ImportError:
    raise ImportError(
        'google-cloud-storage or tqdm are not installed. Please install it using `pip install "minari[gcs]"`'
    )


class GCPStorage(CloudStorage):
    def __init__(self, name: str, key_path: Optional[str] = None) -> None:
        if key_path is None:
            self.storage_client = gcp_storage.Client.create_anonymous_client()
        else:
            self.storage_client = gcp_storage.Client.from_service_account_json(
                json_credentials_path=key_path
            )
        self.bucket = gcp_storage.Bucket(self.storage_client, name)

    def upload_directory(self, path: Path, remote_dir_path: str) -> None:
        # See https://github.com/googleapis/python-storage/issues/27 for discussion on progress bars
        for local_file in path.glob("*"):
            if local_file.is_dir():
                self.upload_directory(
                    local_file, remote_dir_path + "/" + os.path.basename(local_file)
                )
            else:
                remote_path = os.path.join(remote_dir_path, local_file.name)
                self.upload_file(local_file, remote_path)

    def upload_file(self, local_path: Path, remote_path: str) -> None:
        blob = self.bucket.blob(remote_path)
        blob.upload_from_filename(local_path)

    def list_blobs(self, prefix: Optional[str] = None) -> Iterable:
        return self.bucket.list_blobs(prefix=prefix)

    def download_blob(self, blob: Any, file_path: Path) -> None:
        with open(file_path, "wb") as f:
            with tqdm.wrapattr(f, "write", total=blob.size) as file_obj:
                self.storage_client.download_blob_to_file(blob, file_obj)

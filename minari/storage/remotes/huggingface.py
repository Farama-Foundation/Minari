import json
import os
import warnings
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple


try:
    from huggingface_hub import HfApi
    from huggingface_hub.utils import EntryNotFoundError
except ImportError:
    raise ImportError(
        'huggingface_hub is not installed. Please install it using `pip install "minari[hf]"`'
    )

from minari.dataset.minari_storage import METADATA_FILE_NAME
from minari.storage.datasets_root_dir import get_dataset_path
from minari.storage.remotes.cloud_storage import CloudStorage


_NAMESPACE_METADATA_FILENAME = "namespace_metadata.json"


class HuggingFaceStorage(CloudStorage):

    def __init__(self, name: str, token: Optional[str] = None) -> None:
        self.name = name
        self._api = HfApi(token=token)

    def _decompose_path(self, path: str) -> Tuple[str, str]:
        root, *rem = path.split("/")
        return root, "/".join(rem)

    def upload_dataset(self, dataset_id: str) -> None:
        path = get_dataset_path(dataset_id)
        repo_name, path_in_repo = self._decompose_path(dataset_id)
        repo_id = f"{self.name}/{repo_name}"

        self._api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
        self._api.upload_folder(
            repo_id=repo_id,
            folder_path=path,
            path_in_repo=path_in_repo,
            repo_type="dataset",
        )

        try:  # if it is a namespace, register dataset_id metadata
            repo_metadata = self._api.hf_hub_download(
                repo_id=repo_id,
                filename=_NAMESPACE_METADATA_FILENAME,
                repo_type="dataset",
            )

            with open(repo_metadata) as f:
                namespace_metadata = json.load(f)
            registered_datasets = namespace_metadata.get("datasets", [])
            registered_datasets.append(dataset_id)
            namespace_metadata["datasets"] = list(set(registered_datasets))
            with open(repo_metadata, "w") as f:
                json.dump(namespace_metadata, f)

            self._api.upload_file(
                path_or_fileobj=repo_metadata,
                path_in_repo=_NAMESPACE_METADATA_FILENAME,
                repo_id=repo_id,
                repo_type="dataset",
            )
        except EntryNotFoundError:
            pass

    def upload_namespace(self, namespace: str) -> None:
        local_filepath = get_dataset_path(namespace) / _NAMESPACE_METADATA_FILENAME
        repo_name, path_in_repo = self._decompose_path(namespace)
        repo_id = f"{self.name}/{repo_name}"

        self._api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

        if path_in_repo != "":
            repo_metadata_path = self._api.hf_hub_download(
                repo_id=repo_id,
                filename=_NAMESPACE_METADATA_FILENAME,
                repo_type="dataset",
            )

            with open(repo_metadata_path) as f:
                repo_metadata = json.load(f)
            registered_namespaces = repo_metadata.get("namespaces", [])
            registered_namespaces.append(namespace)
            repo_metadata["namespaces"] = list(set(registered_namespaces))
            with open(repo_metadata_path, "w") as f:
                json.dump(repo_metadata, f)

            self._api.upload_file(
                path_or_fileobj=repo_metadata_path,
                path_in_repo=_NAMESPACE_METADATA_FILENAME,
                repo_id=repo_id,
                repo_type="dataset",
            )

        self._api.upload_file(
            path_or_fileobj=local_filepath,
            path_in_repo=os.path.join(path_in_repo, _NAMESPACE_METADATA_FILENAME),
            repo_id=repo_id,
            repo_type="dataset",
        )

    def list_datasets(self, prefix: Optional[str] = None) -> Iterable[str]:
        if prefix is not None:
            group_name, _ = self._decompose_path(prefix)
        else:
            prefix = ""
            group_name = None

        hf_datasets = self._api.list_datasets(author=self.name, dataset_name=group_name)
        for group_info in hf_datasets:
            try:
                repo_metadata = self._api.hf_hub_download(
                    repo_id=group_info.id,
                    filename=_NAMESPACE_METADATA_FILENAME,
                    repo_type="dataset",
                )
            except EntryNotFoundError:
                try:
                    self._api.hf_hub_download(
                        repo_id=group_info.id,
                        filename=f"data/{METADATA_FILE_NAME}",
                        repo_type="dataset",
                    )
                    if group_info.id.startswith(prefix):
                        yield group_info.id
                except Exception:
                    warnings.warn(f"Skipping {group_info.id} as it is malformed.")
            else:
                with open(repo_metadata) as f:
                    namespace_metadata = json.load(f)

                group_datasets = namespace_metadata.get("datasets", [])
                group_datasets = filter(lambda x: x.startswith(prefix), group_datasets)
                yield from group_datasets

    def download_dataset(self, dataset_id: Any, path: Path) -> None:
        repo_id, path_in_repo = self._decompose_path(dataset_id)
        self._api.snapshot_download(
            repo_id=f"{self.name}/{repo_id}",
            allow_patterns=os.path.join(path_in_repo, "*"),
            repo_type="dataset",
            local_dir=path.joinpath(repo_id),
        )

    def get_dataset_metadata(self, dataset_id: str) -> dict:
        repo_id, path_in_repo = self._decompose_path(dataset_id)
        dataset_metadata = self._api.hf_hub_download(
            repo_id=f"{self.name}/{repo_id}",
            filename=Path(path_in_repo, "data", METADATA_FILE_NAME).as_posix(),
            repo_type="dataset",
        )
        with open(dataset_metadata) as f:
            metadata = json.load(f)
        return metadata

    def list_namespaces(self) -> Iterable[str]:
        for hf_dataset in self._api.list_datasets(author=self.name):
            try:
                repo_metadata = self._api.hf_hub_download(
                    repo_id=hf_dataset.id,
                    filename=_NAMESPACE_METADATA_FILENAME,
                    repo_type="dataset",
                )
            except EntryNotFoundError:
                continue
            else:
                with open(repo_metadata) as f:
                    namespace_metadata = json.load(f)

                repo_name = hf_dataset.id.split("/", 1)[1]
                yield from [repo_name] + namespace_metadata.get("namespaces", [])

    def download_namespace_metadata(self, namespace: str, path: Path) -> None:
        repo_id, path_in_repo = self._decompose_path(namespace)
        self._api.hf_hub_download(
            repo_id=f"{self.name}/{repo_id}",
            filename=Path(path_in_repo, _NAMESPACE_METADATA_FILENAME).as_posix(),
            repo_type="dataset",
            local_dir=path.joinpath(repo_id),
            force_download=True,
        )

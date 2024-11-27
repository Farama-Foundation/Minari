import copy
import json
import os
import re
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from minari.storage import get_dataset_path
from minari.storage.hosting import get_cloud_storage
from minari.storage.local import list_non_hidden_dirs


NAMESPACE_REGEX = re.compile(r"[-_\w][-_\w/]*[-_\w]+")
NAMESPACE_METADATA_FILENAME = "namespace_metadata.json"


def create_namespace(
    namespace: str,
    description: Optional[str] = None,
    **kwargs,
) -> None:
    """Create a local namespace.

    Namespaces are a directory-like structure that can contain multiple datasets, or
    other namespaces. Namespaces are prepended onto a ``dataset_id`` with a forward
    slash. For example a dataset with id ``cartpole/test-v0`` resides in
    the ``cartpole`` namespace. Namespaces can be nested.

    Note: The namespace API is an experimental feature and may change in future releases.

    Args:
        namespace (str): identifier for namespace created/updated.
        description (str | None): string metadata describing the namespace. Defaults to None.
        **kwargs: any other metadata in addition to the description.
    """
    validate_namespace(namespace)

    if namespace in list_local_namespaces():
        raise ValueError(f"Namespace '{namespace}' already exists.")

    metadata = copy.deepcopy(kwargs)
    if description is not None:
        metadata["description"] = description
    directory = get_dataset_path(namespace)
    os.makedirs(directory, exist_ok=True)

    with open(directory / NAMESPACE_METADATA_FILENAME, "w") as file:
        json.dump(metadata, file)

    local_namespaces = list_local_namespaces()
    for parent_namespace in namespace_hierarchy(namespace):
        if parent_namespace not in local_namespaces:
            parent_namespace_path = get_dataset_path(parent_namespace)
            with open(parent_namespace_path / NAMESPACE_METADATA_FILENAME, "w") as file:
                json.dump({}, file)


def update_namespace_metadata(
    namespace: str,
    description: Optional[str] = None,
    **kwargs,
) -> None:
    """Update an existing local namespace, overwriting existing namespace metadata.

    Note: The namespace API is an experimental feature and may change in future releases.

    Args:
        namespace (str): identifier for namespace created/updated.
        description (str | None): string metadata describing the namespace. Defaults to None.
        **kwargs: any other metadata in addition to the description.
    """
    validate_namespace(namespace)

    if namespace not in list_local_namespaces():
        raise ValueError(f"Namespace {namespace} does not exist locally.")

    metadata = copy.deepcopy(kwargs)
    metadata["description"] = description
    metadata_filepath = get_dataset_path(namespace) / NAMESPACE_METADATA_FILENAME

    with open(metadata_filepath, "w") as file:
        json.dump(metadata, file)


def get_namespace_metadata(namespace: str) -> Dict[str, Any]:
    """Load local namespace metadata.

    Note: The namespace API is an experimental feature and may change in future releases.

    Args:
        namespace (str): identifier for local namespace.

    Returns:
        Dict[str, Any]: metadata dict.
    """
    validate_namespace(namespace)

    if namespace not in list_local_namespaces():
        raise ValueError(f"Namespace {namespace} does not exist locally.")

    metadata_filepath = get_dataset_path(namespace) / NAMESPACE_METADATA_FILENAME

    with open(metadata_filepath) as file:
        metadata = json.load(file)

    return metadata


def delete_namespace(namespace: str) -> None:
    """Delete local namespace. Only empty namespaces can be deleted.

    Note: The namespace API is an experimental feature and may change in future releases.

    Args:
        namespace (str): identifier for local namespace.
    """
    validate_namespace(namespace)

    if namespace not in list_local_namespaces():
        warnings.warn(f"Namespace '{namespace}' does not exist.", UserWarning)
        return

    directory = get_dataset_path(namespace)
    assert os.path.isdir(directory)
    dir_contents = os.listdir(directory)
    has_metadata = NAMESPACE_METADATA_FILENAME in dir_contents

    if len(dir_contents) != int(has_metadata):
        raise ValueError(
            f"Namespace {directory} is not empty. All datasets must be deleted first."
        )

    if has_metadata:
        os.remove(directory / NAMESPACE_METADATA_FILENAME)

    os.rmdir(directory)


def list_local_namespaces() -> List[str]:
    """Get the names of the namespaces in the local database.

    Note: The namespace API is an experimental feature and may change in future releases.

    Returns:
       List[str]: names of all local namespaces.
    """
    datasets_path = get_dataset_path()
    namespaces = []

    def recurse_directories(base_path: Path, namespace):
        parent_dir = base_path.joinpath(namespace)
        for dir_name in list_non_hidden_dirs(parent_dir):
            dir_path = os.path.join(parent_dir, dir_name)
            namespaced_dir_name = os.path.join(namespace, dir_name)
            dir_contents = os.listdir(dir_path)

            if NAMESPACE_METADATA_FILENAME in dir_contents:
                namespaces.append(namespaced_dir_name)

            # Don't recurse the subdirectories of a Minari dataset
            if "data" not in dir_contents:
                recurse_directories(base_path, namespaced_dir_name)

    recurse_directories(datasets_path, "")

    return sorted(namespaces)


def list_remote_namespaces() -> List[str]:
    """Get the names of the namespaces in the remote server.

    Note: The namespace API is an experimental feature and may change in future releases.

    Returns:
       List[str]: names of all remote namespaces.
    """
    cloud_storage = get_cloud_storage()
    remote_namespaces = cloud_storage.list_namespaces()
    return list(remote_namespaces)


def download_namespace_metadata(namespace: str, overwrite: bool = False) -> None:
    """Download remote namespace to local database.

    Note: The namespace API is an experimental feature and may change in future releases.

    Args:
        namespace (str): identifier for the remote namespace.
        overwrite (bool): whether to overwrite existing local metadata. Defaults to False.
    """
    validate_namespace(namespace)
    if namespace not in list_remote_namespaces():
        raise ValueError(
            f"The namespace '{namespace}' doesn't exist in the remote Farama server."
        )

    cloud_storage = get_cloud_storage()

    if overwrite or namespace not in list_local_namespaces():
        data_path = get_dataset_path()
        (data_path / namespace).mkdir(parents=True, exist_ok=True)
        cloud_storage.download_namespace_metadata(namespace, data_path)
    else:
        warnings.warn(
            f"Skipping update for namespace '{namespace}' due to existing local metadata. Set overwrite=True to overwrite local data.",
            UserWarning,
        )


def upload_namespace(namespace: str, token: str) -> None:
    """Upload a local namespace to the remote server.

    If you would like to upload a namespace please first get in touch with the Farama team at contact@farama.org.

    Note: The namespace API is an experimental feature and may change in future releases.

    Args:
        namespace (str): identifier for the local namespace.
        token (str): authentication token for the remote server.
            Notice, that for GCP, this is the path to the service account key file, while for Hugging Face, this is the API token.
    """
    validate_namespace(namespace)
    local_namespaces = list_local_namespaces()
    remote_namespaces = list_remote_namespaces()
    if namespace not in local_namespaces:
        raise ValueError(f"Namespace '{namespace}' does not exist locally.")
    if namespace in remote_namespaces:
        warnings.warn(
            f"Upload aborted. Namespace '{namespace}' is already in remote.",
            UserWarning,
        )
        return

    cloud_storage = get_cloud_storage(token=token)
    for parent_namespace in namespace_hierarchy(namespace):
        if (
            parent_namespace in local_namespaces
            and parent_namespace not in remote_namespaces
        ):
            print(f"Uploading namespace '{parent_namespace}'")
            cloud_storage.upload_namespace(parent_namespace)


def namespace_hierarchy(namespace: Optional[str]) -> Iterable[str]:
    """Get all parent namespaces of a given namespace.

    Args:
        namespace (str): identifier for the local namespace.

    Returns:
        Iterable[str]: names of all parent namespaces.
    """
    if namespace is None:
        return []

    namespace_parts = namespace.split(os.sep)
    for i in range(len(namespace_parts)):
        yield os.path.join(*namespace_parts[: i + 1])


def validate_namespace(namespace: Optional[str]) -> None:
    """Validate a namespace identifier.

    Note: The namespace API is an experimental feature and may change in future releases.

    Args:
        namespace (str | None): identifier to validate
    """
    if namespace is None:
        raise TypeError("Namespace cannot be None")

    if not NAMESPACE_REGEX.fullmatch(namespace):
        raise ValueError(f"Malformed namespace: {namespace}")

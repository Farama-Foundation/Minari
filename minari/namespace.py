import copy
import json
import os
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

from minari.storage import get_dataset_path
from minari.storage.hosting import get_cloud_storage
from minari.storage.local import list_non_hidden_dirs


NAMESPACE_REGEX = re.compile(r"[-_\w][-_\w/]*[-_\w]+")
NAMESPACE_METADATA_FILE_NAME = "namespace_metadata.json"


def create_namespace(
    namespace: str,
    description: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    overwrite: bool = False,
) -> None:
    """Create or update a local namespace.

    Namespaces are a directory-like structure that can contain multiple datasets, or
    other namespaces. Namespaces are prepended onto a ``dataset_id`` with a forward
    slash. For example a dataset with id ``my_namespace/cartpole-test-v0`` resides in
    the ``my_namespace`` namespace. Namespaces can be nested.

    Args:
        namespace (str): identifier for namespace created/updated.
        description (str | None): string metadata describing the namespace. Defaults to None.
        metadata (Dict[str, Any] | None): extra metadata in addition to the description. Defaults to None.
        overwrite (bool): whether to overwrite existing namespace data. Defaults to False.
    """
    validate_namespace(namespace)
    metadata = {} if metadata is None else copy.deepcopy(metadata)

    if description is not None:
        if "description" not in metadata:
            metadata["description"] = description

        if description != metadata["description"]:
            raise ValueError(
                "Namespace description conflicts with metadata['description']."
            )

    namespace_exists = namespace in list_local_namespaces()

    if namespace_exists:
        existing_metadata = get_namespace_metadata(namespace)

        # Recreating a namespace with the same metadata is a no-op
        if existing_metadata == metadata:
            return

        if existing_metadata is not None and not overwrite:
            raise ValueError(
                f"Metadata for namespace '{namespace}' already exists. Set overwrite=True to overwrite existing metadata."
            )

    directory = os.path.join(get_dataset_path(""), namespace)
    metadata_filepath = os.path.join(directory, NAMESPACE_METADATA_FILE_NAME)
    os.makedirs(directory, exist_ok=True)

    with open(metadata_filepath, "w") as file:
        json.dump(metadata, file)


def update_namespace(
    namespace: str,
    description: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Update an existing local namespace. Overwrites existing namespace data.

    Args:
        namespace (str): identifier for namespace created/updated.
        description (str | None): string metadata describing the namespace. Defaults to None.
        metadata (Dict[str, Any] | None): extra metadata in addition to the description. Defaults to None.
    """
    if namespace not in list_local_namespaces():
        raise ValueError(f"Namespace {namespace} does not exist locally.")

    return create_namespace(namespace, description, metadata, overwrite=True)


def get_namespace_metadata(namespace: str) -> Optional[Dict[str, Any]]:
    """Load local namespace metadata.

    Args:
        namespace (str): identifier for local namespace.

    Returns:
        Optional[Dict[str, Any]]: metadata dict, or None if metadata is empty.
    """
    validate_namespace(namespace)

    if namespace not in list_local_namespaces():
        raise ValueError(f"Namespace {namespace} does not exist locally.")

    filepath = os.path.join(
        get_dataset_path(""), namespace, NAMESPACE_METADATA_FILE_NAME
    )

    with open(filepath) as file:
        metadata = json.load(file)

    return metadata if metadata != {} else None


def delete_namespace(namespace: str) -> None:
    """Delete local namespace. Only empty namespaces can be deleted.

    Args:
        namespace (str): identifier for local namespace.
    """
    validate_namespace(namespace)

    if namespace not in list_local_namespaces():
        raise ValueError(f"Namespace '{namespace}' does not exist.")

    directory = os.path.join(get_dataset_path(""), namespace)

    if not os.path.isdir(directory):
        raise FileNotFoundError(f"No namespace found at {directory}.")

    dir_contents = os.listdir(directory)
    has_metadata = NAMESPACE_METADATA_FILE_NAME in dir_contents

    if len(dir_contents) != int(has_metadata):
        raise ValueError(
            f"Namespace {directory} is not empty. All datasets must be deleted first."
        )

    if has_metadata:
        os.remove(os.path.join(directory, NAMESPACE_METADATA_FILE_NAME))

    os.rmdir(directory)


def list_local_namespaces() -> List[str]:
    """Get the names of the namespaces in the local database.

    Returns:
       List[str]: names of all local namespaces.
    """
    datasets_path = get_dataset_path("")
    namespaces = []

    def recurse_directories(base_path, namespace):
        parent_dir = os.path.join(base_path, namespace)
        for dir_name in list_non_hidden_dirs(parent_dir):
            dir_path = os.path.join(parent_dir, dir_name)
            namespaced_dir_name = os.path.join(namespace, dir_name)
            dir_contents = os.listdir(dir_path)

            if NAMESPACE_METADATA_FILE_NAME in dir_contents:
                namespaces.append(namespaced_dir_name)

            # Don't recurse the subdirectories of a Minari dataset
            if "data" not in dir_contents:
                recurse_directories(base_path, namespaced_dir_name)

    recurse_directories(datasets_path, "")

    return sorted(namespaces)


def list_remote_namespaces() -> List[str]:
    """Get the names of the namespaces in the remote server.

    Returns:
       List[str]: names of all remote namespaces.
    """
    cloud_storage = get_cloud_storage()
    blobs = cloud_storage.list_blobs()

    remote_namespaces = []

    for blob in blobs:
        if os.path.basename(blob.name) == NAMESPACE_METADATA_FILE_NAME:
            namespace = os.path.dirname(blob.name)
            remote_namespaces.append(namespace)

    return remote_namespaces


def download_namespace_metadata(namespace: str, overwrite: bool = False) -> None:
    """Download remote namespace to local database.

    Args:
        namespace (str): identifier for the remote namespace.
        overwrite (bool): whether to overwrite existing local metadata. Defaults to False.
    """
    validate_namespace(namespace)

    if namespace not in list_remote_namespaces():
        raise ValueError(
            f"The namespace '{namespace}' doesn't exist in the remote Farama server."
        )

    metadata_filename = os.path.join(namespace, NAMESPACE_METADATA_FILE_NAME)
    cloud_storage = get_cloud_storage()
    blobs = list(cloud_storage.list_blobs(prefix=metadata_filename))

    if not blobs:
        return

    assert len(blobs) == 1
    remote_metadata = json.loads(blobs[0].download_as_bytes(client=None))
    local_metadata = None

    if namespace in list_local_namespaces():
        local_metadata = get_namespace_metadata(namespace)

    if local_metadata != remote_metadata:
        if overwrite or local_metadata is None:
            create_namespace(namespace, metadata=remote_metadata, overwrite=True)
        else:
            warnings.warn(
                f"Skipping update for namespace '{namespace}' due to existing local metadata. Set overwrite=True to overwrite local data.",
                UserWarning,
            )


def upload_namespace(namespace: str, key_path: str) -> None:
    """Upload a local namespace to the remote server.

    If you would like to upload a namespace please first get in touch with the Farama team at contact@farama.org.

    Args:
        namespace (str): identifier for the local namespace.
        key_path (str): path to the credentials file.
    """
    validate_namespace(namespace)

    if namespace not in list_local_namespaces():
        raise ValueError(f"Namespace '{namespace}' does not exist locally.")

    if namespace in list_remote_namespaces():
        print(f"Upload aborted. Namespace '{namespace}' is already in remote.")
        return

    cloud_storage = get_cloud_storage(key_path=key_path)

    print(f"Uploading namespace '{namespace}'")

    namespace_metadata_path = os.path.join(namespace, NAMESPACE_METADATA_FILE_NAME)
    local_file_path = Path(get_dataset_path(""), namespace_metadata_path)

    cloud_storage.upload_file(local_file_path, namespace_metadata_path)


def validate_namespace(namespace: str) -> None:
    if namespace is None:
        raise TypeError("Namespace cannot be None")

    if not isinstance(namespace, str):
        raise TypeError(f"Namespace must be str, not {type(namespace)}")

    if not NAMESPACE_REGEX.fullmatch(namespace):
        raise ValueError(f"Malformed namespace: {namespace}")

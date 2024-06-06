import os
from typing import Dict, Type

from minari.storage.remotes.cloud_storage import CloudStorage

from .gcp import GCPStorage


DEFAULT_REMOTE = "gcp://minari-datasets"

registry: Dict[str, Type[CloudStorage]] = {"gcp": GCPStorage}


def get_cloud_storage(key_path=None) -> CloudStorage:
    remote_spec = os.getenv("MINARI_REMOTE", DEFAULT_REMOTE)
    cloud_type, name = remote_spec.split("://", maxsplit=1)
    return registry[cloud_type](name, key_path)

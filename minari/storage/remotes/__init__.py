import os
from typing import Callable, Dict, Type

from minari.storage.remotes.cloud_storage import CloudStorage


DEFAULT_REMOTE = "gcp://minari-remote"


def get_gcps() -> Type[CloudStorage]:
    from .gcp import GCPStorage

    return GCPStorage


_registry: Dict[str, Callable[[], Type[CloudStorage]]] = {"gcp": get_gcps}


def get_cloud_storage(key_path=None) -> CloudStorage:
    remote_spec = os.getenv("MINARI_REMOTE", DEFAULT_REMOTE)
    cloud_type, name = remote_spec.split("://", maxsplit=1)
    return _registry[cloud_type]()(name, key_path)

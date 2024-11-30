import os
from typing import Callable, Dict, Optional, Type

from minari.storage.remotes.cloud_storage import CloudStorage


DEFAULT_REMOTE = "hf://farama-minari"


def get_gcps() -> Type[CloudStorage]:
    from .gcp import GCPStorage

    return GCPStorage


def get_hfs() -> Type[CloudStorage]:
    from .huggingface import HuggingFaceStorage

    return HuggingFaceStorage


_registry: Dict[str, Callable[[], Type[CloudStorage]]] = {
    "gcp": get_gcps,
    "hf": get_hfs,
}


def get_cloud_storage(
    remote_path: Optional[str] = None, token: Optional[str] = None
) -> CloudStorage:
    if remote_path is None:
        remote_path = os.getenv("MINARI_REMOTE", DEFAULT_REMOTE)
    cloud_type, name = remote_path.split("://", maxsplit=1)
    return _registry[cloud_type]()(name, token)

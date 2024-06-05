from typing import Dict, Type

from minari.storage.remotes.cloud_storage import CloudStorage

from .gcp import GCPStorage


registry: Dict[str, Type[CloudStorage]] = {
    "gcp": GCPStorage
}


def get_cloud_storage(key_path=None) -> CloudStorage:
    return registry["gcp"]("minari-datasets", key_path)

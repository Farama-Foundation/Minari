from typing import Dict, Type
from .hdf5_storage import _HDF5Storage

from minari.dataset.minari_storage import MinariStorage

registry: Dict[str, Type[MinariStorage]] = {
    "hdf5": _HDF5Storage,
}
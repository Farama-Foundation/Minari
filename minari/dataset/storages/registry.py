from typing import Dict, Type

from minari.dataset.minari_storage import MinariStorage

from .hdf5_storage import _HDF5Storage


registry: Dict[str, Type[MinariStorage]] = {
    "hdf5": _HDF5Storage,
}

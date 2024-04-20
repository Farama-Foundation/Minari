from typing import Dict, Type

from minari.dataset.minari_storage import MinariStorage

from .arrow_storage import ArrowStorage
from .hdf5_storage import HDF5Storage


registry: Dict[str, Type[MinariStorage]] = {
    "hdf5": HDF5Storage,
    "arrow": ArrowStorage,
}

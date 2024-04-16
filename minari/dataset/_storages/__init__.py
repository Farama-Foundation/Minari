from typing import Dict, Type

from minari.dataset.minari_storage import MinariStorage

from .hdf5_storage import HDF5Storage
from .arrow_storage import ArrowStorage


registry: Dict[str, Type[MinariStorage]] = {
    "hdf5": HDF5Storage,
    "arrow": ArrowStorage,  
}

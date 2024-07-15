from typing import Callable, Dict, Type

from minari.dataset.minari_storage import MinariStorage


def get_arrow_storage() -> Type[MinariStorage]:
    from .arrow_storage import ArrowStorage

    return ArrowStorage


def get_hdf5_storage() -> Type[MinariStorage]:
    from .hdf5_storage import HDF5Storage

    return HDF5Storage


registry: Dict[str, Callable[[], Type[MinariStorage]]] = {
    "arrow": get_arrow_storage,
    "hdf5": get_hdf5_storage,
}

from typing import Callable, Dict, List, Type

from minari.dataset.minari_storage import MinariStorage


def get_arrow_storage() -> Type[MinariStorage]:
    from .arrow_storage import ArrowStorage

    return ArrowStorage


def get_hdf5_storage() -> Type[MinariStorage]:
    from .hdf5_storage import HDF5Storage

    return HDF5Storage


_registry: Dict[str, Callable[[], Type[MinariStorage]]] = {
    "arrow": get_arrow_storage,
    "hdf5": get_hdf5_storage,
}


def get_minari_storage(storage_type: str) -> Type[MinariStorage]:
    storage = _registry[storage_type]()
    assert (
        storage.FORMAT == storage_type
    ), f"Storage type mismatch: {storage.FORMAT} != {storage_type}"
    return storage


def get_storage_keys() -> List[str]:
    return list(_registry.keys())

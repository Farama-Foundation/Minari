from typing import Callable, Dict, Type

from minari.dataset.minari_storage import MinariStorage


def get_arrow_storage() -> Type[MinariStorage]:
    from .arrow_storage import ArrowStorage

    return ArrowStorage


def get_hdf5_storage() -> Type[MinariStorage]:
    from .hdf5_storage import HDF5Storage

    return HDF5Storage


registry_factory: Dict[str, Callable[[], Type[MinariStorage]]] = {
    "arrow": get_arrow_storage,
    "hdf5": get_hdf5_storage,
}


class RegistryDict(dict):
    def __getitem__(self, key: str) -> Type[MinariStorage]:
        # Custom behavior
        if key in self:
            value = super().__getitem__(key)
            return value
        else:
            try:
                value = registry_factory[key]()
                self[key] = value
                return value
            except KeyError:
                raise KeyError(f"Storage type {key} not supported")


registry: RegistryDict = RegistryDict()

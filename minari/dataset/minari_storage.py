from __future__ import annotations

import os
import pathlib
from collections import OrderedDict
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import gymnasium as gym
from gymnasium.envs.registration import EnvSpec
import h5py
import numpy as np

from minari.serialization import deserialize_space, serialize_space


PathLike = Union[str, os.PathLike]


class MinariStorage:
    """Class that handles disk access to the data."""

    def __init__(self, data_path: PathLike):
        """Initialize a MinariStorage with an existing data path.
        To create a new dataset, use the class method `new`.

        Args:
            data_path (str or Path): directory containing the data.

        Raises:
            ValueError: if the specified path doesn't exist or doesn't contain any data.

        """
        if not os.path.exists(data_path) or not os.path.isdir(data_path):
            raise ValueError(f"The data path {data_path} doesn't exist")
        file_path = os.path.join(str(data_path), "main_data.hdf5")
        if not os.path.exists(file_path):
            raise ValueError(f"No data found in data path {data_path}")
        self._file_path = file_path

    @classmethod
    def new(
        cls,
        data_path: PathLike,
        observation_space: Optional[gym.Space] = None,
        action_space: Optional[gym.Space] = None,
        env_spec: Optional[EnvSpec] = None
    ) -> MinariStorage:
        """Class method to create a new data storage. 

        Args:
            data_path (str or Path): directory where the data will be stored. 
            observation_space (gymnasium.Space, optional): Gymnasium observation space of the dataset.
            action_space (gymnasium.Space, optional): Gymnasium action space of the dataset.
            env_spec (EnvSpec, optional): Gymnasium EnvSpec of the environment that generates the dataset.

        Returns:
            A new MinariStorage object. 
        """
        data_path = pathlib.Path(data_path)
        data_path.mkdir(exist_ok=True)
        data_path.joinpath("main_data.hdf5").touch(exist_ok=False)
    
        obj = cls(data_path)
        metadata: Dict[str, Any] = {
            "total_episodes": 0,
            "total_steps": 0
        }
        if observation_space is not None:
            metadata["observation_space"] = serialize_space(observation_space)
        if action_space is not None:
            metadata["action_space"] = serialize_space(action_space)
        if env_spec is not None: 
            metadata["env_spec"] = env_spec.to_json()

        obj.update_metadata(metadata)
        return obj

    @property
    def metadata(self) -> Dict:
        """Metadata of the dataset."""
        metadata = {}
        with h5py.File(self._file_path, "r") as file:
            metadata.update(file.attrs)
            if "observation_space" in metadata.keys():
                space_serialization = metadata["observation_space"]
                assert isinstance(space_serialization, str)
                metadata["observation_space"] = deserialize_space(space_serialization)            
            if "action_space" in metadata.keys():
                space_serialization = metadata["action_space"]
                assert isinstance(space_serialization, str)
                metadata["action_space"] = deserialize_space(space_serialization)
            
            return metadata
    
    def update_metadata(self, metadata: Dict):
        """Update the metadata adding/modifying some keys.
        
        Args:
            metadata (dict): dictionary of keys-values to add to the metadata.
        """
        with h5py.File(self._file_path, "a") as file:
            file.attrs.update(metadata)

    def update_episode_metadata(self, metadatas: Iterable[Dict], episode_indices: Optional[Iterable] = None):
        """Update the metadata of episodes.

        Args:
            metadatas (Iterable[Dict]): metadatas, one for each episode.
            episode_indices (Iterable, optional): episode indices to update. 
            If not specified, all the episodes are considered.
        
        Warning:
            In case metadatas and episode_indices have different lengths, the longest is truncated silently.
        """
        if episode_indices is None:
            episode_indices = range(self.total_episodes)
    
        with h5py.File(self._file_path, "a") as file:
            for metadata, episode_id in zip(metadatas, episode_indices):
                ep_group = file[f"episode_{episode_id}"]
                ep_group.attrs.update(metadata)

    def apply(
        self,
        function: Callable[[dict], Any],
        episode_indices: Optional[Iterable] = None,
    ) -> Iterable[Any]:
        """Apply a function to a slice of the data.

        Args:
            function (Callable): function to apply to episodes
            episode_indices (Optional[Iterable]): episodes id to consider

        Returns:
            outs (Iterable): outputs returned by the function applied to episodes
        """
        if episode_indices is None:
            episode_indices = range(self.total_episodes)
        
        ep_dicts = self.get_episodes(episode_indices)
        return map(function, ep_dicts)

    def _decode_space(
        self,
        hdf_ref: Union[h5py.Group, h5py.Dataset, h5py.Datatype],
        space: gym.spaces.Space,
    ) -> Union[Dict, Tuple, List, np.ndarray]:
        assert not isinstance(hdf_ref, h5py.Datatype)

        if isinstance(space, gym.spaces.Tuple):
            assert isinstance(hdf_ref, h5py.Group)
            result = []
            for i in range(len(hdf_ref.keys())):
                result.append(
                    self._decode_space(hdf_ref[f"_index_{i}"], space.spaces[i])
                )
            return tuple(result)
        elif isinstance(space, gym.spaces.Dict):
            assert isinstance(hdf_ref, h5py.Group)
            result = {}
            for key in hdf_ref.keys():
                result[key] = self._decode_space(hdf_ref[key], space.spaces[key])
            return result
        elif isinstance(space, gym.spaces.Text):
            assert isinstance(hdf_ref, h5py.Dataset)
            result = map(lambda string: string.decode("utf-8"), hdf_ref[()])
            return list(result)
        else:
            assert isinstance(hdf_ref, h5py.Dataset)
            return hdf_ref[()]

    def get_episodes(self, episode_indices: Iterable[int]) -> List[dict]:
        """Get a list of episodes.

        Args:
            episode_indices (Iterable[int]): episodes id to return

        Returns:
            episodes (List[dict]): list of episodes data
        """
        out = []
        with h5py.File(self._file_path, "r") as file:
            for ep_idx in episode_indices:
                ep_group = file[f"episode_{ep_idx}"]
                assert isinstance(ep_group, h5py.Group)
                
                ep_dict = {
                    "id": ep_group.attrs.get("id"),
                    "total_timesteps": ep_group.attrs.get("total_steps"),
                    "seed": ep_group.attrs.get("seed"),
                    "observations": self._decode_space(
                        ep_group["observations"], self.metadata["observation_space"]  # TODO: metadata can be slow
                    ),
                    "actions": self._decode_space(
                        ep_group["actions"], self.metadata["action_space"]
                    ),
                }
                for key in {"rewards", "terminations", "truncations"}:
                    group_value = ep_group[key]
                    assert isinstance(group_value, h5py.Dataset)
                    ep_dict[key] = group_value[:]

                out.append(ep_dict)

        return out

    def update_episodes(self, episodes: Iterable[dict]):
        """Update epsiodes in the storage from a list of episode buffer.

        Args:
            episodes (Iterable[dict]): list of episodes buffer.
            They must contain the keys specified in EpsiodeData dataclass, except for `id` which is optional.
            If `id` is specified and exists, the new data is appended to the one in the storage. 
        """
        additional_steps = 0
        with h5py.File(self._file_path, "a", track_order=True) as file:
            for eps_buff in episodes:
                total_episodes = len(file.keys())
                episode_id = eps_buff.pop("id", total_episodes)
                assert episode_id <= total_episodes, "Invalid episode id; ids must be sequential."
                episode_group = _get_from_h5py(file, f"episode_{episode_id}")
                episode_group.attrs["id"] = episode_id
                if "seed" in eps_buff.keys():
                    assert not "seed" in episode_group.attrs.keys()
                    episode_group.attrs["seed"] = eps_buff.pop("seed")
                total_steps = len(eps_buff["rewards"])
                episode_group.attrs["total_steps"] = total_steps
                additional_steps += total_steps

                _add_episode_to_group(eps_buff, episode_group)

            current_steps = file.attrs["total_steps"]
            assert type(current_steps) == np.int64
            total_steps = current_steps + additional_steps
            total_episodes = len(file.keys())

            file.attrs.modify("total_episodes", total_episodes)
            file.attrs.modify("total_steps", total_steps)

    def update_from_storage(self, storage: MinariStorage, copy: bool = False):
        """Update the dataset using another MinariStorage.

        Args:
            storage (MinariStorage): the other MinariStorage from which the data will be taken
            copy (bool): whether to copy the data or create a link. Default value is false.
        """
        with h5py.File(self._file_path, "a", track_order=True) as file:
            last_episode_id = file.attrs["total_episodes"]
            assert type(last_episode_id) == np.int64
            storage_total_episodes = storage.total_episodes

            if copy:
                for id in range(storage.total_episodes):
                    episode = storage.get_episodes([id])
                    episode[0].pop("id")
                    self.update_episodes(episode)
            else:
                for id in range(storage_total_episodes):
                    file[f"episode_{last_episode_id + id}"] = h5py.ExternalLink(storage._file_path, f"/episode_{id}")
                    file[f"episode_{last_episode_id + id}"].attrs.modify(  # TODO: check it doesn't modify original dataset
                        "id", last_episode_id + id
                    )             

            file.attrs.modify("total_episodes", last_episode_id + storage_total_episodes)
            total_steps = file.attrs["total_steps"]
            assert type(total_steps) == np.int64
            file.attrs.modify("total_steps", total_steps + storage.total_steps)

            storage_metadata = storage.metadata
            file.attrs.modify("author", f'{file.attrs["author"]}; {storage_metadata["author"]}')
            file.attrs.modify("author_email", f'{file.attrs["author_email"]}; {storage_metadata["author_email"]}')

    @property
    def data_path(self) -> PathLike:
        """Full path to the `main_data.hdf5` file of the dataset."""
        return os.path.dirname(self._file_path)

    @property
    def total_episodes(self) -> np.int64:
        """Total episodes in the dataset."""
        with h5py.File(self._file_path, "r") as file:
            total_episodes = file.attrs["total_episodes"]
            assert type(total_episodes) == np.int64
            return total_episodes

    @property
    def total_steps(self) -> np.int64:
        """Total steps in the dataset."""
        with h5py.File(self._file_path, "r") as file:
            total_steps = file.attrs["total_steps"]
            assert type(total_steps) == np.int64
            return total_steps

def _get_from_h5py(group: h5py.Group, name: str) -> h5py.Group:
    if name in group:
        subgroup = group.get(name)
        assert isinstance(subgroup, h5py.Group)
    else:
        subgroup = group.create_group(name)

    return subgroup

def _add_episode_to_group(episode_buffer: Dict, episode_group: h5py.Group):
    for key, data in episode_buffer.items():
        if isinstance(data, dict):
            episode_group_to_clear = _get_from_h5py(episode_group, key)
            _add_episode_to_group(data, episode_group_to_clear)
        elif all([isinstance(entry, tuple) for entry in data]):  # list of tuples
            dict_data = {
                f"_index_{str(i)}": [entry[i] for entry in data]
                for i, _ in enumerate(data[0])
            }
            episode_group_to_clear = _get_from_h5py(episode_group, key)
            _add_episode_to_group(dict_data, episode_group_to_clear)
        elif all([isinstance(entry, OrderedDict) for entry in data]):  # list of OrderedDict
            dict_data = {
                key: [entry[key] for entry in data] for key in data[0].keys()
            }
            episode_group_to_clear = _get_from_h5py(episode_group, key)
            _add_episode_to_group(dict_data, episode_group_to_clear)
       
        # leaf data
        elif key in episode_group:
            dataset = episode_group[key]
            assert isinstance(dataset, h5py.Dataset)
            dataset.resize((dataset.shape[0] + len(data), *dataset.shape[1:]))
            dataset[-len(data):] = data
        else:
            dtype = None
            if all(map(lambda elem: isinstance(elem, str), data)):
                dtype = h5py.string_dtype(encoding="utf-8")
            dshape = ()
            if hasattr(data[0], "shape"):
                dshape = data[0].shape
            episode_group.create_dataset(key, data=data, dtype=dtype, chunks=True, maxshape=(None, *dshape))
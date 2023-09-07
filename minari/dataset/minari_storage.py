import os
import pathlib
from collections import OrderedDict
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import gymnasium as gym
import h5py
import numpy as np

from minari.serialization import deserialize_space, serialize_space


PathLike = Union[str, os.PathLike]


class MinariStorage:
    def __init__(self, data_path: PathLike):
        if not os.path.exists(data_path) or not os.path.isdir(data_path):
            raise ValueError(f"The data path {data_path} doesn't exists")
        file_path = os.path.join(str(data_path), "main_data.hdf5")
        if not os.path.exists(file_path):
            raise ValueError(f"No data found in data path {data_path}")
        self._file_path = file_path

    @classmethod
    def new(cls, data_path: PathLike, action_space, observation_space, env_spec=None):
        data_path = pathlib.Path(data_path)
        data_path.mkdir(exist_ok=True)
        data_path.joinpath("main_data.hdf5").touch(exist_ok=False)
    
        obj = cls(data_path)
        metadata = {
            "action_space": serialize_space(action_space),
            "observation_space": serialize_space(observation_space),
            "total_episodes": 0,
            "total_steps": 0
        }
        if env_spec is not None: 
            metadata["env_spec"] = env_spec.to_json()

        obj.update_metadata(metadata)
        return obj

    @property
    def metadata(self) -> Dict:
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
        with h5py.File(self._file_path, "a") as file:
            file.attrs.update(metadata)

    def update_episode_metadata(self, metadatas: List[Dict], episode_indices: Optional[Iterable] = None):
        if episode_indices is None:
            episode_indices = range(self.total_episodes)
        if len(metadatas) != len(list(episode_indices)):
            raise ValueError("The number of metadatas doesn't match the number of episodes in the dataset.")
    
        with h5py.File(self._file_path, "a") as file:
            for metadata, episode_id in zip(metadatas, episode_indices):
                ep_group = file[f"episode_{episode_id}"]
                ep_group.attrs.update(metadata)

    def apply(
        self,
        function: Callable[[dict], Any],
        episode_indices: Optional[Iterable] = None,
    ) -> List[Any]:
        """Apply a function to a slice of the data.

        Args:
            function (Callable): function to apply to episodes
            episode_indices (Optional[Iterable]): epsiodes id to consider

        Returns:
            outs (list): list of outputs returned by the function applied to episodes
        """
        if episode_indices is None:
            episode_indices = range(self.total_episodes)
        out = []
        with h5py.File(self._file_path, "r") as file:
            for ep_idx in episode_indices:
                ep_group = file[f"episode_{ep_idx}"]
                assert isinstance(ep_group, h5py.Group)
                ep_dict = {
                    "id": ep_group.attrs.get("id"),
                    "total_timesteps": ep_group.attrs.get("total_steps"),
                    "seed": ep_group.attrs.get("seed"),
                    # TODO: self.metadata can be slow for decode space? Cache spaces? Cache metadata?
                    "observations": self._decode_space(
                        ep_group["observations"], self.metadata["observation_space"]
                    ),
                    "actions": self._decode_space(
                        ep_group["actions"], self.metadata["action_space"]
                    ),
                    "rewards": ep_group["rewards"][()],
                    "terminations": ep_group["terminations"][()],
                    "truncations": ep_group["truncations"][()],
                }
                out.append(function(ep_dict))

        return out

    def _decode_space(
        self,
        hdf_ref: Union[h5py.Group, h5py.Dataset],
        space: gym.spaces.Space,
    ) -> Union[Dict, Tuple, List, np.ndarray]:
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
            for key in hdf_ref:
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
                out.append(
                    {
                        "id": ep_group.attrs.get("id"),
                        "total_timesteps": ep_group.attrs.get("total_steps"),
                        "seed": ep_group.attrs.get("seed"),
                        "observations": self._decode_space(
                            ep_group["observations"], self.metadata["observation_space"]
                        ),
                        "actions": self._decode_space(
                            ep_group["actions"], self.metadata["action_space"]
                        ),
                        "rewards": ep_group["rewards"][()],
                        "terminations": ep_group["terminations"][()],
                        "truncations": ep_group["truncations"][()],
                    }
                )

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
                episode_group = get_h5py_subgroup(file, f"episode_{episode_id}")
                episode_group.attrs["id"] = episode_id
                if "seed" in eps_buff.keys():
                    assert not "seed" in episode_group.attrs.keys()
                    episode_group.attrs["seed"] = eps_buff.pop("seed")
                total_steps = len(eps_buff["rewards"])
                episode_group.attrs["total_steps"] = total_steps
                additional_steps += total_steps

                # TODO: make it append
                _add_episode_to_group(eps_buff, episode_group)

            total_steps = file.attrs["total_steps"] + additional_steps
            total_episodes = len(file.keys())

            file.attrs.modify("total_episodes", total_episodes)
            file.attrs.modify("total_steps", total_steps)

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
            total_episodes = file.attrs["total_steps"]
            assert type(total_episodes) == np.int64
            return total_episodes

def get_h5py_subgroup(group: h5py.Group, name: str):
    if name in group:
        subgroup = group.get(name)
    else:
        subgroup = group.create_group(name)

    return subgroup

def _add_episode_to_group(episode_buffer: Dict, episode_group: h5py.Group):
    for key, data in episode_buffer.items():
        if isinstance(data, dict):
            episode_group_to_clear = get_h5py_subgroup(episode_group, key)
            _add_episode_to_group(data, episode_group_to_clear)
        elif isinstance(data, int):
            import pdb; pdb.set_trace()
        elif all([isinstance(entry, tuple) for entry in data]):
            # we have a list of tuples, so we need to act appropriately
            dict_data = {
                f"_index_{str(i)}": [entry[i] for entry in data]
                for i, _ in enumerate(data[0])
            }
            episode_group_to_clear = get_h5py_subgroup(episode_group, key)
            _add_episode_to_group(dict_data, episode_group_to_clear)
        elif all([isinstance(entry, OrderedDict) for entry in data]):
            # we have a list of OrderedDicts, so we need to act appropriately
            dict_data = {
                key: [entry[key] for entry in data] for key, value in data[0].items()
            }
            episode_group_to_clear = get_h5py_subgroup(episode_group, key)
            _add_episode_to_group(dict_data, episode_group_to_clear)
        else:  # leaf data
            if isinstance(episode_group, h5py.Dataset):
                pass #TODO
            elif all(map(lambda elem: isinstance(elem, str), data)):
                dtype = h5py.string_dtype(encoding="utf-8")
                episode_group.create_dataset(key, data=data, dtype=dtype, chunks=True)
            else:
                assert np.all(np.logical_not(np.isnan(data)))
                episode_group.create_dataset(key, data=data, chunks=True)
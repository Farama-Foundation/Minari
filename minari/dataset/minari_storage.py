import os
from collections import OrderedDict
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import gymnasium as gym
import h5py
import numpy as np

from minari.serialization import deserialize_space


PathLike = Union[str, bytes, os.PathLike]


class MinariStorage:
    def __init__(self, data_path: PathLike):
        self._data_path = os.path.join(str(data_path), "main_data.hdf5")

    @property
    def metadata(self) -> Dict:
        with h5py.File(self.data_path, "r") as file:
            metadata = file.attrs
            if "observation_space" in metadata.keys():
                space_serialization = metadata["observation_space"]
                assert isinstance(space_serialization, Dict)
                metadata["observation_space"] = deserialize_space(space_serialization)            
            if "action_space" in metadata.keys():
                space_serialization = metadata["action_space"]
                assert isinstance(space_serialization, Dict)
                metadata["action_space"] = deserialize_space(space_serialization)
            
            return dict(metadata)
    
    def update_metadata(self, metadata: Dict):
        with h5py.File(self.data_path, "w") as file:
            file.attrs.update(metadata)

    def update_episode_metadata(self, metadatas: List[Dict], episode_indices: Optional[Iterable] = None):
        if episode_indices is None:
            episode_indices = range(self.total_episodes)
        if len(metadatas) != len(list(episode_indices)):
            raise ValueError("The number of metadatas doesn't match the number of episodes in the dataset.")
    
        with h5py.File(self.data_path, "w") as file:
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
        with h5py.File(self._data_path, "r") as file:
            for ep_idx in episode_indices:
                ep_group = file[f"episode_{ep_idx}"]
                assert isinstance(ep_group, h5py.Group)
                ep_dict = {
                    "id": ep_group.attrs.get("id"),
                    "total_timesteps": ep_group.attrs.get("total_steps"),
                    "seed": ep_group.attrs.get("seed"),
                    # TODO: self.metadata can be slow for decode space? Cache spaces? Cache metadata (bad for consistency)?
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
        with h5py.File(self._data_path, "r") as file:
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

    def update_episodes(self, episodes: List[dict]):
        additional_steps = 0
        with h5py.File(self.data_path, "a", track_order=True) as file:
            for eps_buff in episodes:
                # check episode terminated or truncated
                assert (
                    eps_buff["terminations"][-1] or eps_buff["truncations"][-1]
                ), "Each episode must be terminated or truncated before adding it to a Minari dataset"
                assert len(eps_buff["actions"]) + 1 == len(
                    eps_buff["observations"]
                ), f"Number of observations {len(eps_buff['observations'])} must have an additional \
                                                                                        element compared to the number of action steps {len(eps_buff['actions'])} \
                                                                                        The initial and final observation must be included"
                episode_id = eps_buff["id"]
                episode_group = get_h5py_subgroup(file, f"episode_{episode_id}")
                episode_group.attrs["id"] = episode_id
                if "seed" in eps_buff.keys():
                    assert not "seed" in episode_group.attrs.keys()
                    episode_group.attrs["seed"] = eps_buff["seed"]
                total_steps = len(eps_buff["actions"])
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
        return self._data_path

    @property
    def total_episodes(self) -> int:
        """Total episodes in the dataset."""
        with h5py.File(self.data_path, "r") as file:
            total_episodes = file.attrs["total_episodes"]
            assert isinstance(total_episodes, np.ndarray)
            total_episodes = total_episodes.item()
            assert isinstance(total_episodes, int)
            return total_episodes

def get_h5py_subgroup(group: h5py.Group, name: str) -> h5py.Group:
    if name in group:
        subgroup = group.get(name)
        # assert isinstance(subgroup, h5py.Group)
    else:
        subgroup = group.create_group(name)

    return subgroup

def _add_episode_to_group(episode_buffer: Dict, episode_group: h5py.Group):
    for key, data in episode_buffer.items():
        if isinstance(data, dict):
            episode_group_to_clear = get_h5py_subgroup(episode_group, key)
            _add_episode_to_group(data, episode_group_to_clear)
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
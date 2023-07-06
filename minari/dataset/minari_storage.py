import importlib.metadata
import json
import os
from collections import OrderedDict
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import gymnasium as gym
import h5py
import numpy as np
from gymnasium.envs.registration import EnvSpec
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import Version

from minari.data_collector import DataCollectorV0
from minari.serialization import deserialize_space


# Use importlib due to circular import when: "from minari import __version__"
__version__ = importlib.metadata.version("minari")

PathLike = Union[str, bytes, os.PathLike]


class MinariStorage:
    def __init__(self, data_path: PathLike):
        """Initialize properties of the Minari storage.

        Args:
            data_path (str): full path to the `main_data.hdf5` file of the dataset.
        """
        self._data_path = data_path
        self._extra_data_id = 0
        with h5py.File(self._data_path, "r") as f:
            self._env_spec = EnvSpec.from_json(f.attrs["env_spec"])

            total_episodes = f.attrs["total_episodes"].item()
            assert isinstance(total_episodes, int)
            self._total_episodes: int = total_episodes

            total_steps = f.attrs["total_steps"].item()
            assert isinstance(total_steps, int)
            self._total_steps: int = total_steps

            dataset_id = f.attrs["dataset_id"]
            assert isinstance(dataset_id, str)
            self._dataset_id = dataset_id

            minari_version = f.attrs["minari_version"]
            assert isinstance(minari_version, str)

            # Check that the dataset is compatible with the current version of Minari
            try:
                assert Version(__version__) in SpecifierSet(
                    minari_version
                ), f"The installed Minari version {__version__} is not contained in the dataset version specifier {minari_version}."
                self._minari_version = minari_version
            except InvalidSpecifier:
                print(f"{minari_version} is not a version specifier.")

            self._combined_datasets = f.attrs.get("combined_datasets", default=[])

            # We will default to using the reconstructed observation and action spaces from the dataset
            # and fall back to the env spec env if the action and observation spaces are not both present
            # in the dataset.
            if "action_space" in f.attrs and "observation_space" in f.attrs:
                self._observation_space = deserialize_space(
                    f.attrs["observation_space"]
                )
                self._action_space = deserialize_space(f.attrs["action_space"])
            else:
                # Checking if the base library of the environment is present in the environment
                entry_point = json.loads(f.attrs["env_spec"])["entry_point"]
                lib_full_path = entry_point.split(":")[0]
                base_lib = lib_full_path.split(".")[0]
                env_name = self._env_spec.id

                try:
                    env = gym.make(self._env_spec)
                    self._observation_space = env.observation_space
                    self._action_space = env.action_space
                    env.close()
                except ModuleNotFoundError as e:
                    raise ModuleNotFoundError(
                        f"Install {base_lib} for loading {env_name} data"
                    ) from e

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
                    "observations": self._decode_space(
                        ep_group["observations"], self.observation_space
                    ),
                    "actions": self._decode_space(
                        ep_group["actions"], self.action_space
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
                            ep_group["observations"], self.observation_space
                        ),
                        "actions": self._decode_space(
                            ep_group["actions"], self.action_space
                        ),
                        "rewards": ep_group["rewards"][()],
                        "terminations": ep_group["terminations"][()],
                        "truncations": ep_group["truncations"][()],
                    }
                )

        return out

    def update_from_collector_env(
        self,
        collector_env: DataCollectorV0,
        new_data_file_path: str,
        additional_data_id: int,
    ):

        collector_env.save_to_disk(path=new_data_file_path)

        with h5py.File(new_data_file_path, "r", track_order=True) as new_data_file:
            new_data_total_episodes = new_data_file.attrs["total_episodes"]
            new_data_total_steps = new_data_file.attrs["total_steps"]

        with h5py.File(self.data_path, "a", track_order=True) as file:
            last_episode_id = file.attrs["total_episodes"]
            for id in range(new_data_total_episodes):
                file[f"episode_{last_episode_id + id}"] = h5py.ExternalLink(
                    f"additional_data_{additional_data_id}.hdf5", f"/episode_{id}"
                )
                file[f"episode_{last_episode_id + id}"].attrs.modify(
                    "id", last_episode_id + id
                )

            # Update metadata of minari dataset
            file.attrs.modify(
                "total_episodes", last_episode_id + new_data_total_episodes
            )
            file.attrs.modify(
                "total_steps", file.attrs["total_steps"] + new_data_total_steps
            )
            self._total_episodes = int(file.attrs["total_episodes"].item())

    def update_from_buffer(self, buffer: List[dict], data_path: str):
        additional_steps = 0
        with h5py.File(data_path, "a", track_order=True) as file:
            last_episode_id = file.attrs["total_episodes"]
            for i, eps_buff in enumerate(buffer):
                episode_id = last_episode_id + i
                # check episode terminated or truncated
                assert (
                    eps_buff["terminations"][-1] or eps_buff["truncations"][-1]
                ), "Each episode must be terminated or truncated before adding it to a Minari dataset"
                assert len(eps_buff["actions"]) + 1 == len(
                    eps_buff["observations"]
                ), f"Number of observations {len(eps_buff['observations'])} must have an additional \
                                                                                        element compared to the number of action steps {len(eps_buff['actions'])} \
                                                                                        The initial and final observation must be included"
                seed = eps_buff.pop("seed", None)
                episode_group = clear_episode_buffer(
                    eps_buff, file.create_group(f"episode_{episode_id}")
                )

                episode_group.attrs["id"] = episode_id
                total_steps = len(eps_buff["actions"])
                episode_group.attrs["total_steps"] = total_steps
                additional_steps += total_steps

                if seed is None:
                    episode_group.attrs["seed"] = str(None)
                else:
                    assert isinstance(seed, int)
                    episode_group.attrs["seed"] = seed

                # TODO: save EpisodeMetadataCallback callback in MinariDataset and update new episode group metadata

            file.attrs.modify("total_episodes", last_episode_id + len(buffer))
            file.attrs.modify(
                "total_steps", file.attrs["total_steps"] + additional_steps
            )

            self._total_episodes = int(file.attrs["total_episodes"].item())

    @property
    def observation_space(self):
        """Original observation space of the environment before flatteining (if this is the case)."""
        return self._observation_space

    @property
    def action_space(self):
        """Original action space of the environment before flatteining (if this is the case)."""
        return self._action_space

    @property
    def data_path(self):
        """Full path to the `main_data.hdf5` file of the dataset."""
        return self._data_path

    @property
    def total_steps(self):
        """Total steps recorded in the Minari dataset along all episodes."""
        return self._total_steps

    @property
    def total_episodes(self):
        """Total episodes recorded in the Minari dataset."""
        return self._total_episodes

    @property
    def env_spec(self):
        """Envspec of the environment that has generated the dataset."""
        return self._env_spec

    @property
    def combined_datasets(self) -> List[str]:
        """If this Minari dataset is a combination of other subdatasets, return a list with the subdataset names."""
        if self._combined_datasets is None:
            return []
        else:
            return self._combined_datasets

    @property
    def id(self) -> str:
        """Name of the Minari dataset."""
        return self._dataset_id

    @property
    def minari_version(self) -> str:
        """Version of Minari the dataset is compatible with."""
        return self._minari_version


def clear_episode_buffer(episode_buffer: Dict, episode_group: h5py.Group) -> h5py.Group:
    """Save an episode dictionary buffer into an HDF5 episode group recursively.

    Args:
        episode_buffer (dict): episode buffer
        episode_group (h5py.Group): HDF5 group to store the episode datasets

    Returns:
        episode group: filled HDF5 episode group
    """
    for key, data in episode_buffer.items():
        if isinstance(data, dict):
            if key in episode_group:
                episode_group_to_clear = episode_group[key]
            else:
                episode_group_to_clear = episode_group.create_group(key)
            clear_episode_buffer(data, episode_group_to_clear)
        elif all([isinstance(entry, tuple) for entry in data]):
            # we have a list of tuples, so we need to act appropriately
            dict_data = {
                f"_index_{str(i)}": [entry[i] for entry in data]
                for i, _ in enumerate(data[0])
            }
            if key in episode_group:
                episode_group_to_clear = episode_group[key]
            else:
                episode_group_to_clear = episode_group.create_group(key)

            clear_episode_buffer(dict_data, episode_group_to_clear)
        elif all([isinstance(entry, OrderedDict) for entry in data]):

            # we have a list of OrderedDicts, so we need to act appropriately
            dict_data = {
                key: [entry[key] for entry in data] for key, value in data[0].items()
            }

            if key in episode_group:
                episode_group_to_clear = episode_group[key]
            else:
                episode_group_to_clear = episode_group.create_group(key)
            clear_episode_buffer(dict_data, episode_group_to_clear)
        elif all(map(lambda elem: isinstance(elem, str), data)):
            dtype = h5py.string_dtype(encoding="utf-8")
            episode_group.create_dataset(key, data=data, dtype=dtype, chunks=True)
        else:
            assert np.all(np.logical_not(np.isnan(data)))
            # add seed to attributes
            episode_group.create_dataset(key, data=data, chunks=True)

    return episode_group

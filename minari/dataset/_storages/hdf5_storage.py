from __future__ import annotations

import pathlib
from collections import OrderedDict
from itertools import zip_longest
from typing import Dict, Iterable, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np

from minari.data_collector import EpisodeBuffer
from minari.dataset.minari_storage import MinariStorage


try:
    import h5py
except ImportError:
    raise ImportError(
        'h5py is not installed. Please install it using `pip install "minari[hdf5]"`'
    )

_MAIN_FILE_NAME = "main_data.hdf5"


class HDF5Storage(MinariStorage):
    FORMAT = "hdf5"

    def __init__(
        self,
        data_path: pathlib.Path,
        observation_space: gym.Space,
        action_space: gym.Space,
    ):
        super().__init__(data_path, observation_space, action_space)
        file_path = self.data_path.joinpath(_MAIN_FILE_NAME)
        if not file_path.exists():
            raise ValueError(f"No data found in data path {self.data_path}")
        self._file_path = file_path

    @classmethod
    def _create(
        cls,
        data_path: pathlib.Path,
        observation_space: gym.Space,
        action_space: gym.Space,
    ) -> MinariStorage:
        data_path.joinpath(_MAIN_FILE_NAME).touch(exist_ok=False)
        obj = cls(data_path, observation_space, action_space)
        return obj

    def update_episode_metadata(
        self, metadatas: Iterable[Dict], episode_indices: Optional[Iterable] = None
    ):
        if episode_indices is None:
            episode_indices = range(self.total_episodes)

        sentinel = object()
        with h5py.File(self._file_path, "a") as file:
            for metadata, episode_id in zip_longest(
                metadatas, episode_indices, fillvalue=sentinel
            ):
                if sentinel in (metadata, episode_id):
                    raise ValueError(
                        "Metadatas and episode_indices have different lengths"
                    )

                assert isinstance(metadata, dict)
                ep_group = file[f"episode_{episode_id}"]
                ep_group.attrs.update(metadata)

    def get_episode_metadata(self, episode_indices: Iterable[int]) -> Iterable[Dict]:
        with h5py.File(self._file_path, "r") as file:
            for ep_idx in episode_indices:
                ep_group = file[f"episode_{ep_idx}"]
                assert isinstance(ep_group, h5py.Group)
                metadata: dict = dict(ep_group.attrs)
                metadata = unflatten_dict(metadata)
                if metadata.get("seed") is not None:
                    metadata["seed"] = int(metadata["seed"])

                yield metadata

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

    def get_episodes(self, episode_indices: Iterable[int]) -> Iterable[dict]:
        with h5py.File(self._file_path, "r") as file:
            for ep_idx in episode_indices:
                ep_group = file[f"episode_{ep_idx}"]
                assert isinstance(ep_group, h5py.Group)
                infos = None
                if "infos" in ep_group:
                    info_group = ep_group["infos"]
                    if isinstance(info_group, h5py.Group):  # for backward compatibility
                         infos = _decode_info(info_group)
                    else:
                        infos = read_dict_dataset_from_group(ep_group, "infos")

                ep_dict = {
                    "id": ep_idx,
                    "observations": self._decode_space(
                        ep_group["observations"], self.observation_space
                    ),
                    "actions": self._decode_space(
                        ep_group["actions"], self.action_space
                    ),
                    "infos": infos,
                }
                for key in {"rewards", "terminations", "truncations"}:
                    group_value = ep_group[key]
                    assert isinstance(group_value, h5py.Dataset)
                    ep_dict[key] = group_value[:]

                yield ep_dict

    def update_episodes(self, episodes: Iterable[EpisodeBuffer]):
        additional_steps = 0
        with h5py.File(self._file_path, "a", track_order=True) as file:
            for eps_buff in episodes:
                total_episodes = len(file.keys())
                episode_id = eps_buff.id if eps_buff.id is not None else total_episodes
                assert (
                    episode_id <= total_episodes
                ), "Invalid episode id; ids must be sequential."
                episode_group = _get_from_h5py(file, f"episode_{episode_id}")
                episode_group.attrs["id"] = episode_id
                if eps_buff.seed is not None:
                    assert "seed" not in episode_group.attrs.keys()
                    episode_group.attrs["seed"] = eps_buff.seed
                if eps_buff.options is not None:
                    assert "options" not in episode_group.attrs.keys()
                    flatten_option = flatten_dict(eps_buff.options, "options")
                    episode_group.attrs.update(flatten_option)

                episode_steps = len(eps_buff.rewards)
                episode_group.attrs["total_steps"] = episode_steps
                additional_steps += episode_steps

                dict_buffer = {
                    "observations": eps_buff.observations,
                    "actions": eps_buff.actions,
                    "rewards": eps_buff.rewards,
                    "terminations": eps_buff.terminations,
                    "truncations": eps_buff.truncations,
                    "infos": eps_buff.infos,
                }
                _add_episode_to_group(dict_buffer, episode_group)

            total_episodes = len(file.keys())

        total_steps = self.total_steps + additional_steps
        self.update_metadata(
            {"total_steps": total_steps, "total_episodes": total_episodes}
        )


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
        elif isinstance(data, tuple):
            dict_data = {f"_index_{i}": subdata for i, subdata in enumerate(data)}
            episode_group_to_clear = _get_from_h5py(episode_group, key)
            _add_episode_to_group(dict_data, episode_group_to_clear)
        elif isinstance(data, List) and all(
            isinstance(entry, OrderedDict) for entry in data
        ):  # list of OrderedDict
            dict_data = {key: [entry[key] for entry in data] for key in data[0].keys()}
            episode_group_to_clear = _get_from_h5py(episode_group, key)
            _add_episode_to_group(dict_data, episode_group_to_clear)

        # leaf data
        elif key in episode_group:
            dataset = episode_group[key]
            assert isinstance(dataset, h5py.Dataset)
            dataset.resize((dataset.shape[0] + len(data), *dataset.shape[1:]))
            dataset[-len(data) :] = data
        elif not isinstance(data, Iterable):
            if data is not None:
                episode_group.create_dataset(key, data=data)

        elif key == "infos":
            create_dict_dataset_in_group(episode_group, "infos", data)
        else:
            dtype = None
            if all(map(lambda elem: isinstance(elem, str), data)):
                dtype = h5py.string_dtype(encoding="utf-8")
            dshape = ()
            if hasattr(data[0], "shape"):
                dshape = data[0].shape

            episode_group.create_dataset(
                key, data=data, dtype=dtype, chunks=True, maxshape=(None, *dshape)
            )


def _decode_info(info_group: h5py.Group) -> Dict:
    result = {}
    for key, value in info_group.items():
        if isinstance(value, h5py.Group):
            result[key] = _decode_info(value)
        elif isinstance(value, h5py.Dataset):
            result[key] = value[()]
        else:
            raise ValueError(
                "Infos are in an unsupported format; see Minari documentation for supported formats."
            )
    return result


def flatten_dict(d: Dict, parent_key: str) -> Dict:
    flatten_d = {}
    for k, v in d.items():
        new_key = f"{parent_key}/{k}"
        if isinstance(v, dict):
            flatten_d.update(flatten_dict(v, new_key))
        else:
            flatten_d[new_key] = v
    return flatten_d


def unflatten_dict(d: Dict) -> Dict:
    result = {}
    for k, v in d.items():
        keys = k.split("/")
        current = result
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = v
    return result


def create_dict_dataset_in_group(group, dataset_name, dict_list):
    # Collect all unique keys
    all_keys = set()
    for d in dict_list:
        all_keys.update(d.keys())

    # Determine the structure
    dtype = []
    for key in all_keys:
        # Find the first non-None value for this key
        sample_value = next((d[key] for d in dict_list if key in d), None)

        if sample_value is None:
            # If we didn't find any non-None values, default to float (to allow np.nan)
            dtype.append((key, float))
        elif isinstance(sample_value, str):
            dtype.append((key, h5py.special_dtype(vlen=str)))
        elif isinstance(sample_value, list):
            if all(isinstance(item, str) for item in sample_value):
                dtype.append((key, h5py.special_dtype(vlen=str), (len(sample_value),)))
            else:
                # For simplicity, assume all non-string lists are float
                dtype.append((key, float, (len(sample_value),)))
        elif isinstance(sample_value, (int, float)):
            dtype.append((key, float))
        else:
            dtype.append((key, type(sample_value)))

    # Create a numpy structured array
    data = np.empty(len(dict_list), dtype=dtype)
    for i, d in enumerate(dict_list):
        for key in all_keys:
            if key in d:
                if isinstance(d[key], list):
                    if all(isinstance(item, str) for item in d[key]):
                        data[key][i] = d[key]
                    else:
                        data[key][i] = d[key] + [np.nan] * (len(data[key][i]) - len(d[key]))
                elif isinstance(d[key], (int, float)):
                    data[key][i] = float(d[key])
                else:
                    data[key][i] = d[key]
            else:
                if data[key].dtype.kind == 'f':
                    data[key][i] = np.nan
                elif data[key].dtype.kind == 'O':
                    data[key][i] = ''
                # For other types, it will use the default value (0 for int, False for bool, etc.)

    # Create the dataset within the group
    dataset = group.create_dataset(dataset_name, data=data, chunks=True, compression='gzip')
    # Add metadata to the dataset
    dataset.attrs['num_records'] = len(dict_list)
    dataset.attrs['keys'] = list(all_keys)

    return dataset


def read_dict_dataset_from_group(group, dataset_name):
    dataset = group[dataset_name]

    data = dataset[:]
    dict_list = []
    for row in data:
        d = {}
        for field in row.dtype.names:
            value = row[field]
            if isinstance(value, np.ndarray):
                if value.dtype.kind == 'S' or value.dtype.kind == 'U':
                    d[field] = [v.decode('utf-8') if isinstance(v, bytes) else v for v in value if v]
                else:
                    d[field] = [v for v in value if not np.isnan(v)]
            elif isinstance(value, (bytes, np.bytes_)):
                d[field] = value.decode('utf-8')
            elif isinstance(value, float) and np.isnan(value):
                continue  # Skip NaN values
            else:
                d[field] = value
        dict_list.append(d)
    return dict_list

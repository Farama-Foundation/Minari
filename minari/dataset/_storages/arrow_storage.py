from __future__ import annotations

import json
import pathlib
from itertools import zip_longest
from typing import Any, Dict, Iterable, Optional, Sequence

import gymnasium as gym
import numpy as np


try:
    import pyarrow as pa
    import pyarrow.dataset as ds
except ImportError:
    raise ImportError(
        'pyarrow is not installed. Please install it using `pip install "minari[arrow]"`'
    )

from minari.data_collector.episode_buffer import EpisodeBuffer
from minari.dataset.minari_storage import MinariStorage


_FIXEDLIST_SPACES = (gym.spaces.Box, gym.spaces.MultiDiscrete, gym.spaces.MultiBinary)


class ArrowStorage(MinariStorage):
    FORMAT = "arrow"

    def __init__(
        self,
        data_path: pathlib.Path,
        observation_space: gym.Space,
        action_space: gym.Space,
    ):
        super().__init__(data_path, observation_space, action_space)

    @classmethod
    def _create(
        cls,
        data_path: pathlib.Path,
        observation_space: gym.Space,
        action_space: gym.Space,
    ) -> MinariStorage:
        return cls(data_path, observation_space, action_space)

    def update_episode_metadata(
        self, metadatas: Iterable[Dict], episode_indices: Optional[Iterable] = None
    ):
        if episode_indices is None:
            episode_indices = range(self.total_episodes)

        sentinel = object()
        for new_metadata, episode_id in zip_longest(
            metadatas, episode_indices, fillvalue=sentinel
        ):
            if sentinel in (new_metadata, episode_id):
                raise ValueError("Metadatas and episode_indices have different lengths")

            assert isinstance(new_metadata, dict)
            metadata_path = self.data_path.joinpath(str(episode_id), "metadata.json")

            metadata = {}
            if metadata_path.exists():
                with open(metadata_path) as file:
                    metadata = json.load(file)
            metadata.update(new_metadata)
            with open(metadata_path, "w") as file:
                json.dump(metadata, file, cls=NumpyEncoder)

        self.update_metadata({"dataset_size": self.get_size()})

    def get_episode_metadata(self, episode_indices: Iterable[int]) -> Iterable[Dict]:
        for episode_id in episode_indices:
            metadata_path = self.data_path.joinpath(str(episode_id), "metadata.json")
            with open(metadata_path) as file:
                yield json.load(file)

    def get_episodes(self, episode_indices: Iterable[int]) -> Iterable[dict]:
        dataset = pa.dataset.dataset(
            [
                pa.dataset.dataset(
                    f"{self.data_path}/{ep_id}",
                    format=self.FORMAT,
                    ignore_prefixes=["_", ".", "metadata.json"],
                )
                for ep_id in episode_indices
            ]
        )

        def _to_dict(id, episode):
            return {
                "id": id,
                "observations": _decode_space(
                    self.observation_space, episode["observations"]
                ),
                "actions": _decode_space(self.action_space, episode["actions"][:-1]),
                "rewards": np.asarray(episode["rewards"])[:-1],
                "terminations": np.asarray(episode["terminations"])[:-1],
                "truncations": np.asarray(episode["truncations"])[:-1],
                "infos": (
                    _decode_info(episode["infos"])
                    if "infos" in episode.column_names
                    else {}
                ),
            }

        return map(_to_dict, episode_indices, dataset.to_batches())

    def update_episodes(self, episodes: Iterable[EpisodeBuffer]):
        total_steps = self.total_steps
        total_episodes = self.total_episodes
        for episode_data in episodes:
            episode_id = (
                episode_data.id if episode_data.id is not None else total_episodes
            )
            total_episodes = max(total_episodes, episode_id + 1)
            observations = _encode_space(
                self.observation_space, episode_data.observations
            )
            rewards = np.asarray(episode_data.rewards).reshape(-1)
            terminations = np.asarray(episode_data.terminations).reshape(-1)
            truncations = np.asarray(episode_data.truncations).reshape(-1)
            pad = len(observations) - len(rewards)
            actions = _encode_space(self._action_space, episode_data.actions, pad=pad)

            episode_batch = {
                "episode_id": np.full(len(observations), episode_id, dtype=np.int32),
                "observations": observations,
                "actions": actions,
                "rewards": np.pad(rewards, ((0, pad))),
                "terminations": np.pad(terminations, ((0, pad))),
                "truncations": np.pad(truncations, ((0, pad))),
            }
            if episode_data.infos:
                episode_batch["infos"] = _encode_info(episode_data.infos)
            episode_batch = pa.RecordBatch.from_pydict(episode_batch)

            total_steps += len(rewards)
            ds.write_dataset(
                episode_batch,
                self.data_path,
                format=self.FORMAT,
                partitioning=["episode_id"],
                existing_data_behavior="overwrite_or_ignore",
            )

            episode_metadata: dict = {"id": episode_id, "total_steps": len(rewards)}
            if episode_data.seed is not None:
                episode_metadata["seed"] = episode_data.seed
            if episode_data.options is not None:
                episode_metadata["options"] = episode_data.options
            self.update_episode_metadata([episode_metadata], [episode_id])

        self.update_metadata(
            {
                "total_steps": total_steps,
                "total_episodes": total_episodes,
                "dataset_size": self.get_size(),
            }
        )


def _encode_space(space: gym.Space, values: Any, pad: int = 0):
    if isinstance(space, gym.spaces.Dict):
        assert isinstance(values, dict), values
        arrays, names = [], []
        for key, value in values.items():
            names.append(key)
            arrays.append(_encode_space(space[key], value, pad=pad))
        return pa.StructArray.from_arrays(arrays, names=names)
    if isinstance(space, gym.spaces.Tuple):
        assert isinstance(values, tuple), values
        arrays, names = [], []
        for i, value in enumerate(values):
            names.append(str(i))
            arrays.append(_encode_space(space[i], value, pad=pad))
        return pa.StructArray.from_arrays(arrays, names=names)
    elif isinstance(space, _FIXEDLIST_SPACES):
        values = np.asarray(values)
        assert values.shape[1:] == space.shape
        values = values.reshape(values.shape[0], -1)
        values = np.pad(values, ((0, pad), (0, 0)))
        dtype = pa.list_(pa.from_numpy_dtype(space.dtype), list_size=values.shape[1])
        return pa.FixedSizeListArray.from_arrays(values.reshape(-1), type=dtype)
    elif isinstance(space, gym.spaces.Discrete):
        values = np.asarray(values).reshape(-1, 1)
        values = np.pad(values, ((0, pad), (0, 0)))
        return pa.array(values.squeeze(-1), type=pa.from_numpy_dtype(space.dtype))
    else:
        if not isinstance(values, list):
            values = list(values)
        return pa.array(values + [None] * pad)


def _decode_space(space, values: pa.Array):
    if isinstance(space, gym.spaces.Dict):
        return {
            name: _decode_space(subspace, values.field(name))
            for name, subspace in space.spaces.items()
        }
    elif isinstance(space, gym.spaces.Tuple):
        return tuple(
            [
                _decode_space(subspace, values.field(str(i)))
                for i, subspace in enumerate(space.spaces)
            ]
        )
    elif isinstance(space, _FIXEDLIST_SPACES):
        data = np.stack(values.to_numpy(zero_copy_only=False))
        return data.reshape(-1, *space.shape)
    elif isinstance(space, gym.spaces.Discrete):
        return values.to_numpy()
    else:
        return values.to_pylist()


def _encode_info(info: dict):
    arrays, fields = [], []

    for key, values in info.items():
        if isinstance(values, dict):
            array = _encode_info(values)
            arrays.append(array)
            fields.append(pa.field(key, array.type))

        elif isinstance(values, tuple):
            array = _encode_info({str(i): v for i, v in enumerate(values)})
            arrays.append(array)
            fields.append(pa.field(key, array.type))

        elif isinstance(values, np.ndarray) or (
            isinstance(values, Sequence) and isinstance(values[0], np.ndarray)
        ):
            if isinstance(values, Sequence):
                values = np.stack(values)

            data_enc = ",".join(str(n) for n in values.shape[1:])
            values = values.reshape(len(values), -1)
            dtype = pa.from_numpy_dtype(values.dtype)
            struct = pa.list_(dtype, list_size=values.shape[1])
            arrays.append(
                pa.FixedSizeListArray.from_arrays(values.reshape(-1), type=struct)
            )
            fields.append(pa.field(key, struct, metadata={"shape": data_enc}))

        else:
            array = pa.array(list(values))
            arrays.append(array)
            fields.append(pa.field(key, array.type))

    return pa.StructArray.from_arrays(arrays, fields=fields)


def _decode_info(values: pa.Array):
    nested_dict = {}
    for i, field in enumerate(values.type):
        if isinstance(field, pa.StructArray):
            nested_dict[field.name] = _decode_info(values.field(i))
        else:
            value = np.stack(values.field(i).to_numpy(zero_copy_only=False))
            if field.metadata is not None and b"shape" in field.metadata:
                data_shape = field.metadata[b"shape"].split(b",")
                data_shape = tuple(int(d) for d in data_shape)
                value = value.reshape(len(value), *data_shape)
            nested_dict[field.name] = value
    return nested_dict


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

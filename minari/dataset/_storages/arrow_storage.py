from __future__ import annotations

import json
import pathlib
from itertools import zip_longest
from typing import Any, Dict, Iterable, Optional, Sequence, List

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
            if "infos" in episode.column_names:
                try:
                    infos = decode_info_list(episode["infos"])
                except Exception as e:  # for backwards compatibility
                    try:
                        infos = _decode_info(episode["infos"])
                    except Exception as e:
                        raise ValueError(f"Failed to decode infos: {e}")
            else:
                infos = {}

            return {
                "id": id,
                "observations": _decode_space(
                    self.observation_space, episode["observations"]
                ),
                "actions": _decode_space(self.action_space, episode["actions"][:-1]),
                "rewards": np.asarray(episode["rewards"])[:-1],
                "terminations": np.asarray(episode["terminations"])[:-1],
                "truncations": np.asarray(episode["truncations"])[:-1],
                "infos": infos,
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
                episode_batch["infos"] = encode_info_list(episode_data.infos)
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
            {"total_steps": total_steps, "total_episodes": total_episodes}
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
    elif isinstance(space, gym.spaces.Box):
        values = np.asarray(values)
        assert values.shape[1:] == space.shape
        values = values.reshape(values.shape[0], -1)
        values = np.pad(values, ((0, pad), (0, 0)))
        dtype = pa.list_(pa.from_numpy_dtype(space.dtype), list_size=values.shape[1])
        return pa.FixedSizeListArray.from_arrays(values.reshape(-1), type=dtype)
    elif isinstance(space, gym.spaces.Discrete):
        values = np.asarray(values).reshape(-1, 1)
        values = np.pad(values, ((0, pad), (0, 0)))
        return pa.array(values.squeeze(-1), type=pa.int32())
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
    elif isinstance(space, gym.spaces.Box):
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

            data_shape = values.shape[1:]
            values = values.reshape(len(values), -1)
            dtype = pa.from_numpy_dtype(values.dtype)
            struct = pa.list_(dtype, list_size=values.shape[1])
            arrays.append(
                pa.FixedSizeListArray.from_arrays(values.reshape(-1), type=struct)
            )
            fields.append(pa.field(key, struct, metadata={"shape": bytes(data_shape)}))

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
                data_shape = tuple(field.metadata[b"shape"])
                value = value.reshape(len(value), *data_shape)
            nested_dict[field.name] = value
    return nested_dict


def encode_info_list(info_list: List[Dict[str, Any]]):
    if not info_list:
        return pa.StructArray.from_arrays([], fields=[])

    # Collect all unique keys
    all_keys = set()
    for d in info_list:
        all_keys.update(d.keys())

    arrays, fields = [], []
    for key in all_keys:
        values = [d.get(key) for d in info_list]

        # Handle missing values
        if all(v is None for v in values):
            arrays.append(pa.array(values))
            fields.append(pa.field(key, pa.null()))
            continue

        sample_value = next(v for v in values if v is not None)

        if isinstance(sample_value, dict):
            nested_list = [{k: v.get(k) if v is not None else None for k in sample_value.keys()} for v in values]
            array = encode_info_list(nested_list)
            arrays.append(array)
            fields.append(pa.field(key, array.type))
        elif isinstance(sample_value, tuple):
            nested_list = [{str(i): v[i] if v is not None else None for i in range(len(sample_value))} for v in values]
            array = encode_info_list(nested_list)
            arrays.append(array)
            fields.append(pa.field(key, array.type))
        elif isinstance(sample_value, np.ndarray) or (
                isinstance(sample_value, Sequence) and isinstance(sample_value[0], np.ndarray)
        ):
            # Handle potential None values
            valid_values = [v for v in values if v is not None]
            if isinstance(sample_value, Sequence):
                valid_values = np.stack(valid_values)
            else:
                valid_values = np.array(valid_values)
            data_shape = valid_values.shape[1:]
            valid_values = valid_values.reshape(len(valid_values), -1)
            dtype = pa.from_numpy_dtype(valid_values.dtype)
            struct = pa.list_(dtype, list_size=valid_values.shape[1])
            array = pa.FixedSizeListArray.from_arrays(valid_values.reshape(-1), type=struct)
            # Handle None values
            mask = [v is not None for v in values]
            array = array.fill_null(pa.null())
            array = array.filter(mask)
            arrays.append(array)
            fields.append(pa.field(key, struct, metadata={"shape": bytes(data_shape)}))
        else:
            array = pa.array(values)
            arrays.append(array)
            fields.append(pa.field(key, array.type))

    return pa.StructArray.from_arrays(arrays, fields=fields)


def decode_info_list(values: pa.Array) -> List[Dict[str, Any]]:
    result = []
    for i in range(len(values)):
        nested_dict = {}
        for j, field in enumerate(values.type):
            if pa.types.is_struct(field.type):
                nested_value = decode_info_list(values.field(j)[i])
                if len(nested_value) == 1:
                    nested_dict[field.name] = nested_value[0]
                else:
                    nested_dict[field.name] = nested_value
            else:
                value = values.field(j)[i]
                if value is None:
                    continue
                if isinstance(value, pa.FixedSizeListArray):
                    value = np.array(value.to_numpy(zero_copy_only=False))
                    if field.metadata is not None and b"shape" in field.metadata:
                        data_shape = tuple(field.metadata[b"shape"])
                        value = value.reshape(data_shape)
                else:
                    value = value.as_py()
                nested_dict[field.name] = value
        result.append(nested_dict)
    return result


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

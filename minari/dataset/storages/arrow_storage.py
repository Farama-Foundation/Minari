from __future__ import annotations

import os
import pathlib
from collections import OrderedDict
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import gymnasium as gym
import h5py
from gymnasium.envs.registration import EnvSpec
from minari.dataset.minari_storage import MinariStorage, PathLike
from minari.serialization import deserialize_space, serialize_space
import pyarrow as pa
import pyarrow.parquet as pq


class _ArrowStorage(MinariStorage):

    def get_episodes(self, episode_indices: Iterable[int]) -> List[dict]:
        ...

    def update_episodes(self, episodes: Iterable[dict]):
        for eps_buff in episodes:
            episode_id = eps_buff.pop("id", self.total_episodes)
            episode_path = os.path.join(self.data_path, episode_id)
            if not os.path.exists(episode_path):
                os.makedirs(episode_path)

            table = pa.table(eps_buff.values(), names=eps_buff.keys())

            part_id = len(os.listdir(episode_path))
            pq.write_table(table, f"part-{part_id}.parquet")
from __future__ import annotations

import pathlib

import gymnasium as gym

from minari.dataset._storages.arrow_storage import ArrowStorage


_FIXEDLIST_SPACES = (gym.spaces.Box, gym.spaces.MultiDiscrete, gym.spaces.MultiBinary)


class ParquetStorage(ArrowStorage):
    FORMAT = "parquet"

    def __init__(
        self,
        data_path: pathlib.Path,
        observation_space: gym.Space,
        action_space: gym.Space,
    ):
        super().__init__(data_path, observation_space, action_space)

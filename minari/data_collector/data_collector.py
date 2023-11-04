from __future__ import annotations

import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional, SupportsFloat, Type, Union

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType

from minari.data_collector.callbacks import (
    STEP_DATA_KEYS,
    EpisodeMetadataCallback,
    StepData,
    StepDataCallback,
)
from minari.dataset.minari_dataset import MinariDataset
from minari.dataset.minari_storage import MinariStorage


EpisodeBuffer = Dict[str, Any]  # TODO: narrow this down


class DataCollectorV0(gym.Wrapper):
    r"""Gymnasium environment wrapper that collects step data.

    This wrapper is meant to work as a temporary buffer of the environment data before creating a Minari dataset. The creation of the buffers
    that will be convert to a Minari dataset is agnostic to the user:

    .. code::

        import minari
        import gymnasium as gym

        env = minari.DataCollectorV0(gym.make('EnvID'))

        env.reset()

        for _ in range(num_steps):
            action = env.action_space.sample()
            obs, rew, terminated, truncated, info = env.step()

            if terminated or truncated:
                env.reset()

        dataset = minari.create_dataset_from_collector_env(dataset_id="env_name-dataset_name-v(version)", collector_env=env, **kwargs)

    Some of the characteristics of this wrapper:

        * The step data is stored per episode in dictionaries. This dictionaries are then stored in-memory in a global list buffer. The
          episode dictionaries contain items with list buffers as values for the main episode step datasets `observations`, `actions`,
          `terminations`, and `truncations`, the `infos` key can be a list or another nested dictionary with extra datasets. Separate data
          keys can be added by passing a custom `StepDataCallback` to the wrapper. When creating the HDF5 file the list values in the episode
          dictionary will be stored as datasets and the nested dictionaries will generate a new HDF5 group.

        * A new episode dictionary buffer is created if the env.step(action) call returns `truncated` or `terminated`, or if the environment calls
          env.reset(). If calling reset and the previous episode was not `truncated` or `terminated`, this will automatically be `truncated`.

        * To perform caching the user can set the `max_buffer_steps` or `max_buffer_episodes` before saving the in-memory buffers to a temporary HDF5
          file in disk. If non of `max_buffer_steps` or `max_buffer_episodes` are set, the data will move from in-memory to a permanent location only
          when the Minari dataset is created. To move all the stored data to a permanent location use DataCollectorV0.save_to_disK(path_to_permanent_location).
    """

    def __init__(
        self,
        env: gym.Env,
        step_data_callback: Type[StepDataCallback] = StepDataCallback,
        episode_metadata_callback: Type[
            EpisodeMetadataCallback
        ] = EpisodeMetadataCallback,
        record_infos: bool = False,
        max_buffer_steps: Optional[int] = None,
        observation_space=None,
        action_space=None,
    ):
        """Initialize the data colletor attributes and create the temporary directory for caching.

        Args:
            env (gym.Env): Gymnasium environment
            step_data_callback (type[StepDataCallback], optional): Callback class to edit/update step databefore storing to buffer. Defaults to StepDataCallback.
            episode_metadata_callback (type[EpisodeMetadataCallback], optional): Callback class to add custom metadata to episode group in HDF5 file. Defaults to EpisodeMetadataCallback.
            record_infos (bool, optional): If True record the info return key of each step. Defaults to False.
            max_buffer_steps (Optional[int], optional): number of steps saved in-memory buffers before dumping to HDF5 file in disk. Defaults to None.

        Raises:
            ValueError: `max_buffer_steps` and `max_buffer_episodes` can't be passed at the same time
        """
        super().__init__(env)
        self._step_data_callback = step_data_callback()
        self._episode_metadata_callback = episode_metadata_callback()

        # get path to minari datasets directory
        self.datasets_path = os.environ.get("MINARI_DATASETS_PATH")
        if self.datasets_path is None:
            self.datasets_path = os.path.join(
                os.path.expanduser("~"), ".minari", "datasets"
            )
        # create local directory if it doesn't exist
        if not os.path.exists(self.datasets_path):
            os.makedirs(self.datasets_path)

        self._tmp_dir = tempfile.TemporaryDirectory(dir=self.datasets_path)
        self._storage = MinariStorage.new(
            self._tmp_dir.name,
            observation_space=observation_space,
            action_space=action_space,
            env_spec=self.env.spec,
        )

        if observation_space is None:
            observation_space = self.env.observation_space
        self.dataset_observation_space = observation_space

        if action_space is None:
            action_space = self.env.action_space
        self.dataset_action_space = action_space

        self._record_infos = record_infos
        self.max_buffer_steps = max_buffer_steps

        # Initialzie empty buffer
        self._buffer: List[EpisodeBuffer] = []

        self._step_id = -1
        self._episode_id = -1

    def _add_to_episode_buffer(
        self,
        episode_buffer: EpisodeBuffer,
        step_data: Union[StepData, Dict[str, StepData]],
    ) -> EpisodeBuffer:
        """Add step data dictionary to episode buffer.

        Args:
            episode_buffer (Dict): dictionary episode buffer
            step_data (Dict): dictionary with data for a single step

        Returns:
            Dict: new dictionary episode buffer with added values from step_data
        """
        for key, value in step_data.items():
            if (not self._record_infos and key == "infos") or (value is None):
                continue

            if key not in episode_buffer:
                if isinstance(value, dict):
                    episode_buffer[key] = self._add_to_episode_buffer({}, value)
                else:
                    episode_buffer[key] = [value]
            else:
                if isinstance(value, dict):
                    assert isinstance(
                        episode_buffer[key], dict
                    ), f"Element to be inserted is type 'dict', but buffer accepts type {type(episode_buffer[key])}"

                    episode_buffer[key] = self._add_to_episode_buffer(
                        episode_buffer[key], value
                    )
                else:
                    assert isinstance(
                        episode_buffer[key], list
                    ), f"Element to be inserted is type 'list', but buffer accepts type {type(episode_buffer[key])}"
                    episode_buffer[key].append(value)

        return episode_buffer

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Gymnasium step method."""
        obs, rew, terminated, truncated, info = self.env.step(action)

        step_data = self._step_data_callback(
            env=self.env,
            obs=obs,
            info=info,
            action=action,
            rew=rew,
            terminated=terminated,
            truncated=truncated,
        )
        assert STEP_DATA_KEYS.issubset(
            step_data.keys()
        ), "One or more required keys is missing from 'step-data'."
        assert self.dataset_observation_space.contains(
            step_data["observations"]
        ), "Observations are not in observation space."
        assert self.dataset_action_space.contains(
            step_data["actions"]
        ), "Actions are not in action space."

        self._step_id += 1
        self._buffer[-1] = self._add_to_episode_buffer(self._buffer[-1], step_data)

        if (
            self.max_buffer_steps is not None
            and self._step_id != 0
            and self._step_id % self.max_buffer_steps == 0
        ):
            self._storage.update_episodes(self._buffer)
            self._buffer = [{"id": self._episode_id}]
        if step_data["terminations"] or step_data["truncations"]:
            self._episode_id += 1
            eps_buff = {"id": self._episode_id}
            previous_data = {
                "observations": step_data["observations"],
                "infos": step_data["infos"],
            }
            eps_buff = self._add_to_episode_buffer(eps_buff, previous_data)
            self._buffer.append(eps_buff)

        return obs, rew, terminated, truncated, info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """Gymnasium environment reset."""
        obs, info = self.env.reset(seed=seed, options=options)
        step_data = self._step_data_callback(env=self.env, obs=obs, info=info)
        self._episode_id += 1

        assert STEP_DATA_KEYS.issubset(
            step_data.keys()
        ), "One or more required keys is missing from 'step-data'"

        self._validate_buffer()
        episode_buffer = {"seed": seed if seed else str(None), "id": self._episode_id}
        episode_buffer = self._add_to_episode_buffer(episode_buffer, step_data)
        self._buffer.append(episode_buffer)
        return obs, info

    def _validate_buffer(self):
        if len(self._buffer) > 0:
            if "actions" not in self._buffer[-1].keys():
                self._buffer.pop()
                self._episode_id -= 1
            elif not self._buffer[-1]["terminations"][-1]:
                self._buffer[-1]["truncations"][-1] = True

    def add_to_dataset(self, dataset: MinariDataset):
        """Add extra data to Minari dataset from collector environment buffers (DataCollectorV0).

        Args:
            dataset (MinariDataset): Dataset to add the data
        """
        self._validate_buffer()
        self._storage.update_episodes(self._buffer)
        self._buffer.clear()

        first_id = dataset.storage.total_episodes
        dataset.storage.update_from_storage(self._storage)
        if dataset.episode_indices is not None:
            new_ids = first_id + np.arange(self._storage.total_episodes)
            dataset.episode_indices = np.append(dataset.episode_indices, new_ids)

        self._episode_id = -1
        self._tmp_dir = tempfile.TemporaryDirectory(dir=self.datasets_path)
        self._storage = MinariStorage.new(
            self._tmp_dir.name,
            observation_space=self._storage.observation_space,
            action_space=self._storage.action_space,
            env_spec=self.env.spec,
        )

    def save_to_disk(
        self, path: str | os.PathLike, dataset_metadata: Dict[str, Any] = {}
    ):
        """Save all in-memory buffer data and move temporary files to a permanent location in disk.

        Args:
            path (str): path to store the dataset, e.g.: '/home/foo/datasets/data'
            dataset_metadata (Dict, optional): additional metadata to add to the dataset file. Defaults to {}.
        """
        self._validate_buffer()
        self._storage.update_episodes(self._buffer)
        self._buffer.clear()

        assert (
            "observation_space" not in dataset_metadata.keys()
        ), "'observation_space' is not allowed as an optional key."
        assert (
            "action_space" not in dataset_metadata.keys()
        ), "'action_space' is not allowed as an optional key."
        assert (
            "env_spec" not in dataset_metadata.keys()
        ), "'env_spec' is not allowed as an optional key."
        self._storage.update_metadata(dataset_metadata)

        episode_metadata = self._storage.apply(self._episode_metadata_callback)
        self._storage.update_episode_metadata(episode_metadata)

        files = os.listdir(self._storage.data_path)
        for file in files:
            shutil.move(
                os.path.join(self._storage.data_path, file),
                os.path.join(path, file),
            )

        self._episode_id = -1
        self._tmp_dir = tempfile.TemporaryDirectory(dir=self.datasets_path)
        self._storage = MinariStorage.new(
            self._tmp_dir.name,
            observation_space=self._storage.observation_space,
            action_space=self._storage.action_space,
            env_spec=self.env.spec,
        )

    def close(self):
        """Close the DataCollector.

        Clear buffer and close temporary directory.
        """
        super().close()

        self._buffer.clear()
        shutil.rmtree(self._tmp_dir.name)

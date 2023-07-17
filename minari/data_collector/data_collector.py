from __future__ import annotations

import os
import shutil
import tempfile
from collections import OrderedDict
from typing import Any, Dict, List, Optional, SupportsFloat, Type, Union

import gymnasium as gym
import h5py
import numpy as np
from gymnasium.core import ActType, ObsType

from minari.data_collector.callbacks import (
    STEP_DATA_KEYS,
    EpisodeMetadataCallback,
    StepData,
    StepDataCallback,
)
from minari.serialization import serialize_space


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
        max_buffer_episodes: Optional[int] = None,
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
            max_buffer_episodes (Optional[int], optional): number of episodes saved in-memory buffers before dumping to HDF5 file in disk. Defaults to None.

        Raises:
            ValueError: `max_buffer_steps` and `max_buffer_episodes` can't be passed at the same time
        """
        super().__init__(env)
        self._step_data_callback = step_data_callback()

        if observation_space is None:
            observation_space = self.env.observation_space
        self.dataset_observation_space = observation_space

        if action_space is None:
            action_space = self.env.action_space
        self.dataset_action_space = action_space

        self._episode_metadata_callback = episode_metadata_callback()
        self._record_infos = record_infos

        if max_buffer_steps is not None and max_buffer_episodes is not None:
            raise ValueError("Choose step or episode scheduler not both")

        self.max_buffer_episodes = max_buffer_episodes
        self.max_buffer_steps = max_buffer_steps

        # Initialzie empty buffer
        self._buffer: List[EpisodeBuffer] = [{}]

        self._current_seed: Union[int, str] = str(None)
        self._new_episode = False

        self._step_id = 0

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
        self._tmp_f = h5py.File(
            os.path.join(self._tmp_dir.name, "tmp_dataset.hdf5"), "a", track_order=True
        )  # track insertion order of groups ('episodes_i')

        assert self.env.spec is not None, "Env Spec is None"
        self._tmp_f.attrs["env_spec"] = self.env.spec.to_json()

        self._new_episode = False
        self._reset_called = False

        # Initialize first episode group in temporary hdf5 file
        self._episode_id = 0
        self._eps_group: h5py.Group = self._tmp_f.create_group("episode_0")
        self._eps_group.attrs["id"] = 0

        self._last_episode_group_term_or_trunc = False
        self._last_episode_n_steps = 0

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
                # if the step data comes from a reset call: skip actions, rewards,
                # terminations, and truncations their values are set to None in the StepDataCallback
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

        # add/edit data from step and convert to dictionary step data
        step_data = self._step_data_callback(
            env=self,
            obs=obs,
            info=info,
            action=action,
            rew=rew,
            terminated=terminated,
            truncated=truncated,
        )
        # Force step data dictionary to include keys corresponding to Gymnasium step returns:
        # actions, observations, rewards, terminations, truncations, and infos
        assert STEP_DATA_KEYS.issubset(
            step_data.keys()
        ), "One or more required keys is missing from 'step-data'."
        # Check that the saved observation and action belong to the dataset's observation/action spaces
        assert self.dataset_observation_space.contains(
            step_data["observations"]
        ), "Observations are not in observation space."
        assert self.dataset_action_space.contains(
            step_data["actions"]
        ), "Actions are not in action space."

        self._step_id += 1

        clear_buffers = False
        # check if buffer needs to be cleared to temp file due to maximum step scheduler
        if self.max_buffer_steps is not None:
            clear_buffers = (
                self._step_id % self.max_buffer_steps == 0 and self._step_id != 0
            )

        # Get initial observation/info from previous episode if reset has not been called after termination
        # or truncation. This may happen if the step_data_callback truncates or terminates the episode under
        # certain conditions.
        if self._new_episode and not self._reset_called:
            if isinstance(self._previous_eps_final_obs, dict):
                self._buffer[-1]["observations"] = self._add_to_episode_buffer(
                    {}, self._previous_eps_final_obs
                )
            else:
                self._buffer[-1]["observations"] = [self._previous_eps_final_obs]
            if self._record_infos:
                self._buffer[-1]["infos"] = self._add_to_episode_buffer(
                    {}, self._previous_eps_final_info
                )

            self._new_episode = False

        # add step data to last episode buffer
        self._buffer[-1] = self._add_to_episode_buffer(self._buffer[-1], step_data)

        if step_data["terminations"] or step_data["truncations"]:
            # Save last observation/info to use as initial observation/info in the next episode
            self._previous_eps_final_obs = step_data["observations"]
            if self._record_infos:
                self._previous_eps_final_info = step_data["infos"]
            self._reset_called = False
            self._new_episode = True
            self._buffer[-1]["seed"] = self._current_seed  # type: ignore
            # Only check episode scheduler to save in-memory data to temp HDF5 file when episode is done
            if self.max_buffer_episodes is not None:
                clear_buffers = (self._episode_id + 1) % self.max_buffer_episodes == 0

        if clear_buffers:
            self.clear_buffer_to_tmp_file()

        if clear_buffers or step_data["terminations"] or step_data["truncations"]:
            self._buffer.append({})

        if step_data["terminations"] or step_data["truncations"]:
            self._episode_id += 1

        return obs, rew, terminated, truncated, info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """Gymnasium environment reset."""
        obs, info = self.env.reset(seed=seed, options=options)
        step_data = self._step_data_callback(env=self, obs=obs, info=info)

        assert STEP_DATA_KEYS.issubset(
            step_data.keys()
        ), "One or more required keys is missing from 'step-data'"

        # If last episode in global buffer has saved steps, we need to check if it was truncated or terminated
        # If the last element in the buffer is not an empty dictionary, then we need to auto-truncate the episode.
        if self._buffer[-1]:
            if (
                not self._buffer[-1]["terminations"][-1]
                and not self._buffer[-1]["truncations"][-1]
            ):
                self._buffer[-1]["truncations"][-1] = True
                self._buffer[-1]["seed"] = self._current_seed  # type: ignore

                # New episode
                self._episode_id += 1

                if (
                    self.max_buffer_episodes is not None
                    and self._episode_id % self.max_buffer_episodes == 0
                ):
                    self.clear_buffer_to_tmp_file()

                # add new episode buffer
                self._buffer.append({})
        else:
            # In the case that the past episode is already stored in the tmp hdf5 file because of caching,
            # we need to check if it was truncated or terminated, if not then auto-truncate
            if (
                len(self._buffer) == 1
                and not self._last_episode_group_term_or_trunc
                and self._episode_id != 0
            ):
                self._eps_group["truncations"][-1] = True
                self._last_episode_group_term_or_trunc = True
                self._eps_group.attrs["seed"] = self._current_seed

                # New episode
                self._episode_id += 1

                # Compute metadata, use episode dataset in hdf5 file
                self._episode_metadata_callback(self._eps_group)

        self._buffer[-1] = self._add_to_episode_buffer(self._buffer[-1], step_data)

        if seed is None:
            self._current_seed = str(None)
        else:
            self._current_seed = seed

        self._reset_called = True

        return obs, info

    def clear_buffer_to_tmp_file(self, truncate_last_episode: bool = False):
        """Save the global buffer in-memory to a temporary HDF5 file in disk.

        Args:
            truncate_last_episode (bool, optional): If True the last episode from the buffer will be truncated before saving to disk. Defaults to False.
        """

        def get_h5py_subgroup(group: h5py.Group, name: str) -> h5py.Group:
            """Get a subgroup from an h5py group.

            If the subgroup does not exist, create and return and empty group with the given name.

            Args:
                group (h5py.Group): the h5py group object to look for/create the subgroup.
                name (str): name of the subgroup.

            Returns:
                subgroup (h5py.Group)
            """
            if name in group:
                subgroup = group.get(name)
                assert isinstance(subgroup, h5py.Group)
            else:
                subgroup = group.create_group(name)

            return subgroup

        def clear_buffer(dictionary_buffer: EpisodeBuffer, episode_group: h5py.Group):
            """Inner function to recursively save the nested data dictionaries in an episode buffer.

            Args:
                dictionary_buffer (EpisodeBuffer): ditionary with keys to store as independent HDF5 datasets if the value is a list buffer
                or create another group if value is a dictionary.
                episode_group (h5py.Group): HDF5 group to store the datasets from the dictionary_buffer.
            """
            for key, data in dictionary_buffer.items():
                if isinstance(data, dict):
                    eps_group_to_clear = get_h5py_subgroup(episode_group, key)
                    clear_buffer(data, eps_group_to_clear)
                elif all(map(lambda elem: isinstance(elem, tuple), data)):
                    # we have a list of tuples, so we need to act appropriately
                    dict_data = {
                        f"_index_{str(i)}": [entry[i] for entry in data]
                        for i, _ in enumerate(data[0])
                    }
                    eps_group_to_clear = get_h5py_subgroup(episode_group, key)
                    clear_buffer(dict_data, eps_group_to_clear)
                elif all(map(lambda elem: isinstance(elem, OrderedDict), data)):
                    # we have a list of OrderedDicts, so we need to act appropriately
                    dict_data = {
                        key: [entry[key] for entry in data]
                        for key, value in data[0].items()
                    }
                    eps_group_to_clear = get_h5py_subgroup(episode_group, key)
                    clear_buffer(dict_data, eps_group_to_clear)
                else:
                    if all(map(lambda elem: isinstance(elem, str), data)):
                        data_shape = (len(data),)
                        dtype = h5py.string_dtype(encoding="utf-8")
                    else:
                        data = np.asarray(data)
                        data_shape = data.shape
                        dtype = data.dtype
                        assert np.all(
                            np.logical_not(np.isnan(data))
                        ), "Nan found after cast to nump array, check the type of 'data'."

                    if (
                        not self._last_episode_group_term_or_trunc
                        and key in episode_group
                    ):
                        current_dataset_shape = episode_group[key].shape[0]
                        episode_group[key].resize(
                            current_dataset_shape + len(data), axis=0
                        )
                        episode_group[key][-len(data) :] = data
                    else:
                        if not current_episode_group_term_or_trunc:
                            data_shape = (None,) + data_shape[1:]  # resizable dataset

                        episode_group.create_dataset(
                            key, data=data, maxshape=data_shape, dtype=dtype
                        )

        for i, eps_buff in enumerate(self._buffer):
            # Make sure that the episode has stepped, by checking if the 'actions' key has been added to the episode buffer.
            if "actions" not in eps_buff:
                continue

            current_episode_group_term_or_trunc = (
                eps_buff["terminations"][-1] or eps_buff["truncations"][-1]
            )

            # Check if last episode group is terminated or truncated
            if self._last_episode_group_term_or_trunc:
                # Add new episode group
                current_episode_id = self._episode_id + i + 1 - len(self._buffer)
                self._eps_group = self._tmp_f.create_group(
                    f"episode_{current_episode_id}"
                )
                self._eps_group.attrs["id"] = current_episode_id

            if current_episode_group_term_or_trunc:
                # Add seed to episode metadata if the current episode has finished
                # Remove seed key from episode buffer before storing datasets to file
                self._eps_group.attrs["seed"] = eps_buff.pop("seed")
            clear_buffer(eps_buff, self._eps_group)

            if not self._last_episode_group_term_or_trunc:
                self._last_episode_n_steps += len(eps_buff["actions"])
            else:
                self._last_episode_n_steps = len(eps_buff["actions"])

            if current_episode_group_term_or_trunc:
                # Compute metadata, use episode dataset in hdf5 file
                self._episode_metadata_callback(self._eps_group)

            self._last_episode_group_term_or_trunc = current_episode_group_term_or_trunc

        if not self._last_episode_group_term_or_trunc and truncate_last_episode:
            self._eps_group["truncations"][-1] = True
            self._last_episode_group_term_or_trunc = True
            self._eps_group.attrs["seed"] = self._current_seed

            # New episode
            self._episode_id += 1

            # Compute metadata, use episode dataset in hdf5 file
            self._episode_metadata_callback(self._eps_group)

        # Clear in-memory buffers
        self._buffer.clear()

    def save_to_disk(
        self, path: str, dataset_metadata: Optional[Dict[str, Any]] = None
    ):
        """Save all in-memory buffer data and move temporary HDF5 file to a permanent location in disk.

        Args:
            path (str): path to store permanent HDF5, i.e: '/home/foo/datasets/data.hdf5'
            dataset_metadata (Dict, optional): additional metadata to add to HDF5 dataset file as attributes. Defaults to {}.
        """
        if dataset_metadata is None:
            dataset_metadata = {}

        # Dump everything in memory buffers to tmp_dataset.hdf5 and truncate last episode
        self.clear_buffer_to_tmp_file(truncate_last_episode=True)

        for key, value in dataset_metadata.items():
            self._tmp_f.attrs[key] = value

        assert (
            "observation_space" not in dataset_metadata.keys()
        ), "'observation_space' is not allowed as an optional key."
        assert (
            "action_space" not in dataset_metadata.keys()
        ), "'action_space' is not allowed as an optional key."

        action_space_str = serialize_space(self.dataset_action_space)
        observation_space_str = serialize_space(self.dataset_observation_space)

        self._tmp_f.attrs["action_space"] = action_space_str
        self._tmp_f.attrs["observation_space"] = observation_space_str

        self._buffer.append({})

        # Reset episode count
        self._episode_id = 0

        self._tmp_f.attrs["total_episodes"] = len(self._tmp_f.keys())
        self._tmp_f.attrs["total_steps"] = sum(
            [
                episode_group.attrs["total_steps"]
                for episode_group in self._tmp_f.values()
            ]
        )

        # Close tmp_dataset.hdf5
        self._tmp_f.close()

        # Move tmp_dataset.hdf5 to specified directory
        shutil.move(os.path.join(self._tmp_dir.name, "tmp_dataset.hdf5"), path)

        self._tmp_f = h5py.File(
            os.path.join(self._tmp_dir.name, "tmp_dataset.hdf5"), "a", track_order=True
        )

    def close(self):
        """Close the Gymnasium environment.

        Clear buffer and close temporary directory.
        """
        super().close()

        # Clear buffer
        self._buffer.clear()

        # Close tmp_dataset.hdf5
        self._tmp_f.close()
        shutil.rmtree(self._tmp_dir.name)

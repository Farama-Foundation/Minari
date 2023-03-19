from __future__ import annotations

import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional, SupportsFloat, Type, TypeVar, Union
from typing_extensions import TypedDict

import gymnasium as gym
import h5py
import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType, ObsType


EpisodeBufferValues = TypeVar("EpisodeBufferValues", List[Any], "EpisodeBuffer")
EpisodeBuffer = Dict[str, EpisodeBufferValues]


class StepData(TypedDict):
    observations: Any
    actions: Optional[Any]
    rewards: Optional[Any]
    terminations: Optional[bool]
    truncations: Optional[bool]
    infos: Dict[str, Any]


STEP_DATA_KEYS = {
    "actions",
    "observations",
    "rewards",
    "truncations",
    "terminations",
}


class EpisodeMetadataCallback:
    """Callback to full episode after saving to hdf5 file as a group.

    This callback can be overridden to add extra metadata attributes or statistics to
    each HDF5 episode group in the Minari dataset. The custom callback can then be
    passed to the DataCollectorV0 wrapper to the `episode_metadata_callback` argument.

    TODO: add more default statistics to episode datasets
    """

    def __call__(self, eps_group: h5py.Group):
        """Callback method.

        Override this method to add custom attribute metadata to the episode group.

        Args:
            eps_group (h5py.Group): the HDF5 group that contains an episode's datasets
        """
        eps_group["rewards"].attrs["sum"] = np.sum(eps_group["rewards"])
        eps_group["rewards"].attrs["mean"] = np.mean(eps_group["rewards"])
        eps_group["rewards"].attrs["std"] = np.std(eps_group["rewards"])
        eps_group["rewards"].attrs["max"] = np.max(eps_group["rewards"])
        eps_group["rewards"].attrs["min"] = np.min(eps_group["rewards"])

        eps_group.attrs["total_steps"] = eps_group["actions"].shape[0]


class StepDataCallback:
    """Callback to create step data dictionary from the return data of each Gymnasium environment step.

    The current callback automatically detects observation/action spaces that need
    to be flatten before saving to HDF5 file (currently only supports Dict or Tuple
    Gymnasium spaces. Text, Sequence, and Graph are currently not compatible with
    Minari).

    This callback can be overridden to add extra environment information in each step or
    edit the observation, action, reward, termination, truncation, or info returns.
    """

    def __init__(self, env: gym.Env):
        self.env = env

        def check_flatten_space(space: gym.spaces.Space):
            """Check if space needs to be flatten or if it's not supported by Minari.

            Args:
                space: the Gymnasium space to be checked

            Returns:
                bool: True if space needs to be flatten before storing in HDF5 dataset. False otherwise.

            ValueError: If space is/contains Text, Sequence, or Graph space types
            """
            if isinstance(space, spaces.Dict):
                for s in space.spaces.values():
                    check_flatten_space(s)
                return True
            elif isinstance(space, spaces.Tuple):
                for s in space.spaces:
                    check_flatten_space(s)
                return True
            elif isinstance(
                self.env.observation_space, (spaces.Text, spaces.Sequence, spaces.Graph)
            ):
                ValueError(f"Minari doesn't support space of type {space}")
            else:
                return False

        # check if observation/action need to be flatten before saving to HDF5
        self.flatten_observation = check_flatten_space(self.env.observation_space)
        self.flatten_action = check_flatten_space(self.env.action_space)

    def __call__(
        self,
        env: gym.Env,
        obs: Any,
        info: Dict[str, Any],
        action: Optional[Any] = None,
        rew: Optional[Any] = None,
        terminated: Optional[bool] = None,
        truncated: Optional[bool] = None,
    ) -> StepData:
        """Callback method.

        The input arguments belong to a Gymnasium stepping transition: `obs, rew, terminated, truncated, info = env.step(action)`.
        Override this method to add additional keys or edit each environment's step returns. Additional nested dictionaries can be added to the returned step dictionary
        as follows:

        ```
            class CustomStepDataCallback(StepDataCallback):
                def __call__(self, env, **kwargs):
                    step_data = super().__call__(env, **kwargs)

                    step_data['environment_states'] = {}
                    step_data['environment_states']['pose'] = {}
                    step_data['environment_states']['pose']['position'] = env.position
                    step_data['environment_states']['pose']['orientation'] = env.orientation
                    step_data['environment_states']['velocity'] = env.velocity

                    return step_data
        ```

        The episode groups in the HDF5 file of this Minari dataset will contain a subgroup called `environment_states` with dataset `velocity` and another subgroup called `pose`
        with datasets `position` and `orientation`

        Args:
            env (gym.Env): current Gymnasium environment.
            obs (Any): observation returned by `env.step(action)`
            info (Dict): information dictionary returned by `env.step(action)`
            action (Optional[Any], optional): stepping action in `env.step(action)`. Defaults to None.
            rew (Optional[Any], optional): reward returned by `env.step(action)`. Defaults to None.
            terminated (Optional[Any], optional): terminated returned by `env.step(action)`. Defaults to None.
            truncated (Optional[Any], optional): truncated returned by `env.step(action)`. Defaults to None.

        Returns:
            Dict: dictionary step data. Must contain the keys in STEP_DATA_KEYS = {'actions', 'observations',
                    'rewards', 'terminations', 'truncations', 'infos'}. Additional key's can be added with nested dictionaries
        """
        if action is not None:
            # Flatten the actions
            if self.flatten_action:
                action = spaces.flatten(self.env.action_space, action)
        # Flatten the observations
        if self.flatten_observation:
            obs = spaces.flatten(self.env.observation_space, obs)

        step_data: StepData = {
            "actions": action,
            "observations": obs,
            "rewards": rew,
            "terminations": terminated,
            "truncations": truncated,
            "infos": info,
        }

        return step_data


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

        dataset = minari.create_dataset_from_collector_env(dataset_name="EnvID-dataset", collector_env=env, **kwargs)

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
        self.env = env
        self._step_data_callback = step_data_callback(env)

        self._episode_metadata_callback = episode_metadata_callback()
        self._record_infos = record_infos

        if max_buffer_steps is not None and max_buffer_episodes is not None:
            raise ValueError("Choose step or episode scheduler not both")

        self.max_buffer_episodes = max_buffer_episodes
        self.max_buffer_steps = max_buffer_steps

        # Initialzie empty buffer
        self._buffer: List[EpisodeBuffer] = [{key: [] for key in STEP_DATA_KEYS}]

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

        assert self.env.spec is not None
        self._tmp_f.attrs["env_spec"] = self.env.spec.to_json()
        self._tmp_f.attrs[
            "flatten_observation"
        ] = self._step_data_callback.flatten_observation
        self._tmp_f.attrs["flatten_action"] = self._step_data_callback.flatten_action

        self._new_episode = False

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
            buffer (Dict): dictionary episode buffer
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
                    assert isinstance(episode_buffer[key], dict)
                    episode_buffer[key] = self._add_to_episode_buffer(
                        episode_buffer[key], value
                    )
                else:
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

        # force step data dicitonary to include keys corresponding to Gymnasium step returns:
        # actions, observations, rewards, terminations, truncatins, and infos
        assert STEP_DATA_KEYS.issubset(step_data.keys())

        self._step_id += 1

        # check if buffer needs to be cleared to temp file due to maximum step scheduler
        if self.max_buffer_steps is not None:
            clear_buffers = (
                self._step_id % self.max_buffer_steps == 0 and self._step_id != 0
            )
        else:
            clear_buffers = False

        # Get initial observation from previous episode if reset has not been called after termination or truncation
        # This may happen if the step_data_callback truncates or terminates the episode under certain conditions.
        if self._new_episode and not self._reset_called:
            self._buffer[-1]["observations"] = [self._previous_eps_final_obs]
            self._new_episode = False

        # add step data to last episode buffer
        self._buffer[-1] = self._add_to_episode_buffer(self._buffer[-1], step_data)

        if step_data["terminations"] or step_data["truncations"]:
            # New episode
            self._episode_id += 1
            self._previous_eps_final_obs = step_data["observations"]
            self._reset_called = False
            self._new_episode = True
            self._buffer[-1]["seed"] = self._current_seed
            # Only check episode scheduler to save in-memory data to temp HDF5 file when episode is done
            if self.max_buffer_episodes is not None:
                clear_buffers = self._episode_id % self.max_buffer_episodes == 0

        if clear_buffers:
            self.clear_buffer_to_tmp_file()

        # add new episode buffer to global buffer when episode finishes with truncation or termination
        if clear_buffers or step_data["terminations"] or step_data["truncations"]:
            self._buffer.append({key: [] for key in STEP_DATA_KEYS})

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

        assert STEP_DATA_KEYS.issubset(step_data.keys())

        # If last episode in global buffer has saved steps
        if len(self._buffer[-1]["actions"]) > 0:
            # If the last episode is not term/trunc then truncate the episode
            if (
                not self._buffer[-1]["terminations"][-1]
                and not self._buffer[-1]["truncations"][-1]
            ):
                self._buffer[-1]["truncations"][-1] = True
                self._buffer[-1]["seed"] = self._current_seed

                # New episode
                self._episode_id += 1

                if (
                    self.max_buffer_episodes is not None
                    and self._episode_id % self.max_buffer_episodes == 0
                ):
                    self.clear_buffer_to_tmp_file()

                # add new episode buffer
                self._buffer.append({key: [] for key in STEP_DATA_KEYS})

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

        def clear_buffer(dictionary_buffer: EpisodeBuffer, episode_group: h5py.Group):
            """Inner function to recursively save the nested data dictionaries in an episode buffer.

            Args:
                dictionary_buffer (EpisodeBuffer): ditionary with keys to store as independent HDF5 datasets if the value is a list buffer
                or create another group if value is a dictionary.
                episode_group (h5py.Group): HDF5 group to store the datasets from the dictionary_buffer.
            """
            for key, data in dictionary_buffer.items():
                if isinstance(data, dict):
                    if key in episode_group:
                        eps_group_to_clear = episode_group[key]
                    else:
                        eps_group_to_clear = episode_group.create_group(key)
                    clear_buffer(data, eps_group_to_clear)
                else:
                    # convert data to numpy
                    np_data = np.asarray(data)
                    assert np.all(np.logical_not(np.isnan(np_data)))

                    # Check if last episode group is terminated or truncated
                    if (
                        not self._last_episode_group_term_or_trunc
                        and key in episode_group
                    ):
                        # Append to last episode group datasets
                        if key not in STEP_DATA_KEYS and key != "infos":
                            # check current dataset size directly from hdf5 since
                            # non step data (actions, obs, rew, term, trunc) may not be
                            # added in a per-step/sequential basis, including "infos"
                            current_dataset_shape = episode_group[key].shape[0]
                        else:
                            current_dataset_shape = self._last_episode_n_steps
                            if key == "observations":
                                current_dataset_shape += (
                                    1  # include initial observation
                                )
                        episode_group[key].resize(
                            current_dataset_shape + len(data), axis=0
                        )
                        episode_group[key][-len(data) :] = np_data
                    else:
                        if not current_episode_group_term_or_trunc:
                            # Create resizable datasets
                            episode_group.create_dataset(
                                key,
                                data=np_data,
                                maxshape=(None,) + np_data.shape[1:],
                                chunks=True,
                            )
                        else:
                            # Dump everything to episode group
                            episode_group.create_dataset(key, data=np_data, chunks=True)

        for i, eps_buff in enumerate(self._buffer):
            if len(eps_buff["actions"]) == 0:
                # Make sure that the episode has stepped
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

    def save_to_disk(self, path: str, dataset_metadata: Dict = {}):
        """Save all in-memory buffer data and move temporary HDF5 file to a permanent location in disk.

        Args:
            path (str): path to store permanent HDF5, i.e: '/home/foo/datasets/data.hdf5'
            dataset_metadata (Dict, optional): additional metadata to add to HDF5 dataset file as attributes. Defaults to {}.
        """
        # Dump everything in memory buffers to tmp_dataset.hdf5 and truncate last episode
        self.clear_buffer_to_tmp_file(truncate_last_episode=True)

        for key, value in dataset_metadata.items():
            self._tmp_f.attrs[key] = value

        self._buffer.append({key: [] for key in STEP_DATA_KEYS})

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

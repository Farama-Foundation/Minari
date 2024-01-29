from __future__ import annotations

import copy
import inspect
import os
import secrets
import shutil
import tempfile
import warnings
from typing import Any, Callable, Dict, List, Optional, SupportsFloat, Type, Union

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType
from gymnasium.envs.registration import EnvSpec

from minari.data_collector.callbacks import (
    STEP_DATA_KEYS,
    EpisodeMetadataCallback,
    StepData,
    StepDataCallback,
)
from minari.dataset.minari_dataset import MinariDataset
from minari.dataset.minari_storage import MinariStorage


# H5Py supports ints up to uint64
AUTOSEED_BIT_SIZE = 64

EpisodeBuffer = Dict[str, Any]  # TODO: narrow this down


def __getattr__(name):
    if name == "DataCollectorV0":
        stacklevel = len(inspect.stack(0))
        warnings.warn("DataCollectorV0 is deprecated and will be removed. Use DataCollector instead.", DeprecationWarning, stacklevel=stacklevel)
        return DataCollector
    elif name == "__path__":
        return False  # see https://stackoverflow.com/a/60803436
    else:
        raise ImportError(f"cannot import name '{name}' from '{__name__}' ({__file__})")


class DataCollector(gym.Wrapper):
    r"""Gymnasium environment wrapper that collects step data.

    This wrapper is meant to work as a temporary buffer of the environment data before creating a Minari dataset. The creation of the buffers
    that will be convert to a Minari dataset is agnostic to the user:

    .. code::

        import minari
        import gymnasium as gym

        env = minari.DataCollector(gym.make('EnvID'))

        env.reset()

        for _ in range(num_steps):
            action = env.action_space.sample()
            obs, rew, terminated, truncated, info = env.step()

            if terminated or truncated:
                env.reset()

        dataset = env.create_dataset(dataset_id="env_name-dataset_name-v(version)", **kwargs)

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
          when the Minari dataset is created. To move all the stored data to a permanent location use DataCollector.save_to_disK(path_to_permanent_location).
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
        """Initialize the data collector attributes and create the temporary directory for caching.

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

        self._record_infos = record_infos
        self._reference_info = None
        self.max_buffer_steps = max_buffer_steps

        # Initialzie empty buffer
        self._buffer: List[EpisodeBuffer] = []

        self._step_id = -1
        self._episode_id = -1

    def _add_step_data(
        self,
        episode_buffer: EpisodeBuffer,
        step_data: Union[StepData, Dict],
    ):
        """Add step data dictionary to episode buffer.

        Args:
            episode_buffer (Dict): dictionary episode buffer
            step_data (Dict): dictionary with data for a single step

        Returns:
            Dict: new dictionary episode buffer with added values from step_data
        """
        dict_data = dict(step_data)
        if not self._record_infos:
            dict_data = {k: v for k, v in step_data.items() if k != "infos"}
        else:
            assert self._reference_info is not None
            if not _check_infos_same_shape(
                self._reference_info, step_data["infos"]
            ):
                raise ValueError(
                    "Info structure inconsistent with info structure returned by original reset."
                )

        self._add_to_episode_buffer(episode_buffer, dict_data)

    def _add_to_episode_buffer(
        self,
        episode_buffer: EpisodeBuffer,
        step_data: Dict[str, Any],
    ):
        for key, value in step_data.items():
            if value is None:
                continue

            if key not in episode_buffer:
                episode_buffer[key] = {} if isinstance(value, dict) else []

            if isinstance(value, dict):
                assert isinstance(
                    episode_buffer[key], dict
                ), f"Element to be inserted is type 'dict', but buffer accepts type {type(episode_buffer[key])}"

                self._add_to_episode_buffer(episode_buffer[key], value)
            else:
                assert isinstance(
                    episode_buffer[key], list
                ), f"Element to be inserted is type 'list', but buffer accepts type {type(episode_buffer[key])}"
                episode_buffer[key].append(value)

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

        # Force step data dictionary to include keys corresponding to Gymnasium step returns:
        # actions, observations, rewards, terminations, truncations, and infos
        assert STEP_DATA_KEYS.issubset(
            step_data.keys()
        ), "One or more required keys is missing from 'step-data'."
        # Check that the stored obs/act spaces comply with the dataset spaces
        assert self._storage.observation_space.contains(
            step_data["observations"]
        ), "Observations are not in observation space."
        assert self._storage.action_space.contains(
            step_data["actions"]
        ), "Actions are not in action space."

        self._step_id += 1
        self._add_step_data(self._buffer[-1], step_data)

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
            self._add_step_data(eps_buff, previous_data)
            self._buffer.append(eps_buff)

        return obs, rew, terminated, truncated, info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """Gymnasium environment reset.

        If no seed is set, one will be automatically generated, for reproducibility,
        unless ``minari_autoseed=False`` in the ``options`` dictionary.

        Args:
            seed (optional int): The seed that is used to initialize the environment's PRNG.
                If no seed is specified, one will be automatically generated (by default).
            options (optional dict): Additional information to specify how the environment is reset.
                Set ``minari_autoseed=False`` to disable automatic seeding.

        Returns:
            observation (ObsType): Observation of the initial state.
            info (dictionary): Auxiliary information complementing ``observation``.
        """
        autoseed_enabled = (not options) or options.get("minari_autoseed", True)
        if seed is None and autoseed_enabled:
            seed = secrets.randbits(AUTOSEED_BIT_SIZE)

        obs, info = self.env.reset(seed=seed, options=options)
        step_data = self._step_data_callback(env=self.env, obs=obs, info=info)
        self._episode_id += 1

        if self._record_infos and self._reference_info is None:
            self._reference_info = step_data["infos"]

        assert STEP_DATA_KEYS.issubset(
            step_data.keys()
        ), "One or more required keys is missing from 'step-data'"

        self._validate_buffer()
        episode_buffer = {
            "seed": str(None) if seed is None else seed,
            "id": self._episode_id
        }
        self._add_step_data(episode_buffer, step_data)
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
        """Add extra data to Minari dataset from collector environment buffers (DataCollector).

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

    def create_dataset(
        self,
        dataset_id: str,
        eval_env: Optional[str | gym.Env | EnvSpec] = None,
        algorithm_name: Optional[str] = None,
        author: Optional[str] = None,
        author_email: Optional[str] = None,
        code_permalink: Optional[str] = None,
        ref_min_score: Optional[float] = None,
        ref_max_score: Optional[float] = None,
        expert_policy: Optional[Callable[[ObsType], ActType]] = None,
        num_episodes_average_score: int = 100,
        minari_version: Optional[str] = None,
    ):
        """Create a Minari dataset using the data collected from stepping with a Gymnasium environment wrapped with a `DataCollector` Minari wrapper.

        The ``dataset_id`` parameter corresponds to the name of the dataset, with the syntax as follows:
        ``(env_name-)(dataset_name)(-v(version))`` where ``env_name`` identifies the name of the environment used to generate the dataset ``dataset_name``.
        This ``dataset_id`` is used to load the Minari datasets with :meth:`minari.load_dataset`.

        Args:
            dataset_id (str): name id to identify Minari dataset
            buffer (list[Dict[str, Union[list, Dict]]]): list of episode dictionaries with data
            eval_env (Optional[str|gym.Env|EnvSpec]): Gymnasium environment(gym.Env)/environment id(str)/environment spec(EnvSpec) to use for evaluation with the dataset. After loading the dataset, the environment can be recovered as follows: `MinariDataset.recover_environment(eval_env=True).
                                                    If None the `env` used to collect the buffer data should be used for evaluation.
            algorithm_name (Optional[str], optional): name of the algorithm used to collect the data. Defaults to None.
            author (Optional[str], optional): author that generated the dataset. Defaults to None.
            author_email (Optional[str], optional): email of the author that generated the dataset. Defaults to None.
            code_permalink (Optional[str], optional): link to relevant code used to generate the dataset. Defaults to None.
            ref_min_score( Optional[float], optional): minimum reference score from the average returns of a random policy. This value is later used to normalize a score with :meth:`minari.get_normalized_score`. If default None the value will be estimated with a default random policy.
            ref_max_score (Optional[float], optional: maximum reference score from the average returns of a hypothetical expert policy. This value is used in :meth:`minari.get_normalized_score`. Default None.
            expert_policy (Optional[Callable[[ObsType], ActType], optional): policy to compute `ref_max_score` by averaging the returns over a number of episodes equal to  `num_episodes_average_score`.
                                                                            `ref_max_score` and `expert_policy` can't be passed at the same time. Default to None
            num_episodes_average_score (int): number of episodes to average over the returns to compute `ref_min_score` and `ref_max_score`. Default to 100.
            minari_version (Optional[str], optional): Minari version specifier compatible with the dataset. If None (default) use the installed Minari version.

        Returns:
            MinariDataset
        """
        # TODO: move the import to top of the file after removing minari.create_dataset_from_collector_env() in 0.5.0
        from minari.utils import _generate_dataset_metadata, _generate_dataset_path
        dataset_path = _generate_dataset_path(dataset_id)
        metadata: Dict[str, Any] = _generate_dataset_metadata(
            dataset_id,
            copy.deepcopy(self.env.spec),
            eval_env,
            algorithm_name,
            author,
            author_email,
            code_permalink,
            ref_min_score,
            ref_max_score,
            expert_policy,
            num_episodes_average_score,
            minari_version,
        )

        self.save_to_disk(dataset_path, metadata)

        # will be able to calculate dataset size only after saving the disk, so updating the dataset metadata post `save_to_disk` method

        dataset = MinariDataset(dataset_path)
        metadata['dataset_size'] = dataset.storage.get_size()
        dataset.storage.update_metadata(metadata)
        return dataset

    def save_to_disk(
        self, path: str | os.PathLike, dataset_metadata: Dict[str, Any] = {}
    ):
        """Save all in-memory buffer data and move temporary files to a permanent location in disk.

        Args:
            path (str): path to store the dataset, e.g.: '/home/foo/datasets/data'
            dataset_metadata (Dict, optional): additional metadata to add to the dataset file. Defaults to {}.
        """
        warnings.warn("This method is deprecated and will become private in v0.5.0.", DeprecationWarning, stacklevel=2)
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


def _check_infos_same_shape(info_1: dict, info_2: dict):
    if info_1.keys() != info_2.keys():
        return False
    for key in info_1.keys():
        if type(info_1[key]) is not type(info_2[key]):
            return False
        if isinstance(info_1[key], dict):
            return _check_infos_same_shape(info_1[key], info_2[key])
        elif isinstance(info_1[key], np.ndarray):
            return (info_1[key].shape == info_2[key].shape) and (info_1[key].dtype == info_2[key].dtype)
    return True

from __future__ import annotations

import copy
import os
import secrets
import shutil
import tempfile
import warnings
from typing import Any, Callable, Dict, Optional, SupportsFloat, Type

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType
from gymnasium.envs.registration import EnvSpec

from minari.data_collector.callbacks import EpisodeMetadataCallback, StepDataCallback
from minari.data_collector.episode_buffer import EpisodeBuffer
from minari.dataset.minari_dataset import MinariDataset, parse_dataset_id
from minari.dataset.minari_storage import MinariStorage
from minari.namespace import create_namespace, list_local_namespaces
from minari.utils import _generate_dataset_metadata, _generate_dataset_path


# H5Py supports ints up to uint64
AUTOSEED_BIT_SIZE = 64


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

        dataset = env.create_dataset(dataset_id="env_name/dataset_name-v(version)", **kwargs)

    Some of the characteristics of this wrapper:

        * The step data is stored per episode in dictionaries. This dictionaries are then stored in-memory in a global list buffer. The
          episode dictionaries contain items with list buffers as values for the main episode step datasets `observations`, `actions`,
          `terminations`, and `truncations`, the `infos` key can be a list or another nested dictionary with extra datasets.

        * A new episode dictionary buffer is created if the env.step(action) call returns `truncated` or `terminated`, or if the environment calls
          env.reset(). If calling reset and the previous episode was not `truncated` or `terminated`, this will automatically be `truncated`.

    """

    def __init__(
        self,
        env: gym.Env,
        step_data_callback: Type[StepDataCallback] = StepDataCallback,
        episode_metadata_callback: Type[
            EpisodeMetadataCallback
        ] = EpisodeMetadataCallback,
        record_infos: bool = False,
        observation_space: Optional[gym.Space] = None,
        action_space: Optional[gym.Space] = None,
        data_format: Optional[str] = None,
    ):
        """Initialize the data collector attributes and create the temporary directory for caching.

        Args:
            env (gym.Env): Gymnasium environment
            step_data_callback (type[StepDataCallback], optional): Callback class to edit/update step databefore storing to buffer. Defaults to StepDataCallback.
            episode_metadata_callback (type[EpisodeMetadataCallback], optional): Callback class to add custom metadata to episode group in HDF5 file. Defaults to EpisodeMetadataCallback.
            record_infos (bool, optional): If True record the info return key of each step. Defaults to False.
            observation_space (gym.Space): Observation space of the dataset. The default value is the environment observation space.
            action_space (gym.Space): Action space of the dataset. The default value is the environment action space.
            data_format (str, optional): Data format to store the data in the Minari dataset. If None (defaults), it will use the default format of MinariStorage.
        """
        super().__init__(env)
        self._step_data_callback = step_data_callback()
        self._episode_metadata_callback = episode_metadata_callback()

        self.datasets_path = os.environ.get("MINARI_DATASETS_PATH")
        if self.datasets_path is None:
            self.datasets_path = os.path.join(
                os.path.expanduser("~"), ".minari", "datasets"
            )
        if not os.path.exists(self.datasets_path):
            os.makedirs(self.datasets_path)
        self.data_format = data_format

        if observation_space is None:
            observation_space = env.observation_space
        self._observation_space = observation_space
        if action_space is None:
            action_space = env.action_space
        self._action_space = action_space

        self._record_infos = record_infos
        self._buffer: Optional[EpisodeBuffer] = None
        self._episode_id = 0
        self._reset_storage()

    def _reset_storage(self):
        self._episode_id = 0
        self._tmp_dir = tempfile.TemporaryDirectory(dir=self.datasets_path)
        data_format_kwarg = (
            {"data_format": self.data_format} if self.data_format is not None else {}
        )
        self._storage = MinariStorage.new(
            self._tmp_dir.name,
            observation_space=self._observation_space,
            action_space=self._action_space,
            env_spec=self.env.spec,
            **data_format_kwarg,
        )

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

        if not self._storage.observation_space.contains(step_data["observation"]):
            warnings.warn(
                "Observation is not in observation space.\n"
                f"Observation: {step_data['observation']}\nSpace: {self._storage.observation_space}"
            )
        if not self._storage.action_space.contains(step_data["action"]):
            warnings.warn(
                "Action is not in action space.\n"
                f"Action: {step_data['action']}\nSpace: {self._storage.action_space}",
            )

        assert self._buffer is not None
        if not self._record_infos:
            step_data["info"] = {}
        self._buffer = self._buffer.add_step_data(step_data)

        if step_data["termination"] or step_data["truncation"]:
            self._storage.update_episodes([self._buffer])
            self._episode_id += 1
            self._buffer = EpisodeBuffer(
                id=self._episode_id,
                observations=step_data["observation"],
                infos=step_data["info"],
            )

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
        self._flush_to_storage()

        autoseed_enabled = (not options) or options.get("minari_autoseed", True)
        if seed is None and autoseed_enabled:
            seed = secrets.randbits(AUTOSEED_BIT_SIZE)

        obs, info = self.env.reset(seed=seed, options=options)
        step_data = self._step_data_callback(env=self.env, obs=obs, info=info)

        if not self._storage.observation_space.contains(step_data["observation"]):
            warnings.warn(
                "Observation is not in observation space.\n"
                f"Observation: {step_data['observation']}\nSpace: {self._storage.observation_space}"
            )

        self._buffer = EpisodeBuffer(
            id=self._episode_id,
            seed=seed,
            options=options,
            observations=step_data["observation"],
            infos=step_data["info"] if self._record_infos else None,
        )
        return obs, info

    def add_to_dataset(self, dataset: MinariDataset):
        """Add extra data to Minari dataset from collector environment buffers (DataCollector).

        Args:
            dataset (MinariDataset): Dataset to add the data
        """
        self._flush_to_storage()

        first_id = dataset.storage.total_episodes
        dataset.storage.update_from_storage(self._storage)
        if dataset.episode_indices is not None:
            new_ids = first_id + np.arange(self._storage.total_episodes)
            dataset.episode_indices = np.append(dataset.episode_indices, new_ids)

        self._reset_storage()

    def create_dataset(
        self,
        dataset_id: str,
        eval_env: Optional[str | gym.Env | EnvSpec] = None,
        algorithm_name: Optional[str] = None,
        author: Optional[str | set] = None,
        author_email: Optional[str | set] = None,
        code_permalink: Optional[str] = None,
        ref_min_score: Optional[float] = None,
        ref_max_score: Optional[float] = None,
        expert_policy: Optional[Callable[[ObsType], ActType]] = None,
        num_episodes_average_score: int = 100,
        description: Optional[str] = None,
        requirements: Optional[list] = None,
    ):
        """Create a Minari dataset using the data collected from stepping with a Gymnasium environment wrapped with a `DataCollector` Minari wrapper.

        The ``dataset_id`` parameter corresponds to the name of the dataset, with the syntax as follows:
        ``(namespace/)(env_name/)dataset_name(-v[version])`` where ``env_name`` identifies the name of the environment used to generate the dataset. The `namespace` is optional.
        This ``dataset_id`` is used to load the Minari datasets with :meth:`minari.load_dataset`.

        Args:
            dataset_id (str): name id to identify Minari dataset
            eval_env (str | gym.Env | EnvSpec, optional): Gymnasium environment(gym.Env)/environment id(str)/environment spec(EnvSpec) to use for evaluation with the dataset. After loading the dataset, the environment can be recovered as follows: `MinariDataset.recover_environment(eval_env=True).
                                                    If None the `env` used to collect the buffer data should be used for evaluation.
            algorithm_name (str, optional): name of the algorithm used to collect the data. Defaults to None.
            author (str | set, optional): name of the author(s) that generated the dataset. Defaults to None.
            author_email (str | set, optional): email(s) of the author(s) that generated the dataset. Defaults to None.
            code_permalink (str, optional): link to relevant code used to generate the dataset. Defaults to None.
            ref_min_score(float, optional): minimum reference score from the average returns of a random policy. This value is later used to normalize a score with :meth:`minari.get_normalized_score`. If default None the value will be estimated with a default random policy.
            ref_max_score (float, optional): maximum reference score from the average returns of a hypothetical expert policy. This value is used in :meth:`minari.get_normalized_score`. Default None.
            expert_policy (Callable[[ObsType], ActType], optional): policy to compute `ref_max_score` by averaging the returns over a number of episodes equal to  `num_episodes_average_score`.
                                                                            `ref_max_score` and `expert_policy` can't be passed at the same time. Default to None
            num_episodes_average_score (int): number of episodes to average over the returns to compute `ref_min_score` and `ref_max_score`. Default to 100.
            description (str, optional): description of the dataset being created. Defaults to None.
            requirements (list of str, optional): list of requirements in pip-style to load the environment and reproduce the dataset. For example, `mujoco>=3.1.0,<3.2.0`, which indicate the supported version range for mujoco package. Defaults to None.

        Returns:
            MinariDataset
        """
        namespace = parse_dataset_id(dataset_id)[0]

        if namespace is not None and namespace not in list_local_namespaces():
            create_namespace(namespace)

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
            description,
            requirements,
        )

        self._save_to_disk(dataset_path, metadata)
        return MinariDataset(dataset_path)

    def _flush_to_storage(self):
        if self._buffer is not None and len(self._buffer) > 0:
            if not self._buffer.terminations[-1]:
                self._buffer.truncations[-1] = True
            self._storage.update_episodes([self._buffer])
            self._episode_id += 1
        self._buffer = None

    def _save_to_disk(
        self, path: str | os.PathLike, dataset_metadata: Dict[str, Any] = {}
    ):
        """Save all in-memory buffer data and move temporary files to a permanent location in disk.

        Args:
            path (str): path to store the dataset, e.g.: '/home/foo/datasets/data'
            dataset_metadata (Dict, optional): additional metadata to add to the dataset file. Defaults to {}.
        """
        self._flush_to_storage()

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

        self._reset_storage()

    def close(self):
        """Close the DataCollector.

        Clear buffer and close temporary directory.
        """
        super().close()
        self._buffer = None
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
            return (info_1[key].shape == info_2[key].shape) and (
                info_1[key].dtype == info_2[key].dtype
            )
    return True

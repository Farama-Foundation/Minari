from __future__ import annotations

import copy
import os
import warnings
from typing import Any, Callable, Dict, List, Optional, Union

import gymnasium as gym
import h5py
import numpy as np
from gymnasium.core import ActType, ObsType
from gymnasium.envs.registration import EnvSpec
from gymnasium.wrappers import RecordEpisodeStatistics

from minari import DataCollectorV0
from minari.dataset.minari_dataset import MinariDataset
from minari.dataset.minari_storage import clear_episode_buffer
from minari.serialization import serialize_space
from minari.storage.datasets_root_dir import get_dataset_path


class RandomPolicy:
    """A random action selection policy to compute `ref_min_score`."""

    def __init__(self, env: gym.Env):
        self.action_space = env.action_space
        self.action_space.seed(123)
        self.observation_space = env.observation_space

    def __call__(self, observation: ObsType) -> ActType:
        assert self.observation_space.contains(observation)
        return self.action_space.sample()


def combine_datasets(
    datasets_to_combine: List[MinariDataset], new_dataset_id: str, copy: bool = False
):
    """Combine a group of MinariDataset in to a single dataset with its own name id.

    A new HDF5 metadata attribute will be added to the new dataset called `combined_datasets`. This will
    contain a list of strings with the dataset names that were combined to form this new Minari dataset.

    Args:
        datasets_to_combine (list[MinariDataset]): list of datasets to be combined
        new_dataset_id (str): name id for the newly created dataset
        copy (bool): whether to copy the data to a new dataset or to create external link (see h5py.ExternalLink)
    """
    new_dataset_path = get_dataset_path(new_dataset_id)

    # Check if dataset already exists
    if not os.path.exists(new_dataset_path):
        new_dataset_path = os.path.join(new_dataset_path, "data")
        os.makedirs(new_dataset_path)
        new_data_path = os.path.join(new_dataset_path, "main_data.hdf5")
    else:
        raise ValueError(
            f"A Minari dataset with ID {new_dataset_id} already exists and it cannot be overridden. Please use a different dataset name or version."
        )

    with h5py.File(new_data_path, "a", track_order=True) as combined_data_file:
        combined_data_file.attrs["total_episodes"] = 0
        combined_data_file.attrs["total_steps"] = 0
        combined_data_file.attrs["dataset_id"] = new_dataset_id

        combined_data_file.attrs["combined_datasets"] = [
            dataset.spec.dataset_id for dataset in datasets_to_combine
        ]

        current_env_spec = None

        for dataset in datasets_to_combine:
            if not isinstance(dataset, MinariDataset):
                raise ValueError(f"The dataset {dataset} is not of type MinariDataset.")
            dataset_env_spec = dataset.spec.env_spec

            assert isinstance(dataset_env_spec, EnvSpec)
            # We have to check that all datasets can be merged by checking that they come from the same
            # environments. However, we override the time limit max_episode_steps with the max among all
            # the datasets to be combined. Then we check if the rest of the env_spec attributes are from
            # the same environment.
            if current_env_spec is None:
                current_env_spec = dataset_env_spec
            elif dataset_env_spec.max_episode_steps is not None:
                if current_env_spec.max_episode_steps is None:
                    current_env_spec.max_episode_steps = (
                        dataset_env_spec.max_episode_steps
                    )
                else:
                    if (
                        current_env_spec.max_episode_steps
                        < dataset_env_spec.max_episode_steps
                    ):
                        current_env_spec.max_episode_steps = (
                            dataset_env_spec.max_episode_steps
                        )
                    else:
                        dataset_env_spec.max_episode_steps = (
                            current_env_spec.max_episode_steps
                        )

            if current_env_spec != dataset_env_spec:
                raise ValueError(
                    "The datasets to be combined have different values for `env_spec` attribute."
                )

            last_episode_id = combined_data_file.attrs["total_episodes"]
            if copy:
                with h5py.File(dataset.spec.data_path, "r") as dataset_file:
                    for id in range(dataset.total_episodes):
                        dataset_file.copy(
                            dataset_file[f"episode_{id}"],
                            combined_data_file,
                            name=f"episode_{last_episode_id + id}",
                        )
                        combined_data_file[
                            f"episode_{last_episode_id + id}"
                        ].attrs.modify("id", last_episode_id + id)
            else:
                for id in range(dataset.total_episodes):
                    combined_data_file[
                        f"episode_{last_episode_id + id}"
                    ] = h5py.ExternalLink(dataset.spec.data_path, f"/episode_{id}")
                    combined_data_file[f"episode_{last_episode_id + id}"].attrs.modify(
                        "id", last_episode_id + id
                    )

            # Update metadata of minari dataset
            combined_data_file.attrs.modify(
                "total_episodes", last_episode_id + dataset.total_episodes
            )
            combined_data_file.attrs.modify(
                "total_steps",
                combined_data_file.attrs["total_steps"] + dataset.spec.total_steps,
            )

            # TODO: list of authors, and emails
            with h5py.File(dataset.spec.data_path, "r") as dataset_file:
                combined_data_file.attrs.modify("author", dataset_file.attrs["author"])
                combined_data_file.attrs.modify(
                    "author_email", dataset_file.attrs["author_email"]
                )

        assert current_env_spec is not None
        combined_data_file.attrs["env_spec"] = current_env_spec.to_json()

    return MinariDataset(new_data_path)


def split_dataset(
    dataset: MinariDataset, sizes: List[int], seed: Optional[int] = None
) -> List[MinariDataset]:
    """Split a MinariDataset in multiple datasets.

    Args:
        dataset (MinariDataset): the MinariDataset to split
        sizes (List[int]): sizes of the resulting datasets
        seed (Optiona[int]): random seed

    Returns:
        datasets (List[MinariDataset]): resulting list of datasets
    """
    if sum(sizes) > dataset.total_episodes:
        raise ValueError(
            "Incompatible arguments: the sum of sizes exceeds ",
            f"the number of episodes in the dataset ({dataset.total_episodes})",
        )
    generator = np.random.default_rng(seed=seed)
    indices = generator.permutation(dataset.episode_indices)
    out_datasets = []
    start_idx = 0
    for length in sizes:
        end_idx = start_idx + length
        slice_dataset = MinariDataset(
            dataset.spec.data_path, indices[start_idx:end_idx]
        )
        out_datasets.append(slice_dataset)
        start_idx = end_idx

    return out_datasets


def get_average_reference_score(
    env: gym.Env,
    policy: Callable[[ObsType], ActType],
    num_episodes: int,
) -> float:

    env = RecordEpisodeStatistics(env, num_episodes)
    episode_returns = []
    obs, _ = env.reset(seed=123)
    for _ in range(num_episodes):
        while True:
            action = policy(obs)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                episode_returns.append(info["episode"]["r"])
                obs, _ = env.reset()
                break

    mean_ref_score = np.mean(episode_returns, dtype=np.float32)
    return float(mean_ref_score)


def create_dataset_from_buffers(
    dataset_id: str,
    env: gym.Env,
    buffer: List[Dict[str, Union[list, Dict]]],
    algorithm_name: Optional[str] = None,
    author: Optional[str] = None,
    author_email: Optional[str] = None,
    code_permalink: Optional[str] = None,
    action_space: Optional[gym.spaces.Space] = None,
    observation_space: Optional[gym.spaces.Space] = None,
    ref_min_score: Optional[float] = None,
    ref_max_score: Optional[float] = None,
    expert_policy: Optional[Callable[[ObsType], ActType]] = None,
    num_episodes_average_score: int = 100,
):
    """Create Minari dataset from a list of episode dictionary buffers.

    The ``dataset_id`` parameter corresponds to the name of the dataset, with the syntax as follows:
    ``(env_name-)(dataset_name)(-v(version))`` where ``env_name`` identifies the name of the environment used to generate the dataset ``dataset_name``.
    This ``dataset_id`` is used to load the Minari datasets with :meth:`minari.load_dataset`.

    Each episode dictionary buffer must have the following items:
        * `observations`: np.ndarray of step observations. shape = (total_episode_steps + 1, (observation_shape)). Should include initial and final observation
        * `actions`: np.ndarray of step action. shape = (total_episode_steps + 1, (action_shape)).
        * `rewards`: np.ndarray of step rewards. shape = (total_episode_steps + 1, 1).
        * `terminations`: np.ndarray of step terminations. shape = (total_episode_steps + 1, 1).
        * `truncations`: np.ndarray of step truncations. shape = (total_episode_steps + 1, 1).

    Other additional items can be added as long as the values are np.ndarray's or other nested dictionaries.

    Args:
        dataset_id (str): name id to identify Minari dataset
        env (gym.Env): Gymnasium environment used to collect the buffer data
        buffer (list[Dict[str, Union[list, Dict]]]): list of episode dictionaries with data
        algorithm_name (Optional[str], optional): name of the algorithm used to collect the data. Defaults to None.
        author (Optional[str], optional): author that generated the dataset. Defaults to None.
        author_email (Optional[str], optional): email of the author that generated the dataset. Defaults to None.
        code_permalink (Optional[str], optional): link to relevant code used to generate the dataset. Defaults to None.
        ref_min_score (Optional[float], optional): minimum reference score from the average returns of a random policy. This value is later used to normalize a score with :meth:`minari.get_normalized_score`. If default None the value will be estimated with a default random policy.
                                                    Also note that this attribute will be added to the Minari dataset only if `ref_max_score` or `expert_policy` are assigned a valid value other than None.
        ref_max_score (Optional[float], optional: maximum reference score from the average returns of a hypothetical expert policy. This value is used in `MinariDataset.get_normalized_score()`. Default None.
        expert_policy (Optional[Callable[[ObsType], ActType], optional): policy to compute `ref_max_score` by averaging the returns over a number of episodes equal to  `num_episodes_average_score`.
                                                                        `ref_max_score` and `expert_policy` can't be passed at the same time. Default to None
        num_episodes_average_score (int): number of episodes to average over the returns to compute `ref_min_score` and `ref_max_score`. Default to 100.

    Returns:
        MinariDataset
    """
    # NoneType warnings
    if code_permalink is None:
        warnings.warn(
            "`code_permalink` is set to None. For reproducibility purposes it is highly recommended to link your dataset to versioned code.",
            UserWarning,
        )
    if author is None:
        warnings.warn(
            "`author` is set to None. For longevity purposes it is highly recommended to provide an author name.",
            UserWarning,
        )
    if author_email is None:
        warnings.warn(
            "`author_email` is set to None. For longevity purposes it is highly recommended to provide an author email, or some other obvious contact information.",
            UserWarning,
        )

    if observation_space is None:
        observation_space = env.observation_space
    if action_space is None:
        action_space = env.action_space

    if expert_policy is not None and ref_max_score is not None:
        raise ValueError(
            "Can't pass a value for `expert_policy` and `ref_max_score` at the same time."
        )

    dataset_path = get_dataset_path(dataset_id)

    # Check if dataset already exists
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(dataset_path, "data")
        os.makedirs(dataset_path)
        data_path = os.path.join(dataset_path, "main_data.hdf5")

        total_steps = 0
        with h5py.File(data_path, "w", track_order=True) as file:
            for i, eps_buff in enumerate(buffer):
                # check episode terminated or truncated
                assert (
                    eps_buff["terminations"][-1] or eps_buff["truncations"][-1]
                ), "Each episode must be terminated or truncated before adding it to a Minari dataset"
                assert len(eps_buff["actions"]) + 1 == len(
                    eps_buff["observations"]
                ), f"Number of observations {len(eps_buff['observations'])} must have an additional \
                                                                                        element compared to the number of action steps {len(eps_buff['actions'])} \
                                                                                        The initial and final observation must be included"
                seed = eps_buff.pop("seed", None)
                eps_group = clear_episode_buffer(
                    eps_buff, file.create_group(f"episode_{i}")
                )

                eps_group.attrs["id"] = i
                total_steps = len(eps_buff["actions"])
                eps_group.attrs["total_steps"] = total_steps
                total_steps += total_steps

                if seed is None:
                    eps_group.attrs["seed"] = str(None)
                else:
                    assert isinstance(seed, int)
                    eps_group.attrs["seed"] = seed

                # TODO: save EpisodeMetadataCallback callback in MinariDataset and update new episode group metadata

            file.attrs["total_episodes"] = len(buffer)
            file.attrs["total_steps"] = total_steps

            file.attrs[
                "env_spec"
            ] = env.spec.to_json()  # pyright: ignore [reportOptionalMemberAccess]
            file.attrs["dataset_id"] = dataset_id

            action_space_str = serialize_space(action_space)
            observation_space_str = serialize_space(observation_space)

            file.attrs["action_space"] = action_space_str
            file.attrs["observation_space"] = observation_space_str

            if expert_policy is not None or ref_max_score is not None:
                env = copy.deepcopy(env)
                if ref_min_score is None:
                    ref_min_score = get_average_reference_score(
                        env, RandomPolicy(env), num_episodes_average_score
                    )

                if expert_policy is not None:
                    ref_max_score = get_average_reference_score(
                        env, expert_policy, num_episodes_average_score
                    )

                file.attrs["ref_max_score"] = ref_max_score
                file.attrs["ref_min_score"] = ref_min_score
                file.attrs["num_episodes_average_score"] = num_episodes_average_score

        return MinariDataset(data_path)
    else:
        raise ValueError(
            f"A Minari dataset with ID {dataset_id} already exists and it cannot be overridden. Please use a different dataset name or version."
        )


def create_dataset_from_collector_env(
    dataset_id: str,
    collector_env: DataCollectorV0,
    algorithm_name: Optional[str] = None,
    author: Optional[str] = None,
    author_email: Optional[str] = None,
    code_permalink: Optional[str] = None,
    ref_min_score: Optional[float] = None,
    ref_max_score: Optional[float] = None,
    expert_policy: Optional[Callable[[ObsType], ActType]] = None,
    num_episodes_average_score: int = 100,
):
    """Create a Minari dataset using the data collected from stepping with a Gymnasium environment wrapped with a `DataCollectorV0` Minari wrapper.

    The ``dataset_id`` parameter corresponds to the name of the dataset, with the syntax as follows:
    ``(env_name-)(dataset_name)(-v(version))`` where ``env_name`` identifies the name of the environment used to generate the dataset ``dataset_name``.
    This ``dataset_id`` is used to load the Minari datasets with :meth:`minari.load_dataset`.

    Args:
        dataset_id (str): name id to identify Minari dataset
        collector_env (DataCollectorV0): Gymnasium environment used to collect the buffer data
        buffer (list[Dict[str, Union[list, Dict]]]): list of episode dictionaries with data
        algorithm_name (Optional[str], optional): name of the algorithm used to collect the data. Defaults to None.
        author (Optional[str], optional): author that generated the dataset. Defaults to None.
        author_email (Optional[str], optional): email of the author that generated the dataset. Defaults to None.
        code_permalink (Optional[str], optional): link to relevant code used to generate the dataset. Defaults to None.
        ref_min_score( Optional[float], optional): minimum reference score from the average returns of a random policy. This value is later used to normalize a score with :meth:`minari.get_normalized_score`. If default None the value will be estimated with a default random policy.
        ref_max_score (Optional[float], optional: maximum reference score from the average returns of a hypothetical expert policy. This value is used in :meth:`minari.get_normalized_score`. Default None.
        expert_policy (Optional[Callable[[ObsType], ActType], optional): policy to compute `ref_max_score` by averaging the returns over a number of episodes equal to  `num_episodes_average_score`.
                                                                        `ref_max_score` and `expert_policy` can't be passed at the same time. Default to None
        num_episodes_average_score (int): number of episodes to average over the returns to compute `ref_min_score` and `ref_max_score`. Default to 100.

    Returns:
        MinariDataset
    """
    # NoneType warnings
    if code_permalink is None:
        warnings.warn(
            "`code_permalink` is set to None. For reproducibility purposes it is highly recommended to link your dataset to versioned code.",
            UserWarning,
        )
    if author is None:
        warnings.warn(
            "`author` is set to None. For longevity purposes it is highly recommended to provide an author name.",
            UserWarning,
        )
    if author_email is None:
        warnings.warn(
            "`author_email` is set to None. For longevity purposes it is highly recommended to provide an author email, or some other obvious contact information.",
            UserWarning,
        )
    if expert_policy is not None and ref_max_score is not None:
        raise ValueError(
            "Can't pass a value for `expert_policy` and `ref_max_score` at the same time."
        )

    assert collector_env.datasets_path is not None
    dataset_path = os.path.join(collector_env.datasets_path, dataset_id)

    # Check if dataset already exists
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(dataset_path, "data")
        os.makedirs(dataset_path)
        data_path = os.path.join(dataset_path, "main_data.hdf5")
        dataset_metadata: Dict[str, Any] = {
            "dataset_id": str(dataset_id),
            "algorithm_name": str(algorithm_name),
            "author": str(author),
            "author_email": str(author_email),
            "code_permalink": str(code_permalink),
        }

        if expert_policy is not None or ref_max_score is not None:
            env = copy.deepcopy(collector_env.env)
            if ref_min_score is None:
                ref_min_score = get_average_reference_score(
                    env, RandomPolicy(env), num_episodes_average_score
                )

            if expert_policy is not None:
                ref_max_score = get_average_reference_score(
                    env, expert_policy, num_episodes_average_score
                )
            dataset_metadata.update(
                {
                    "ref_max_score": ref_max_score,
                    "ref_min_score": ref_min_score,
                    "num_episodes_average_score": num_episodes_average_score,
                }
            )

        collector_env.save_to_disk(
            data_path,
            dataset_metadata=dataset_metadata,
        )
        return MinariDataset(data_path)
    else:
        raise ValueError(
            f"A Minari dataset with ID {dataset_id} already exists and it cannot be overridden. Please use a different dataset name or version."
        )


def get_normalized_score(
    dataset: MinariDataset, returns: Union[float, np.float32]
) -> Union[float, np.float32]:
    r"""Normalize undiscounted return of an episode.

    This function was originally provided in the `D4RL repository <https://github.com/Farama-Foundation/D4RL/blob/71a9549f2091accff93eeff68f1f3ab2c0e0a288/d4rl/offline_env.py#L71>`_.
    The computed normalized episode return (normalized score) facilitates the comparison of algorithm performance across different tasks. The returned normalized score will be in a range between 0 and 1.
    Where 0 corresponds to the minimum reference score calculated as the average of episode returns collected from a random policy in the environment, and 1 corresponds to a maximum reference score computed as
    the average of episode returns from an hypothetical expert policy. These two values are stored as optional attributes in a MinariDataset as `ref_min_score` and `ref_max_score` respectively.

    The formula to normalize an episode return is:

    .. math:: normalize\_score = \frac{return - ref\_min\_score}{ref\_max\_score - ref\_min\_score}

    .. warning:: This utility function is under testing and will not be available in every Minari dataset. For now, only the datasets imported from D4RL will contain the `ref_min_score` and `ref_max_score` attributes.

    Args:
        dataset (MinariDataset): the MinariDataset with respect to which normalize the score. Must contain the reference score attributes `ref_min_score` and `ref_max_score`.
        returns (float | np.float32): a single value or array of episode undiscounted returns to normalize.

    Returns:
        normalized_scores
    """
    with h5py.File(dataset.spec.data_path, "r") as f:
        ref_min_score = f.attrs.get("ref_min_score", default=None)
        ref_max_score = f.attrs.get("ref_max_score", default=None)
    if ref_min_score is None or ref_max_score is None:
        raise ValueError(
            f"Reference score not provided for dataset {dataset.spec.dataset_id}. Can't compute the normalized score."
        )

    assert isinstance(ref_min_score, float)
    assert isinstance(ref_max_score, float)

    return (returns - ref_min_score) / (ref_max_score - ref_min_score)

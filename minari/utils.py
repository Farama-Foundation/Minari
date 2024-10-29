from __future__ import annotations

import copy
import importlib.metadata
import os
import re
import warnings
from typing import Any, Callable, Dict, Iterable, List, Optional

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType
from gymnasium.envs.registration import EnvSpec
from gymnasium.wrappers import RecordEpisodeStatistics  # type: ignore

from minari.data_collector.episode_buffer import EpisodeBuffer
from minari.dataset.minari_dataset import MinariDataset
from minari.dataset.minari_storage import MinariStorage
from minari.serialization import deserialize_space
from minari.storage.datasets_root_dir import get_dataset_path


# Use importlib due to circular import when: "from minari import __version__"
__version__ = importlib.metadata.version("minari")


def validate_datasets_to_combine(
    datasets_to_combine: List[MinariDataset],
) -> EnvSpec | None:
    """Check if the given datasets can be combined.

    Tests if the datasets were created with the same environment (`env_spec`) and re-calculates the
    `max_episode_steps` argument.

    Also checks that the datasets obs/act spaces are the same.

    Args:
        datasets_to_combine (List[MinariDataset]): list of MinariDataset to combine

    Returns:
        combined_dataset_env_spec (EnvSpec): the resulting EnvSpec of combining the MinariDatasets

    """
    # get first among the dataset's env_spec which is not None
    first_not_none_env_spec = next(
        (
            dataset.spec.env_spec
            for dataset in datasets_to_combine
            if dataset.spec.env_spec is not None
        ),
        None,
    )

    # early return where all datasets have no env_spec
    if first_not_none_env_spec is None:
        return None

    common_env_spec = copy.deepcopy(first_not_none_env_spec)

    # updating the common_env_spec's max_episode_steps & checking equivalence of all env specs
    for dataset in datasets_to_combine:
        assert isinstance(dataset, MinariDataset)
        env_spec = dataset.spec.env_spec
        if env_spec is not None:
            if (
                common_env_spec.max_episode_steps is None
                or env_spec.max_episode_steps is None
            ):
                common_env_spec.max_episode_steps = None
            else:
                common_env_spec.max_episode_steps = max(
                    common_env_spec.max_episode_steps, env_spec.max_episode_steps
                )
            # setting max_episode_steps in object's copy to same value for sake of checking equality
            env_spec_copy = copy.deepcopy(env_spec)
            env_spec_copy.max_episode_steps = common_env_spec.max_episode_steps
            if env_spec_copy != common_env_spec:
                raise ValueError(
                    "The datasets to be combined have different values for `env_spec` attribute."
                )
        else:
            raise ValueError(
                "Cannot combine datasets having env_spec with those having no env_spec."
            )

    return common_env_spec


class RandomPolicy:
    """A random action selection policy to compute `ref_min_score`."""

    def __init__(self, env: gym.Env):
        self.action_space = env.action_space
        self.action_space.seed(123)
        self.observation_space = env.observation_space

    def __call__(self, observation: ObsType) -> ActType:
        assert self.observation_space.contains(observation)
        return self.action_space.sample()


def combine_datasets(datasets_to_combine: List[MinariDataset], new_dataset_id: str):
    """Combine a group of MinariDataset in to a single dataset with its own name id.

    The new dataset will contain a metadata attribute `combined_datasets` containing a list
    with the dataset names that were combined to form this new Minari dataset.

    Args:
        datasets_to_combine (list[MinariDataset]): list of datasets to be combined
        new_dataset_id (str): name id for the newly created dataset

    Returns:
        combined_dataset (MinariDataset): the resulting MinariDataset
    """
    combined_dataset_env_spec = validate_datasets_to_combine(datasets_to_combine)

    new_dataset_path = get_dataset_path(new_dataset_id)
    new_dataset_path.mkdir()
    new_storage = MinariStorage.new(
        new_dataset_path.joinpath("data"),
        env_spec=combined_dataset_env_spec,
        observation_space=datasets_to_combine[0].observation_space,
        action_space=datasets_to_combine[0].action_space,
        data_format=datasets_to_combine[0].storage.FORMAT,
    )

    new_storage.update_metadata(
        {
            "dataset_id": new_dataset_id,
            "combined_datasets": [
                dataset.spec.dataset_id for dataset in datasets_to_combine
            ],
            "minari_version": __version__,
        }
    )

    for dataset in datasets_to_combine:
        new_storage.update_from_storage(dataset.storage)

    return MinariDataset(new_storage)


def split_dataset(
    dataset: MinariDataset, sizes: List[int], seed: Optional[int] = None
) -> List[MinariDataset]:
    """Split a MinariDataset in multiple datasets.

    Args:
        dataset (MinariDataset): the MinariDataset to split
        sizes (List[int]): sizes of the resulting datasets
        seed (Optional[int]): random seed

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
            obs, _, terminated, truncated, info = env.step(
                action  # pyright: ignore[reportGeneralTypeIssues]
            )
            if terminated or truncated:
                episode_returns.append(info["episode"]["r"])
                obs, _ = env.reset()
                break

    mean_ref_score = np.mean(episode_returns, dtype=np.float32)
    return float(mean_ref_score)


def _generate_dataset_path(dataset_id: str) -> str | os.PathLike:
    """Checks if the dataset already exists locally, then create and return the data storage directory."""
    dataset_path = get_dataset_path(dataset_id)
    if os.path.exists(dataset_path):
        raise ValueError(
            f"A Minari dataset with ID {dataset_id} already exists and it cannot be overridden. Please use a different dataset name or version."
        )

    dataset_path = os.path.join(dataset_path, "data")
    os.makedirs(dataset_path)

    return dataset_path


def _generate_dataset_metadata(
    dataset_id: str,
    env_spec: Optional[EnvSpec],
    eval_env: Optional[str | gym.Env | EnvSpec],
    algorithm_name: Optional[str],
    author: Optional[str | set[str]],
    author_email: Optional[str | set[str]],
    code_permalink: Optional[str],
    ref_min_score: Optional[float],
    ref_max_score: Optional[float],
    expert_policy: Optional[Callable[[ObsType], ActType]],
    num_episodes_average_score: int,
    description: Optional[str],
    requirements: Optional[list],
) -> Dict[str, Any]:
    """Return the metadata dictionary of the dataset."""
    dataset_metadata: Dict[str, Any] = {
        "dataset_id": dataset_id,
    }
    # NoneType warnings
    if code_permalink is None:
        warnings.warn(
            "`code_permalink` is set to None. For reproducibility purposes it is highly recommended to link your dataset to versioned code.",
            UserWarning,
        )
    else:
        dataset_metadata["code_permalink"] = code_permalink

    if author is None:
        warnings.warn(
            "`author` is set to None. For longevity purposes it is highly recommended to provide an author name.",
            UserWarning,
        )
    else:
        dataset_metadata["author"] = {author} if isinstance(author, str) else author

    if author_email is None:
        warnings.warn(
            "`author_email` is set to None. For longevity purposes it is highly recommended to provide an author email, or some other obvious contact information.",
            UserWarning,
        )
    else:
        dataset_metadata["author_email"] = (
            {author_email} if isinstance(author_email, str) else author_email
        )

    if algorithm_name is None:
        warnings.warn(
            "`algorithm_name` is set to None. For reproducibility purpose it's highly recommended to set your algorithm",
            UserWarning,
        )
    else:
        dataset_metadata["algorithm_name"] = algorithm_name

    if description is None:
        warnings.warn(
            "`description` is set to None. For longevity purposes it is highly recommended to provide a description of the dataset",
            UserWarning,
        )
    else:
        dataset_metadata["description"] = description

    dataset_metadata["minari_version"] = __version__

    if expert_policy is not None and ref_max_score is not None:
        raise ValueError(
            "Can't pass a value for `expert_policy` and `ref_max_score` at the same time."
        )

    if eval_env is None:
        warnings.warn(
            f"`eval_env` is set to None. If another environment is intended to be used for evaluation please specify corresponding Gymnasium environment (gym.Env | gym.envs.registration.EnvSpec). "
            f"If None the environment used to collect the data (`env={env_spec}`) will be used for this purpose.",
            UserWarning,
        )
        eval_env_spec = env_spec
    else:
        if isinstance(eval_env, str):
            eval_env_spec = gym.spec(eval_env)
        elif isinstance(eval_env, EnvSpec):
            eval_env_spec = eval_env
        elif isinstance(eval_env, gym.Env):
            eval_env_spec = eval_env.spec
        else:
            raise ValueError(
                "The `eval_env` argument must be of types str|EnvSpec|gym.Env"
            )
        assert eval_env_spec is not None
        dataset_metadata["eval_env_spec"] = eval_env_spec.to_json()

    if env_spec is None:
        warnings.warn(
            "env_spec is None, no environment spec is provided during collection for this dataset",
            UserWarning,
        )

    if eval_env_spec is not None and (
        expert_policy is not None or ref_max_score is not None
    ):
        env_ref_score = gym.make(eval_env_spec)
        if ref_min_score is None:
            ref_min_score = get_average_reference_score(
                env_ref_score, RandomPolicy(env_ref_score), num_episodes_average_score
            )

        if expert_policy is not None:
            ref_max_score = get_average_reference_score(
                env_ref_score, expert_policy, num_episodes_average_score
            )
        dataset_metadata["ref_max_score"] = ref_max_score
        dataset_metadata["ref_min_score"] = ref_min_score
        dataset_metadata["num_episodes_average_score"] = num_episodes_average_score

    if requirements is not None:
        dataset_metadata["requirements"] = requirements
    return dataset_metadata


def create_dataset_from_buffers(
    dataset_id: str,
    buffer: List[EpisodeBuffer],
    env: Optional[str | gym.Env | EnvSpec] = None,
    eval_env: Optional[str | gym.Env | EnvSpec] = None,
    algorithm_name: Optional[str] = None,
    author: Optional[str | set[str]] = None,
    author_email: Optional[str | set[str]] = None,
    code_permalink: Optional[str] = None,
    action_space: Optional[gym.spaces.Space] = None,
    observation_space: Optional[gym.spaces.Space] = None,
    ref_min_score: Optional[float] = None,
    ref_max_score: Optional[float] = None,
    expert_policy: Optional[Callable[[ObsType], ActType]] = None,
    num_episodes_average_score: int = 100,
    description: Optional[str] = None,
    data_format: Optional[str] = None,
    requirements: Optional[list] = None,
):
    """Create Minari dataset from a list of episode dictionary buffers.

    The ``dataset_id`` parameter corresponds to the name of the dataset, with the syntax as follows:
    ``(namespace/)(env_name/)dataset_name(-v[version])`` where ``env_name`` identifies, if present, the name of the environment used to generate the dataset ``dataset_name`` and ``namespace`` optionally groups datasets together.
    This ``dataset_id`` is used to load the Minari datasets with :meth:`minari.load_dataset`.

    Args:
        dataset_id (str): name id to identify Minari dataset.
        buffer (list[EpisodeBuffer]): list of episode buffer with data.
        env (str | gym.Env | EnvSpec, optional): Gymnasium environment(gym.Env)/environment id(str)/environment spec(EnvSpec) used to collect the buffer data. Defaults to None.
        eval_env (str | gym.Env | EnvSpec, optional): Gymnasium environment(gym.Env)/environment id(str)/environment spec(EnvSpec) to use for evaluation with the dataset. After loading the dataset, the environment can be recovered as follows: `MinariDataset.recover_environment(eval_env=True).
                                                If None, and if the `env` used to collect the buffer data is available, latter will be used for evaluation.
        algorithm_name (str, optional): name of the algorithm used to collect the data. Defaults to None.
        author (str | set, optional): name of the author(s) that generated the dataset. Defaults to None.
        author_email (str | set, optional): email(s) of the author(s) that generated the dataset. Defaults to None.
        code_permalink (str, optional): link to relevant code used to generate the dataset. Defaults to None.
        ref_min_score (float, optional): minimum reference score from the average returns of a random policy. This value is later used to normalize a score with :meth:`minari.get_normalized_score`. If default None the value will be estimated with a default random policy.
                                                    Also note that this attribute will be added to the Minari dataset only if `ref_max_score` or `expert_policy` are assigned a valid value other than None.
        ref_max_score (float, optional): maximum reference score from the average returns of a hypothetical expert policy. This value is used in `MinariDataset.get_normalized_score()`. Default None.
        expert_policy (Callable[[ObsType], ActType], optional): policy to compute `ref_max_score` by averaging the returns over a number of episodes equal to  `num_episodes_average_score`.
                                                                        `ref_max_score` and `expert_policy` can't be passed at the same time. Default to None
        num_episodes_average_score (int): number of episodes to average over the returns to compute `ref_min_score` and `ref_max_score`. Default to 100.
                observation_space:
        action_space (gym.spaces.Space, optional): action space of the environment. If None (default) use the environment action space.
        observation_space (gym.spaces.Space, optional): observation space of the environment. If None (default) use the environment observation space.
        description (str, optional): description of the dataset being created. Defaults to None.
        data_format (str, optional): Data format to store the data in the Minari dataset. If None (defaults), it will use the default format of MinariStorage.
        requirements (list of str, optional): list of requirements in pip-style to load the environment and reproduce the dataset. For example, `mujoco>=3.1.0,<3.2.0`, which indicate the supported version range for mujoco package. Defaults to None.

    Returns:
        MinariDataset
    """
    dataset_path = _generate_dataset_path(dataset_id)

    if env is None:
        env_spec = None
        if observation_space is None or action_space is None:
            raise ValueError(
                "Both observation space and action space must be provided, if env is None"
            )
    else:
        if isinstance(env, EnvSpec):
            env_spec = env
        elif isinstance(env, str):
            env_spec = gym.spec(env)
        elif isinstance(env, gym.Env):
            env_spec = env.spec
        else:
            raise TypeError("Unsupported env type")

        gym_env: gym.Env = gym.make(env) if isinstance(env, (str, EnvSpec)) else env
        observation_space = observation_space or gym_env.observation_space
        action_space = action_space or gym_env.action_space

    metadata = _generate_dataset_metadata(
        dataset_id,
        env_spec,
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

    data_format_kwarg = {"data_format": data_format} if data_format is not None else {}
    storage = MinariStorage.new(
        dataset_path,
        observation_space=observation_space,
        action_space=action_space,
        env_spec=env_spec,
        **data_format_kwarg,
    )

    storage.update_metadata(metadata)
    storage.update_episodes(buffer)
    return MinariDataset(storage)


def get_normalized_score(dataset: MinariDataset, returns: np.ndarray) -> np.ndarray:
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
        returns (np.ndarray): a single value or array of episode undiscounted returns to normalize.

    Returns:
        normalized_scores
    """
    ref_min_score = dataset.storage.metadata.get("ref_min_score")
    ref_max_score = dataset.storage.metadata.get("ref_max_score")

    if ref_min_score is None or ref_max_score is None:
        raise ValueError(
            f"Reference score not provided for dataset {dataset.spec.dataset_id}. Can't compute the normalized score."
        )

    return (returns - ref_min_score) / (ref_max_score - ref_min_score)


def get_env_spec_dict(env_spec: EnvSpec) -> Dict[str, str]:
    """Create dict of the environment specs, including observation and action space."""
    try:
        env = gym.make(env_spec)
        action_space_table = env.action_space.__repr__().replace("\n", "")
        observation_space_table = env.observation_space.__repr__().replace("\n", "")
    except Exception as e:
        warnings.warn(f"Failed to make env {env_spec.id}, {e}")
        action_space_table, observation_space_table = None, None

    md_dict = {"ID": env_spec.id}
    if observation_space_table is not None:
        md_dict["Observation Space"] = f"`{re.sub(' +', ' ', observation_space_table)}`"
    if action_space_table is not None:
        md_dict["Action Space"] = f"`{re.sub(' +', ' ', action_space_table)}`"

    md_dict.update(
        {
            "entry_point": f"`{env_spec.entry_point}`",
            "max_episode_steps": str(env_spec.max_episode_steps),
            "reward_threshold": str(env_spec.reward_threshold),
            "nondeterministic": f"`{env_spec.nondeterministic}`",
            "order_enforce": f"`{env_spec.order_enforce}`",
            "disable_env_checker": f"`{env_spec.disable_env_checker}`",
            "kwargs": f"`{env_spec.kwargs}`",
            "additional_wrappers": f"`{env_spec.additional_wrappers}`",
            "vector_entry_point": f"`{env_spec.vector_entry_point}`",
        }
    )

    return {k: str(v) for k, v in md_dict.items()}


def get_dataset_spec_dict(dataset_spec: Dict) -> Dict[str, str]:
    """Create dict of the dataset specs, including observation and action space."""
    code_link = dataset_spec.get("code_permalink")
    action_space = dataset_spec.get("action_space")
    obs_space = dataset_spec.get("observation_space")

    md_dict = {
        "Total Steps": str(dataset_spec["total_steps"]),
        "Total Episodes": str(dataset_spec["total_episodes"]),
    }

    if obs_space is not None:
        if isinstance(obs_space, (dict, str)):
            obs_space = deserialize_space(obs_space)
        dataset_observation_space = obs_space.__repr__().replace("\n", "")
        md_dict["Dataset Observation Space"] = f"`{dataset_observation_space}`"

    if action_space is not None:
        if isinstance(action_space, (dict, str)):
            action_space = deserialize_space(action_space)
        dataset_action_space = action_space.__repr__().replace("\n", "")
        md_dict["Dataset Action Space"] = f"`{dataset_action_space}`"

    from minari import supported_dataset_versions

    version = dataset_spec["minari_version"]
    supported = (
        "supported" if version in supported_dataset_versions else "not supported"
    )
    author = dataset_spec.get("author", "Not provided")
    if not isinstance(author, str) and isinstance(author, Iterable):
        author = ", ".join(author)
    email = dataset_spec.get("author_email", "Not provided")
    if not isinstance(email, str) and isinstance(email, Iterable):
        email = ", ".join(email)
    assert isinstance(author, str)
    assert isinstance(email, str)
    md_dict.update(
        {
            "Algorithm": dataset_spec.get("algorithm_name", "Not provided"),
            "Author": author,
            "Email": email,
            "Code Permalink": f"[{code_link}]({code_link})",
            "Minari Version": f"`{version}` ({supported})",
            "Download": f"`minari download {dataset_spec['dataset_id']}`",
        }
    )

    return md_dict

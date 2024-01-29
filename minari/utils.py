from __future__ import annotations

import copy
import importlib.metadata
import os
import re
import warnings
from typing import Any, Callable, Dict, List, Optional, Union

import gymnasium as gym
import numpy as np
import portion as P
from gymnasium.core import ActType, ObsType
from gymnasium.envs.registration import EnvSpec
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import Version

from minari import DataCollector
from minari.dataset.minari_dataset import MinariDataset
from minari.dataset.minari_storage import MinariStorage
from minari.serialization import deserialize_space
from minari.storage.datasets_root_dir import get_dataset_path


# Use importlib due to circular import when: "from minari import __version__"
__version__ = importlib.metadata.version("minari")


def combine_minari_version_specifiers(specifier_set: SpecifierSet) -> SpecifierSet:
    """Calculates the Minari version specifier by intersecting a group of Minari version specifiers.

    Used to calculate the `minari_version` metadata attribute when combining multiple datasets. The function
    assumes that all the given version specifiers at least contain the current minari version thus the version
    specifier sets will always intersect. Also, it doesn't take into account any pre-releases since Farama projects
    don't use them for versioning.

    The supported version specifier operators are those in PEP 440(https://peps.python.org/pep-0440/#version-specifiers),
    except for '==='.

    Args:
        specifier_set (SpecifierSet): set of all version specifiers to intersect

    Returns:
        version_specifier (SpecifierSet): resulting version specifier

    """
    specifiers = sorted(specifier_set, key=str)

    exclusion_specifiers = filter(lambda spec: spec.operator == "!=", specifiers)
    inclusion_specifiers = filter(lambda spec: spec.operator != "!=", specifiers)

    inclusion_interval = P.closed(-P.inf, P.inf)

    # Intersect the version intervals compatible with the datasets
    for spec in inclusion_specifiers:
        operator = spec.operator
        version = spec.version
        if operator[0] == ">":
            if operator[-1] == "=":
                inclusion_interval &= P.closedopen(Version(version), P.inf)
            else:
                inclusion_interval &= P.open(Version(version), P.inf)
        elif operator[0] == "<":
            if operator[-1] == "=":
                inclusion_interval &= P.openclosed(-P.inf, Version(version))
            else:
                inclusion_interval &= P.open(-P.inf, Version(version))
        elif operator == "==":
            if version[-1] == "*":
                version = Version(version[:-2])
                release = list(version.release)
                release[-1] = 0
                release[-2] += 1
                max_release = Version(".".join(str(r) for r in release))
                inclusion_interval &= P.closedopen(version, max_release)
            else:
                inclusion_interval &= P.singleton(Version(version))
        elif operator == "~=":
            release = list(Version(version).release)
            release[-1] = 0
            release[-2] += 1
            max_release = Version(".".join(str(r) for r in release))
            inclusion_interval &= P.closedopen(Version(version), max_release)

    # Convert the intersection of version intervals to a version specifier
    final_version_specifier = SpecifierSet()

    if inclusion_interval.lower == inclusion_interval.upper:
        assert inclusion_interval.lower == Version(
            __version__
        ), f"The local installed version of Minari, {__version__}, must comply with the equality version specifier: =={inclusion_interval.lower}"
        final_version_specifier &= f"=={inclusion_interval.lower}"
        # There is just one compatible version of Minari
        return final_version_specifier

    if inclusion_interval.lower != -P.inf and inclusion_interval.upper != P.inf:
        lower_version = Version(str(inclusion_interval.lower))
        next_release = list(lower_version.release)
        next_release[-1] = 0
        next_release[-2] += 1
        next_release = Version(".".join(str(r) for r in next_release))
        upper_version = Version(str(inclusion_interval.upper))
        if (
            inclusion_interval.left == P.CLOSED
            and inclusion_interval.left == P.CLOSED
            and upper_version == next_release
        ):
            final_version_specifier &= f"~={str(lower_version)}"
        else:
            if inclusion_interval.left == P.CLOSED:
                operator = ">="
            else:
                operator = ">"
            final_version_specifier &= f"{operator}{str(inclusion_interval.lower)}"

            if inclusion_interval.right == P.CLOSED:
                operator = "<="
            else:
                operator = "<"
            final_version_specifier &= f"{operator}{str(inclusion_interval.upper)}"
    else:
        if inclusion_interval.lower != -P.inf:
            if inclusion_interval.left == P.CLOSED:
                operator = ">="
            else:
                operator = ">"
            final_version_specifier &= f"{operator}{str(inclusion_interval.lower)}"
        if inclusion_interval.upper != P.inf:
            if inclusion_interval.right == P.CLOSED:
                operator = "<="
            else:
                operator = "<"
            final_version_specifier &= f"{operator}{str(inclusion_interval.upper)}"

    # If the versions to be excluded fall inside the previous calculated version specifier
    # add them to the specifier as `!=`
    for spec in exclusion_specifiers:
        version = spec.version
        if version[-1] == "*":
            version = Version(version[:-2])
            release = list(version.release)
            release[-1] = 0
            release[-2] += 1
            max_release = Version(".".join(str(r) for r in release))
            exclusion_interval = P.closedopen(version, max_release)
            if inclusion_interval.overlaps(exclusion_interval):
                final_version_specifier &= str(spec)
        elif version in final_version_specifier:
            final_version_specifier &= str(spec)

    return final_version_specifier


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
    first_not_none_env_spec = next((dataset.spec.env_spec for dataset in datasets_to_combine if dataset.spec.env_spec is not None), None)

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
            raise ValueError("Cannot combine datasets having env_spec with those having no env_spec.")

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

    # Compute intersection of Minari version specifiers
    datasets_minari_version_specifiers = SpecifierSet()
    for dataset in datasets_to_combine:
        datasets_minari_version_specifiers &= dataset.spec.minari_version

    minari_version_specifier = combine_minari_version_specifiers(
        datasets_minari_version_specifiers
    )

    new_dataset_path = get_dataset_path(new_dataset_id)
    new_dataset_path.mkdir()
    new_storage = MinariStorage.new(
        new_dataset_path.joinpath("data"), env_spec=combined_dataset_env_spec,
        observation_space=datasets_to_combine[0].observation_space,
        action_space=datasets_to_combine[0].action_space
    )

    new_storage.update_metadata(
        {
            "dataset_id": new_dataset_id,
            "combined_datasets": [
                dataset.spec.dataset_id for dataset in datasets_to_combine
            ],
            "minari_version": str(minari_version_specifier),
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
            obs, _, terminated, truncated, info = env.step(action)  # pyright: ignore[reportGeneralTypeIssues]
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
    author: Optional[str],
    author_email: Optional[str],
    code_permalink: Optional[str],
    ref_min_score: Optional[float],
    ref_max_score: Optional[float],
    expert_policy: Optional[Callable[[ObsType], ActType]],
    num_episodes_average_score: int,
    minari_version: Optional[str],
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
        dataset_metadata["author"] = author

    if author_email is None:
        warnings.warn(
            "`author_email` is set to None. For longevity purposes it is highly recommended to provide an author email, or some other obvious contact information.",
            UserWarning,
        )
    else:
        dataset_metadata["author_email"] = author_email

    if algorithm_name is None:
        warnings.warn(
            "`algorithm_name` is set to None. For reproducibility purpose it's highly recommended to set your algorithm",
            UserWarning,
        )
    else:
        dataset_metadata["algorithm_name"] = algorithm_name

    if minari_version is None:
        version = Version(__version__)
        release = version.release
        # For __version__ = X.Y.Z, by default compatibility with version X.Y or later, but not (X+1).0 or later.
        minari_version = f"~={'.'.join(str(x) for x in release[:2])}"
        warnings.warn(
            f"`minari_version` is set to None. The compatible dataset version specifier for Minari will be set to {minari_version}.",
            UserWarning,
        )
    # Check if the installed Minari version falls inside the minari_version specifier
    try:
        assert Version(__version__) in SpecifierSet(
            minari_version
        ), f"The installed Minari version {__version__} is not contained in the dataset version specifier {minari_version}."
    except InvalidSpecifier:
        print(f"{minari_version} is not a version specifier.")

    dataset_metadata["minari_version"] = minari_version

    if expert_policy is not None and ref_max_score is not None:
        raise ValueError(
            "Can't pass a value for `expert_policy` and `ref_max_score` at the same time."
        )

    if eval_env is None:
        warnings.warn(
            f"`eval_env` is set to None. If another environment is intended to be used for evaluation please specify corresponding Gymnasium environment (gym.Env | gym.envs.registration.EnvSpec).\
              If None the environment used to collect the data (`env={env_spec}`) will be used for this purpose.",
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

    if eval_env_spec is not None and (expert_policy is not None or ref_max_score is not None):
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

    return dataset_metadata


def create_dataset_from_buffers(
    dataset_id: str,
    buffer: List[Dict[str, Union[list, Dict]]],
    env: Optional[str | gym.Env | EnvSpec] = None,
    eval_env: Optional[str | gym.Env | EnvSpec] = None,
    algorithm_name: Optional[str] = None,
    author: Optional[str] = None,
    author_email: Optional[str] = None,
    code_permalink: Optional[str] = None,
    minari_version: Optional[str] = None,
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
        * `actions`: np.ndarray of step action. shape = (total_episode_steps, (action_shape)).
        * `rewards`: np.ndarray of step rewards. shape = (total_episode_steps, 1).
        * `terminations`: np.ndarray of step terminations. shape = (total_episode_steps, 1).
        * `truncations`: np.ndarray of step truncations. shape = (total_episode_steps, 1).

    Other additional items can be added as long as the values are np.ndarray's or other nested dictionaries.

    Args:
        dataset_id (str): name id to identify Minari dataset.
        buffer (list[Dict[str, Union[list, Dict]]]): list of episode dictionaries with data.
        env (Optional[str|gym.Env|EnvSpec]): Gymnasium environment(gym.Env)/environment id(str)/environment spec(EnvSpec) used to collect the buffer data. Defaults to None.
        eval_env (Optional[str|gym.Env|EnvSpec]): Gymnasium environment(gym.Env)/environment id(str)/environment spec(EnvSpec) to use for evaluation with the dataset. After loading the dataset, the environment can be recovered as follows: `MinariDataset.recover_environment(eval_env=True).
                                                If None, and if the `env` used to collect the buffer data is available, latter will be used for evaluation.
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
                observation_space:
        action_space (Optional[gym.spaces.Space]): action space of the environment. If None (default) use the environment action space.
        observation_space (Optional[gym.spaces.Space]): observation space of the environment. If None (default) use the environment observation space.
        minari_version (Optional[str], optional): Minari version specifier compatible with the dataset. If None (default) use the installed Minari version.

    Returns:
        MinariDataset
    """
    dataset_path = _generate_dataset_path(dataset_id)

    if isinstance(env, str):
        env_spec = gym.spec(env)
    elif isinstance(env, EnvSpec):
        env_spec = env
    elif isinstance(env, gym.Env):
        env_spec = env.spec
    elif env is None:
        if observation_space is None or action_space is None:
            raise ValueError("Both observation space and action space must be provided, if env is None")
        env_spec = None
    else:
        raise ValueError("The `env` argument must be of types str|EnvSpec|gym.Env|None")

    if isinstance(env, (str, EnvSpec)):
        env = gym.make(env)
    if observation_space is None:
        assert isinstance(env, gym.Env)
        observation_space = env.observation_space
    if action_space is None:
        assert isinstance(env, gym.Env)
        action_space = env.action_space

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
        minari_version,
    )

    storage = MinariStorage.new(
        dataset_path,
        observation_space=observation_space,
        action_space=action_space,
        env_spec=env_spec,
    )

    # adding `update_metadata` before hand too, as for small envs, the absence of metadata is causing a difference of some 10ths of MBs leading to errors in unit tests.
    storage.update_metadata(metadata)
    storage.update_episodes(buffer)

    metadata['dataset_size'] = storage.get_size()
    storage.update_metadata(metadata)

    return MinariDataset(storage)


def create_dataset_from_collector_env(
    dataset_id: str,
    collector_env: DataCollector,
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
        collector_env (DataCollector): Gymnasium environment used to collect the buffer data
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
    warnings.warn("This function is deprecated and will be removed in v0.5.0. Please use DataCollector.create_dataset() instead.", DeprecationWarning, stacklevel=2)
    dataset = collector_env.create_dataset(
        dataset_id=dataset_id,
        eval_env=eval_env,
        algorithm_name=algorithm_name,
        author=author,
        author_email=author_email,
        code_permalink=code_permalink,
        ref_min_score=ref_min_score,
        ref_max_score=ref_max_score,
        expert_policy=expert_policy,
        num_episodes_average_score=num_episodes_average_score,
        minari_version=minari_version,
    )
    return dataset


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
    env = gym.make(env_spec.id)

    action_space_table = env.action_space.__repr__().replace("\n", "")
    observation_space_table = env.observation_space.__repr__().replace("\n", "")

    md_dict = {
        "ID": env_spec.id,
        "Observation Space": f"`{re.sub(' +', ' ', observation_space_table)}`",
        "Action Space": f"`{re.sub(' +', ' ', action_space_table)}`",
        "entry_point": f"`{env_spec.entry_point}`",
        "max_episode_steps": env_spec.max_episode_steps,
        "reward_threshold": env_spec.reward_threshold,
        "nondeterministic": f"`{env_spec.nondeterministic}`",
        "order_enforce": f"`{env_spec.order_enforce}`",
        "autoreset": f"`{env_spec.autoreset}`",
        "disable_env_checker": f"`{env_spec.disable_env_checker}`",
        "kwargs": f"`{env_spec.kwargs}`",
        "additional_wrappers": f"`{env_spec.additional_wrappers}`",
        "vector_entry_point": f"`{env_spec.vector_entry_point}`",
    }

    return {k: str(v) for k, v in md_dict.items()}


def get_dataset_spec_dict(
        dataset_spec: Union[Dict[str, Union[str, int, bool]], Dict[str, str]],
        print_version: bool = False
) -> Dict[str, str]:
    """Create dict of the dataset specs, including observation and action space."""
    code_link = dataset_spec["code_permalink"]
    action_space = dataset_spec["action_space"]
    obs_space = dataset_spec["observation_space"]

    assert isinstance(action_space, str)
    assert isinstance(obs_space, str)

    dataset_action_space = (
        deserialize_space(action_space).__repr__().replace("\n", "")
    )
    dataset_observation_space = (
        deserialize_space(obs_space)
        .__repr__()
        .replace("\n", "")
    )

    version = str(dataset_spec['minari_version'])

    if print_version:
        version += f" ({__version__} installed)"

    md_dict = {
        "Total Timesteps": dataset_spec["total_steps"],
        "Total Episodes": dataset_spec["total_episodes"],
        "Dataset Observation Space": f"`{dataset_observation_space}`",
        "Dataset Action Space": f"`{dataset_action_space}`",
        "Algorithm": dataset_spec["algorithm_name"],
        "Author": dataset_spec["author"],
        "Email": dataset_spec["author_email"],
        "Code Permalink": f"[{code_link}]({code_link})",
        "Minari Version": version,
        "Download": f"`minari.download_dataset(\"{dataset_spec['dataset_id']}\")`"
    }

    return md_dict

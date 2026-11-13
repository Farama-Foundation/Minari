from unittest.mock import patch

import gymnasium as gym
import numpy as np
import pytest

import minari
from minari import MinariDataset
from minari.data_collector import EpisodeBuffer


@pytest.fixture(params=["hdf5", "arrow"])
def source_dataset(request):
    space = gym.spaces.Box(0, 4, (1,), dtype=np.float32)
    buffers = []
    for index in range(4):
        value = np.array([index], dtype=np.float32)
        buffers.append(
            EpisodeBuffer(
                observations=[value.copy(), value.copy()],
                actions=[value.copy()],
                rewards=[float(index)],
                terminations=[True],
                truncations=[False],
            )
        )
    with pytest.warns(UserWarning, match="env_spec is None"):
        return minari.create_dataset_from_buffers(
            dataset_id="split-source-v0",
            buffer=buffers,
            observation_space=space,
            action_space=space,
            ref_min_score=0.0,
            ref_max_score=1.0,
            author="Minari tests",
            author_email="tests@example.com",
            algorithm_name="synthetic",
            code_permalink="https://example.com/tests",
            description="Synthetic episodes for split validation.",
            data_format=request.param,
        )


@pytest.mark.parametrize("sizes", [[-1, 2], [2, -1, 3], [2, -1]])
def test_split_rejects_negative_sizes_before_sampling(source_dataset, sizes):
    original_indices = source_dataset.episode_indices.copy()
    with (
        patch(
            "minari.utils.np.random.default_rng", wraps=np.random.default_rng
        ) as generator,
        patch("minari.utils.MinariDataset", wraps=MinariDataset) as dataset_view,
    ):
        with pytest.raises(ValueError, match="non-negative"):
            minari.split_dataset(source_dataset, sizes=sizes, seed=0)

        generator.assert_not_called()
        dataset_view.assert_not_called()
    np.testing.assert_array_equal(source_dataset.episode_indices, original_indices)


@pytest.mark.parametrize("sizes", [[1, 3], [2, 0, 2], [1, 1]])
def test_split_preserves_valid_sizes_and_source(source_dataset, sizes):
    original_indices = source_dataset.episode_indices.copy()
    original_episodes = list(source_dataset)

    splits = minari.split_dataset(source_dataset, sizes=sizes, seed=0)
    repeated = minari.split_dataset(source_dataset, sizes=sizes, seed=0)

    assert [split.total_episodes for split in splits] == sizes
    selected = [index for split in splits for index in split.episode_indices]
    assert len(selected) == len(set(selected)) == sum(sizes)
    assert set(selected).issubset(set(original_indices))
    if sum(sizes) == source_dataset.total_episodes:
        assert set(selected) == set(original_indices)

    for split, same_seed_split in zip(splits, repeated):
        np.testing.assert_array_equal(
            split.episode_indices, same_seed_split.episode_indices
        )
        for source_index, episode in zip(split.episode_indices, split):
            original = original_episodes[source_index]
            for field in (
                "observations",
                "actions",
                "rewards",
                "terminations",
                "truncations",
            ):
                np.testing.assert_array_equal(
                    getattr(episode, field), getattr(original, field)
                )

    np.testing.assert_array_equal(source_dataset.episode_indices, original_indices)
    for original, current in zip(original_episodes, source_dataset):
        for field in ("observations", "actions", "rewards"):
            np.testing.assert_array_equal(
                getattr(current, field), getattr(original, field)
            )

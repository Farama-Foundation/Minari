import gymnasium as gym
import numpy as np
import pytest

from minari.data_collector.episode_buffer import EpisodeBuffer
from minari.dataset.minari_storage import MinariStorage, is_image_space


@pytest.mark.parametrize("format", ["arrow", "parquet", "hdf5"])
@pytest.mark.parametrize("channels", [1, 2, 4])
def test_non_jpeg_channel_shapes_round_trip_losslessly(tmp_path, format, channels):
    shape = (32, 32, channels)
    space = gym.spaces.Box(0, 255, shape, dtype=np.uint8)
    storage = MinariStorage.new(
        tmp_path,
        observation_space=space,
        action_space=gym.spaces.Discrete(2),
        data_format=format,
    )
    observations = np.random.default_rng(0).integers(
        0, 256, size=(2, *shape), dtype=np.uint8
    )
    storage.update_episodes(
        [
            EpisodeBuffer(
                observations=observations,
                actions=[0],
                rewards=[0.0],
                terminations=[True],
                truncations=[False],
            )
        ]
    )
    result = next(iter(storage.get_episodes([0])))
    np.testing.assert_array_equal(result["observations"], observations)


@pytest.mark.parametrize("shape", [(32, 32), (32, 32, 3)])
def test_jpeg_compatible_spaces_keep_image_detection(shape):
    assert is_image_space(gym.spaces.Box(0, 255, shape, dtype=np.uint8))

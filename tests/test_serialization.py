import gymnasium as gym
import numpy as np
import pytest

from minari.serialization import deserialize_space, serialize_space


@pytest.mark.parametrize(
    "space",
    [
        gym.spaces.Box(low=-1, high=4, shape=(2,), dtype=np.float32),
        gym.spaces.Box(low=-1, high=4, shape=(3,), dtype=np.float32),
        gym.spaces.Box(low=-1, high=4, shape=(2, 2, 2), dtype=np.float32),
        gym.spaces.Box(low=-1, high=4, shape=(3, 3, 3), dtype=np.float32),
        gym.spaces.Tuple(
            (
                gym.spaces.Discrete(1),
                gym.spaces.Discrete(5),
            )
        ),
        gym.spaces.Tuple(
            (
                gym.spaces.Box(low=-1, high=4, dtype=np.float32),
                gym.spaces.Discrete(5),
            )
        ),
        gym.spaces.Dict(
            {
                "component_1": gym.spaces.Box(low=-1, high=1, dtype=np.float32),
                "component_2": gym.spaces.Dict(
                    {
                        "subcomponent_1": gym.spaces.Box(
                            low=2, high=3, dtype=np.float32
                        ),
                        "subcomponent_2": gym.spaces.Box(
                            low=4, high=5, dtype=np.float32
                        ),
                    }
                ),
            }
        ),
        gym.spaces.Tuple(
            (
                gym.spaces.Box(low=2, high=3, dtype=np.float32),
                gym.spaces.Box(low=4, high=5, dtype=np.float32),
            )
        ),
        gym.spaces.Tuple(
            (
                gym.spaces.Box(low=2, high=3, dtype=np.float32),
                gym.spaces.Tuple(
                    (
                        gym.spaces.Box(low=2, high=3, dtype=np.float32),
                        gym.spaces.Box(low=4, high=5, dtype=np.float32),
                    )
                ),
            )
        ),
        gym.spaces.Tuple(
            (
                gym.spaces.Box(low=2, high=3, dtype=np.float32),
                gym.spaces.Tuple(
                    (
                        gym.spaces.Box(low=2, high=3, dtype=np.float32),
                        gym.spaces.Dict(
                            {
                                "component_1": gym.spaces.Box(
                                    low=-1, high=1, dtype=np.float32
                                ),
                                "component_2": gym.spaces.Dict(
                                    {
                                        "subcomponent_1": gym.spaces.Box(
                                            low=2, high=3, dtype=np.float32
                                        ),
                                        "subcomponent_2": gym.spaces.Tuple(
                                            (
                                                gym.spaces.Box(
                                                    low=4, high=5, dtype=np.float32
                                                ),
                                                gym.spaces.Discrete(10),
                                            )
                                        ),
                                    }
                                ),
                            }
                        ),
                    )
                ),
            )
        ),
    ],
)
def test_space_serialize_deserialize(space):

    space_str = serialize_space(space)
    reconstructed_space = deserialize_space(space_str)
    reserialized_space_str = serialize_space(reconstructed_space)
    assert space_str == reserialized_space_str

    space.seed(0)
    reconstructed_space.seed(0)
    action_1 = space.sample()
    action_2 = reconstructed_space.sample()
    assert space.contains(action_2)
    assert reconstructed_space.contains(action_1)

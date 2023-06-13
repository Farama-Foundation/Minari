import gymnasium as gym
import numpy as np


test_spaces = [
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
                    "subcomponent_1": gym.spaces.Box(low=2, high=3, dtype=np.float32),
                    "subcomponent_2": gym.spaces.Box(low=4, high=5, dtype=np.float32),
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
]

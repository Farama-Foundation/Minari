from collections import OrderedDict
from typing import Any

import numpy as np
import pytest
from gymnasium import spaces
from gymnasium.spaces.utils import flatten

from minari.serialization import deserialize_space, serialize_space
from tests.common import test_spaces, unsupported_test_spaces


@pytest.mark.parametrize("to_string", [False, True])
@pytest.mark.parametrize("container", ["root", "tuple", "dict"])
def test_dict_space_round_trip_preserves_flattened_observations(to_string, container):
    space: spaces.Space = spaces.Dict(
        OrderedDict(
            [
                ("velocity", spaces.Box(-10, 10, shape=(1,), dtype=np.float32)),
                ("position", spaces.Box(-10, 10, shape=(2,), dtype=np.float32)),
            ]
        )
    )
    observation: Any = {
        "velocity": np.array([7], dtype=np.float32),
        "position": np.array([1, 2], dtype=np.float32),
    }
    if container == "tuple":
        space = spaces.Tuple((spaces.Discrete(2), space))
        observation = (1, observation)
    elif container == "dict":
        space = spaces.Dict(
            OrderedDict([("state", space), ("alive", spaces.Discrete(2))])
        )
        observation = {"state": observation, "alive": 1}

    reconstructed = deserialize_space(serialize_space(space, to_string=to_string))

    np.testing.assert_array_equal(
        flatten(reconstructed, observation), flatten(space, observation)
    )
    assert reconstructed == space


@pytest.mark.parametrize("space", test_spaces)
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


@pytest.mark.parametrize("space", unsupported_test_spaces)
def test_space_serialize_deserialize_unsupported(space):
    with pytest.raises(
        NotImplementedError, match=r"No serialization method available for .+"
    ):
        serialize_space(space)

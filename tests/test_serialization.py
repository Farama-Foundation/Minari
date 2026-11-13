import gymnasium as gym
import numpy as np
import pytest

from minari.serialization import (
    _DISCRETE_ACCEPTS_DTYPE,
    deserialize_space,
    serialize_space,
)
from tests.common import test_spaces, unsupported_test_spaces


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


@pytest.mark.skipif(
    not _DISCRETE_ACCEPTS_DTYPE,
    reason="`spaces.Discrete` takes a dtype from Gymnasium 1.2.2",
)
@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_discrete_keeps_its_dtype(dtype):
    space = gym.spaces.Discrete(5, start=-2, dtype=dtype)

    reconstructed = deserialize_space(serialize_space(space))

    assert reconstructed.dtype == space.dtype
    assert reconstructed == space

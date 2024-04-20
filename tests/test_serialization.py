import pytest

from minari.serialization import deserialize_space, serialize_space
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

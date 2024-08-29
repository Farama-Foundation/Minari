import gymnasium as gym
import pytest


pytest.importorskip("torchrl")
from torchrl.data.datasets.minari_data import MinariExperienceReplay  # noqa: E402


def test_torch_minari_experience_replay():
    """
    Tests creation of the torchrl MinariExperienceReplay object.

    According to their documentation, torchrl uses the MinariExperienceReplay
    object as a way to download datasets from the remote Farama server.
    """

    batch_size = 32

    dataset = MinariExperienceReplay("D4RL/door/human-v2", batch_size=batch_size)
    env = gym.make("AdroitHandDoor-v1")

    sample = dataset.sample()

    assert sample.batch_size[0] == batch_size
    assert sample.shape[0] == batch_size

    assert sample[0]["action"].shape == env.action_space.shape
    assert sample[0]["observation"].shape == env.observation_space.shape

    # Check that the dataset cannot be written to
    with pytest.raises(RuntimeError):
        dataset.empty()

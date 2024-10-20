import tempfile

import pytest
from gymnasium import register, registry
from pytest import MonkeyPatch


@pytest.fixture(autouse=True)
def tmp_dataset_dir():
    """Generate a temporary directory for Minari datasets."""
    tmp_dir = tempfile.TemporaryDirectory()
    with MonkeyPatch.context() as mp:
        mp.setenv("MINARI_DATASETS_PATH", tmp_dir.name)
        yield tmp_dir.name
    tmp_dir.cleanup()


@pytest.fixture
def register_dummy_envs():
    env_names = [
        "DummyBoxEnv",
        "DummyInfoEnv",
        "DummySingleStepEnv",
        "DummyInconsistentInfoEnv",
        "DummyMultiDimensionalBoxEnv",
        "DummyMultiSpaceEnv",
        "DummyTupleDiscreteBoxEnv",
        "DummyDictEnv",
        "DummyTupleEnv",
        "DummyTextEnv",
        "DummyComboEnv",
    ]
    for env_name in env_names:
        register(
            id=f"{env_name}-v0",
            entry_point=f"tests.common:{env_name}",
            max_episode_steps=5,
        )

    yield

    for env_name in env_names:
        registry.pop(f"{env_name}-v0")

import tempfile

import pytest
from pytest import MonkeyPatch


@pytest.fixture(autouse=True)
def tmp_dataset_dir():
    """Generate a temporary directory for Minari datasets."""
    tmp_dir = tempfile.TemporaryDirectory()
    with MonkeyPatch.context() as mp:
        mp.setenv("MINARI_DATASETS_PATH", tmp_dir.name)
        yield tmp_dir.name
    tmp_dir.cleanup()

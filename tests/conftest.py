import os
import tempfile

import pytest


@pytest.fixture(autouse=True)
def tmp_dataset_dir():
    """Generate a temporary directory for Minari datasets."""
    tmp_dir = tempfile.TemporaryDirectory()
    os.environ["MINARI_DATASETS_PATH"] = tmp_dir.name
    yield tmp_dir.name
    tmp_dir.cleanup()

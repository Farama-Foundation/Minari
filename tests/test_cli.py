import pytest
from typer.testing import CliRunner

from minari.cli import app
from minari.storage.hosting import list_remote_datasets
from tests.dataset.test_dataset_download import get_latest_compatible_dataset_id


runner = CliRunner()


def test_list_app():
    result = runner.invoke(app, ["list", "local", "--all"])
    assert result.exit_code == 0
    # some of the other columns may be cut off by Rich
    assert "Name" in result.stdout

    result = runner.invoke(app, ["list", "remote"])
    assert result.exit_code == 0
    assert "Minari datasets in Farama server" in result.stdout


@pytest.mark.parametrize(
    "dataset_id",
    [get_latest_compatible_dataset_id(namespace="D4RL/pen", dataset_name="human")],
)
def test_dataset_download_then_delete(dataset_id: str):
    """Test download dataset invocation from CLI.

    The downloading functionality is already tested in test_dataset_download.py so this
    is primarily to assert that the CLI is working as expected.
    """
    result = runner.invoke(app, ["download", dataset_id])

    assert result.exit_code == 0
    assert f"Downloading {dataset_id} from Farama servers..." in result.stdout
    assert f"Dataset {dataset_id} downloaded to" in result.stdout

    result = runner.invoke(app, ["delete", dataset_id], input="n")
    assert result.exit_code == 1  # aborted but no error
    assert "Aborted" in result.stdout

    result = runner.invoke(app, ["delete", dataset_id], input="😳")
    assert result.exit_code == 1
    assert "Error: invalid input" in result.stdout

    result = runner.invoke(app, ["delete", dataset_id], input="y")
    assert result.exit_code == 0
    assert f"Dataset {dataset_id} deleted!" in result.stdout


@pytest.mark.parametrize(
    "dataset_id",
    list_remote_datasets(compatible_minari_version=True),
)
def test_minari_show(dataset_id):
    result = runner.invoke(app, ["show", dataset_id])
    assert result.exit_code == 0

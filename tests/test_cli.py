import pytest
from typer.testing import CliRunner

from minari.cli import app
from minari.storage.local import delete_dataset, list_local_datasets
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
    [get_latest_compatible_dataset_id(env_name="pen", dataset_name="human")],
)
def test_dataset_download_then_delete(dataset_id: str):
    """Test download dataset invocation from CLI.

    the downloading functionality is already tested in test_dataset_download.py so this is primarily to assert that the CLI is working as expected.
    """
    # might have to clear up the local dataset first.
    # ideally this seems like it could just be handled by the tests
    if dataset_id in list_local_datasets():
        delete_dataset(dataset_id)

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

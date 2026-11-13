"""Offline regression tests for dataset version filtering in the CLI."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

import minari
from minari.cli import app


@pytest.mark.parametrize("path", ["local", "remote", "hf://farama-minari/datasets"])
@pytest.mark.parametrize("all_flag", [[], ["--all"], ["-a"]])
def test_list_version_filters(monkeypatch, path, all_flag):
    list_datasets = Mock(return_value={})
    if path == "local":
        monkeypatch.setattr("minari.cli.local.list_local_datasets", list_datasets)
    else:
        monkeypatch.setattr("minari.cli.hosting.list_remote_datasets", list_datasets)

    result = CliRunner().invoke(app, ["list", path, *all_flag])

    assert result.exit_code == 0, result.output
    assert list_datasets.call_count == 1
    assert list_datasets.call_args.kwargs["latest_version"] is (not all_flag)
    assert list_datasets.call_args.kwargs["compatible_minari_version"] is (not all_flag)


@pytest.mark.parametrize("all_flag", [[], ["--all"], ["-a"]])
def test_list_local_versions(tmp_dataset_dir, monkeypatch, all_flag):
    monkeypatch.setenv("COLUMNS", "200")
    for version in range(3):
        dataset_id = f"example-v{version}"
        data_path = Path(tmp_dataset_dir, dataset_id, "data")
        data_path.mkdir(parents=True)
        metadata = {
            "dataset_id": dataset_id,
            "minari_version": minari.__version__ if version < 2 else "0.0.0",
            "total_episodes": 1,
            "total_steps": 1,
            "dataset_size": 0.1,
        }
        (data_path / "metadata.json").write_text(json.dumps(metadata))

    result = CliRunner().invoke(app, ["list", "local", *all_flag])

    assert result.exit_code == 0, result.output
    assert "example-v1" in result.stdout
    assert ("example-v0" in result.stdout) is bool(all_flag)
    assert ("example-v2" in result.stdout) is bool(all_flag)

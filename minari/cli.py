"""Minari CLI commands."""
import os
from typing import List, Optional
from typing_extensions import Annotated

import typer
from rich import print
from rich.table import Table
from rich.tree import Tree

from minari import __version__
from minari.storage import get_dataset_path, hosting, local
from minari.utils import combine_datasets


app = typer.Typer()
list_app = typer.Typer()
app.add_typer(list_app, name="list", short_help="List Minari datasets.")


def _version_callback(value: bool):
    """Show installed Minari version."""
    if value:
        typer.echo(f"Minari version: {__version__}")
        raise typer.Exit()


def _show_dataset_table(datasets, table_title):
    table = Table(title=table_title)

    table.add_column("Name", justify="left", style="cyan", no_wrap=True)
    table.add_column("Total Episodes", justify="right", style="green")
    table.add_column("Total Steps", justify="right", style="green")
    table.add_column("Description", justify="left", style="yellow")
    table.add_column("Author", justify="left", style="magenta")
    table.add_column("Email", justify="left", style="magenta")

    for dst_metadata in datasets.values():
        assert isinstance(dst_metadata["dataset_id"], str)
        assert isinstance(dst_metadata["author"], str)
        assert isinstance(dst_metadata["author_email"], str)
        table.add_row(
            dst_metadata["dataset_id"],
            str(dst_metadata["total_episodes"]),
            str(dst_metadata["total_steps"]),
            "Coming soon ...",
            dst_metadata["author"],
            dst_metadata["author_email"],
        )

    print(table)


@app.callback()
def common(
    version: Annotated[
        Optional[bool],
        typer.Option(
            "--version",
            "-v",
            callback=_version_callback,
            help="Show installed Minari version.",
        ),
    ] = None,
):
    """Minari is a tool for collecting and hosting Offline datasets for Reinforcement Learning environments based on the Gymnaisum API."""
    pass


@list_app.command("remote")
def list_remote(
    all: Annotated[
        bool, typer.Option("--all", "-a", help="Show all dataset versions.")
    ] = False
):
    """List Minari datasets hosted in the Farama server."""
    if all:
        datasets = hosting.list_remote_datasets()
    else:
        datasets = hosting.list_remote_datasets(
            latest_version=True, compatible_minari_version=True
        )
    table_title = "Minari datasets in Farama server"
    _show_dataset_table(datasets, table_title)


@list_app.command("local")
def list_local(
    all: Annotated[
        bool, typer.Option("--all", "-a", help="Show all dataset versions.")
    ] = False
):
    """List local Minari datasets."""
    if all:
        datasets = local.list_local_datasets()
    else:
        datasets = local.list_local_datasets(
            latest_version=True, compatible_minari_version=True
        )
    dataset_dir = os.environ.get(
        "MINARI_DATASETS_PATH",
        os.path.join(os.path.expanduser("~"), ".minari/datasets/"),
    )
    table_title = f"Local Minari datasets('{dataset_dir}')"
    _show_dataset_table(datasets, table_title)


@app.command()
def delete(datasets: Annotated[List[str], typer.Argument()]):
    """Delete datasets from local database."""
    # check that the given local datasets exist
    local_dsts = local.list_local_datasets()
    non_matching_local = [dst for dst in datasets if dst not in local_dsts]
    if len(non_matching_local) > 0:
        local_dataset_path = get_dataset_path("")
        tree = Tree(
            f"The following datasets can't be found locally at `{local_dataset_path}`",
            style="red",
        )
        for dst in non_matching_local:
            tree.add(dst, style="magenta")
        print(tree)
        raise typer.Abort()

    # prompt to delete datasets
    datasets_to_delete = {
        local_name: local_dsts[local_name]
        for local_name in local_dsts.keys()
        if local_name in datasets
    }
    _show_dataset_table(datasets_to_delete, "Delete local Minari datasets")
    typer.confirm("Are you sure you want to delete these local datasets?", abort=True)

    for dst in datasets:
        local.delete_dataset(dst)


@app.command()
def download(
    datasets: Annotated[List[str], typer.Argument()],
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Perform a force download.")
    ] = False,
):
    """Download Minari datasets from Farama server."""
    # check if datasets exist in remote server
    remote_dsts = hosting.list_remote_datasets()

    non_matching_remote = [dst for dst in datasets if dst not in remote_dsts]
    if len(non_matching_remote) > 0:
        tree = Tree(
            "The following datasets can't be found in the remote Farama server",
            style="red",
        )
        for dst in non_matching_remote:
            tree.add(dst, style="magenta")
        print(tree)
        raise typer.Abort()

    # check existing datasets locally and ask again if you want them to be overridden
    local_dsts = local.list_local_datasets()
    datasets_to_override = {
        local_name: local_dsts[local_name]
        for local_name in local_dsts.keys()
        if local_name in datasets
    }

    if len(datasets_to_override) > 0 and not force:
        _show_dataset_table(datasets_to_override, "Download remote Minari datasets")
        typer.confirm(
            "Are you sure you want to download and override these local datasets?",
            abort=True,
        )

    # download datastets
    for dst in datasets:
        hosting.download_dataset(dst, force_download=force)


@app.command()
def upload(
    datasets: Annotated[List[str], typer.Argument()],
    key_path: Annotated[str, typer.Option()],
):
    """Upload Minari datasets to the remote Farama server."""
    local_dsts = local.list_local_datasets()
    remote_dsts = hosting.list_remote_datasets()

    # check that the given local datasets exist
    non_matching_local = [dst for dst in datasets if dst not in local_dsts]
    if len(non_matching_local) > 0:
        local_dataset_path = get_dataset_path("")
        tree = Tree(
            f"The following datasets can't be found locally at `{local_dataset_path}`",
            style="red",
        )
        for dst in non_matching_local:
            tree.add(dst, style="magenta")
        print(tree)
        raise typer.Abort()

    # check that non of the datasets exist in the Farama server
    matching_remote = {
        remote_name: remote_dsts[remote_name]
        for remote_name in remote_dsts.keys()
        if remote_name in datasets
    }
    if len(matching_remote) > 0:
        print(
            "[red]The following datasets are already present in the Farama server, please contact the Farama team at contact@farama.org to upload your datasets.[/red]",
        )
        _show_dataset_table(matching_remote, "Matching datasets in Farama server")
        raise typer.Abort()

    # Upload datasets
    for dst in datasets:
        hosting.upload_dataset(dst, key_path)


@app.command()
def combine(
    datasets: Annotated[List[str], typer.Argument()],
    dataset_id: Annotated[str, typer.Option()],
):
    """Combine multiple datasets into a single Minari dataset."""
    local_dsts = local.list_local_datasets()
    # check dataset name doesn't exist locally
    if dataset_id in local_dsts:
        print(
            f"[red]Dataset name {dataset_id} already exist in the local Minari datasets.[/red]",
        )
        raise typer.Abort()

    # check list of local datasets to combine exist
    non_matching_local = [dst for dst in datasets if dst not in local_dsts]
    if len(non_matching_local) > 0:
        local_dataset_path = get_dataset_path("")
        tree = Tree(
            f"The following datasets can't be found locally at `{local_dataset_path}`",
            style="red",
        )
        for dst in non_matching_local:
            tree.add(dst, style="magenta")
        print(tree)
        raise typer.Abort()
    if len(datasets) > 1:
        minari_datasets = list(map(lambda x: local.load_dataset(x), datasets))
        combine_datasets(minari_datasets, dataset_id)
        print(
            f"The datasets [green]{datasets}[/green] were successfully combined into [blue]{dataset_id}[/blue]!"
        )
    else:
        print(
            f"[red]The list of local datasets to combine {datasets} must be of size two or greater.[/red]",
        )
        raise typer.Abort()


if __name__ == "__main__":
    app()

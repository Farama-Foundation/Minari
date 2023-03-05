from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

from minari import __version__
from minari.storage import hosting, local


app = typer.Typer()


def version_callback(value: bool):
    if value:
        typer.echo(f"Minari version: {__version__}")
        raise typer.Exit()


@app.callback()
def common(
    version: Optional[bool] = typer.Option(
        None, "--version", "-v", callback=version_callback
    )
):
    """Minari is a tool for collecting and hosting Offline datasets for Reinforcement Learning environments based on the Gymnaisum API."""
    pass


@app.command()
def list(storage: str):
    """List Minari datasets."""
    table_title = " "
    if storage == "remote":
        datasets = hosting.list_remote_datasets()
        table_title = "Minari datasets in Farama server"
    else:
        datasets = local.list_local_datasets()
        dataset_dir = "home/rodrigo/.minari/"
        table_title = f"Local Minari datasets('{dataset_dir}')"

    table = Table(title=table_title)

    table.add_column("Name", justify="left", style="cyan", no_wrap=True)
    table.add_column("Total Episodes", justify="right", style="green")
    table.add_column("Total Steps", justify="right", style="green")
    table.add_column("Description", justify="left", style="yellow")
    table.add_column("Author", justify="left", style="magenta")
    table.add_column("Email", justify="left", style="magenta")

    for dst_metadata in datasets.values():
        assert isinstance(dst_metadata["dataset_name"], str)
        assert isinstance(dst_metadata["author"], str)
        assert isinstance(dst_metadata["author_email"], str)
        table.add_row(
            dst_metadata["dataset_name"],
            str(dst_metadata["total_episodes"]),
            str(dst_metadata["total_steps"]),
            "Coming soon ...",
            dst_metadata["author"],
            dst_metadata["author_email"],
        )

    console = Console()
    console.print(table)


@app.command()
def delete(datasets: List[str]):
    """Delete datasets from local database."""
    for dst in datasets:
        local.delete_dataset(dst)


@app.command()
def download(datasets: List[str]):
    # check existing datasets locally and ask again if you want them to be overridden
    # check if datasets exist in remote server, if not
    for dst in datasets:
        hosting.download_dataset(dst)


@app.command()
def upload(datasets: List[str], key_path: str = typer.Option(...)):
    """Upload Minari datasets to the remote Farama server."""
    for dst in datasets:
        hosting.upload_dataset(dst, key_path)


@app.command()
def combine(dataset_name, datasets):

    pass


if __name__ == "__main__":
    app()

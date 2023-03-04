from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

import minari
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
    if storage == "remote":
        hosting.list_remote_datasets()
    elif storage == "local":
        local_dataset_names = local.list_local_datasets(verbose=False)
        dataset_dir = "home/rodrigo/.minari/"
        table = Table(title=f"Local Minari datasets('{dataset_dir}')")

        table.add_column("Name", justify="left", style="cyan", no_wrap=True)
        table.add_column("Total Episodes", justify="right", style="green")
        table.add_column("Total Steps", justify="right", style="green")
        table.add_column("Description", justify="left", style="magenta")
        table.add_column("Author", justify="left", style="magenta")
        table.add_column("Email", justify="left", style="magenta")

        for dst in map(lambda x: minari.load_dataset(x), local_dataset_names):

            table.add_row(
                dst.name,
                str(dst.total_episodes),
                str(dst.total_steps),
                " ",
                dst.author,
                dst.email,
            )

        console = Console()
        console.print(table)
    else:
        pass


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
def upload(datasets: List[str]):
    """Upload Minari datasets to the remote Farama server."""
    pass


@app.command()
def combine(dataset_name, datasets):

    pass


if __name__ == "__main__":
    app()

"""Minari CLI commands."""

import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional
from typing_extensions import Annotated

import typer
from gymnasium.envs.registration import EnvSpec
from rich import print
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from minari import __version__
from minari.dataset.minari_dataset import gen_dataset_id, parse_dataset_id
from minari.storage import get_dataset_path, hosting, local
from minari.utils import combine_datasets, get_dataset_spec_dict, get_env_spec_dict


app = typer.Typer()
list_app = typer.Typer()
app.add_typer(list_app, name="list", short_help="List Minari datasets.")


def _version_callback(value: bool):
    """Show installed Minari version."""
    if value:
        typer.echo(f"Minari version: {__version__}")
        raise typer.Exit()


def _show_dataset_table(datasets: Dict[str, Dict[str, Any]], table_title: str):
    # Collect compatible versions of each dataset
    dataset_versions = defaultdict(list)

    for dataset_id in datasets.keys():
        namespace, dataset_name, version = parse_dataset_id(dataset_id)
        dataset_id_versionless = gen_dataset_id(namespace, dataset_name)
        dataset_versions[dataset_id_versionless].append(version)

    # "Versions" column is only displayed if there are multiple versions
    display_versions = any([len(x) > 1 for x in dataset_versions.values()])

    # Build the Rich Table
    table = Table(title=table_title)

    table.add_column("Name", justify="left", style="cyan", no_wrap=True)

    if display_versions:
        table.add_column("Versions", justify="right", style="green", no_wrap=True)

    table.add_column("Total Episodes", justify="right", style="green", no_wrap=True)
    table.add_column("Total Steps", justify="right", style="green", no_wrap=True)
    table.add_column("Dataset Size", justify="left", style="green", no_wrap=True)
    table.add_column("Author", justify="left", style="magenta", no_wrap=True)

    previous_namespace = None

    for dataset_prefix, versions in dataset_versions.items():
        dataset_id = f"{dataset_prefix}-v{max(versions)}"
        dst_metadata = datasets[dataset_id]
        author = dst_metadata.get("author", "Unknown")
        if not isinstance(author, str) and isinstance(author, Iterable):
            author = ", ".join(author)
        dataset_size = dst_metadata.get("dataset_size", "Unknown")
        if dataset_size != "Unknown":
            dataset_size = f"{str(dataset_size)} MB"
        author_email = dst_metadata.get("author_email", "Unknown")
        if not isinstance(author_email, str) and isinstance(author_email, Iterable):
            author_email = ", ".join(author_email)

        assert isinstance(dst_metadata["dataset_id"], str)
        assert isinstance(author, str)
        assert isinstance(author_email, str)

        docs_url = dst_metadata.get("docs_url", None)
        compatible_versions = ", ".join(
            [f"v{x}" for x in sorted(versions, reverse=True)]
        )

        if docs_url is not None:
            dataset_id_text = f"[link={docs_url}]{dataset_id}[/link]"
        else:
            dataset_id_text = dataset_id

        namespace, _, _ = parse_dataset_id(dataset_id)

        if namespace != previous_namespace:
            table.add_section()
            previous_namespace = namespace

        # Build the current table row
        rows = []
        rows.append(dataset_id_text)

        if display_versions:
            rows.append(compatible_versions)

        rows.append(str(dst_metadata["total_episodes"]))
        rows.append(str(dst_metadata["total_steps"]))
        rows.append(dataset_size)
        rows.append(author)
        table.add_row(*rows)

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
            latest_version=False, compatible_minari_version=True
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
            latest_version=False, compatible_minari_version=True
        )
    dataset_dir = os.environ.get(
        "MINARI_DATASETS_PATH",
        os.path.join(os.path.expanduser("~"), ".minari/datasets/"),
    )
    table_title = f"Local Minari datasets('{dataset_dir}')"
    _show_dataset_table(datasets, table_title)


@app.command()
def show(dataset: Annotated[str, typer.Argument()]):
    """Describe a local or remote dataset, and its environment."""
    local_datasets = local.list_local_datasets()

    # Try to find a local dataset first, then fall back to remote datasets
    if dataset in local_datasets:
        dst_metadata = local_datasets[dataset]
    else:
        remote_datasets = hosting.list_remote_datasets()

        if dataset in remote_datasets:
            dst_metadata = remote_datasets[dataset]
        else:
            local_dataset_path = get_dataset_path("")
            print(
                Text(
                    f"""The dataset `{dataset}` can't be found locally"""
                    f"""(at `{local_dataset_path}`) or remotely.""",
                    style="red",
                )
            )
            raise typer.Abort()

    dataset_id = dst_metadata["dataset_id"]
    description = dst_metadata.get("description")
    docs_url = dst_metadata.get("docs_url")

    if docs_url is not None:
        dataset_id_text = f"[{dataset_id}]({docs_url})"
    else:
        dataset_id_text = dataset_id

    dataset_spec_table = Table(show_header=False, highlight=True)
    dataset_spec_table.add_column(style="bold")
    dataset_spec_table.add_column(style="not bold")

    for key, value in get_dataset_spec_dict(dst_metadata).items():
        md = Markdown(
            str(value), inline_code_lexer="python", inline_code_theme="monokai"
        )
        dataset_spec_table.add_row(key, md)

    print(Markdown(f"""# {dataset_id_text}"""))

    if description is not None:
        print(Markdown(f"""\n## Description\n {description} """))

    print(Markdown("## Dataset Specs"))
    print(dataset_spec_table)

    for env_type in ["env_spec", "eval_env_spec"]:
        env_spec_json = dst_metadata.get(env_type)

        if env_spec_json is not None:
            assert isinstance(env_spec_json, str)
            env_spec_json = (  # for gymnasium 1.0.0 compatibility
                env_spec_json.replace('"order_enforce": true,', "")
                .replace('"apply_api_compatibility": false,', "")
                .replace('"autoreset": false, ', "")
            )
            env_spec = EnvSpec.from_json(env_spec_json)
            env_spec_table = Table(show_header=False, highlight=True)
            env_spec_table.add_column(style="bold")
            env_spec_table.add_column(style="not bold")

            for key, value in get_env_spec_dict(env_spec).items():
                md = Markdown(
                    value, inline_code_lexer="python", inline_code_theme="monokai"
                )
                env_spec_table.add_row(key, md)

            if env_type == "env_spec":
                print(Markdown("## Environment Specs"))
            else:
                print(Markdown("## Evaluation Environment Specs"))

            print(env_spec_table)


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

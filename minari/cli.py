"""Minari CLI commands."""

import os
from collections import deque
from typing import Any, Dict, List, Optional
from typing_extensions import Annotated

import typer
from gymnasium.envs.registration import EnvSpec
from rich import print
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from minari import __version__
from minari.dataset.minari_dataset import parse_dataset_id
from minari.namespace import namespace_hierarchy
from minari.storage import get_dataset_path, hosting, local
from minari.storage.remotes import DEFAULT_REMOTE
from minari.utils import combine_datasets, get_dataset_spec_dict, get_env_spec_dict


app = typer.Typer()


def _version_callback(value: bool):
    """Show installed Minari version."""
    if value:
        typer.echo(f"Minari version: {__version__}")
        raise typer.Exit()


class TableTree:

    def __init__(
        self,
        name: Optional[str] = None,
        total_episodes: int = 0,
        total_steps: int = 0,
        size: float = 0,
        authors: Optional[set] = None,
        count: int = 0,
        sub_nodes: Optional[Dict] = None,
        docs_url: Optional[str] = None,
    ):
        self.name = name
        self.total_episodes = total_episodes
        self.total_steps = total_steps
        self.size = size
        self.authors = authors if authors is not None else set()
        self.count = count
        self.sub_nodes = sub_nodes if sub_nodes is not None else {}
        self.docs_url = docs_url

    def update(self, other):
        self.total_episodes += other.total_episodes
        self.total_steps += other.total_steps
        self.size += other.size
        self.authors.update(other.authors)
        self.count += other.count

    def to_row(self) -> List[str]:
        if len(self.authors) == 0:
            authors = "Unknown"
        else:
            authors = ", ".join(self.authors)

        name = self.name
        assert name is not None
        if len(self.sub_nodes) > 0:
            name = f"[bold]{name}/...[/bold]\n [italic]Group with {self.count} datasets[/italic]"
        if self.docs_url is not None:
            name = f"[link={self.docs_url}]{name}[/link]"

        return [
            name,
            TableTree.print_num(self.total_episodes),
            TableTree.print_num(self.total_steps),
            self.print_size(),
            authors,
        ]

    def print_size(self) -> str:
        if self.size < 1_000:
            return f"{self.size:.1f} MB"
        elif self.size < 1_000_000:
            return f"{self.size / 1_000:.2f} GB"
        else:
            return f"{self.size / 1_000_000:.2f} TB"

    @staticmethod
    def print_num(num: float) -> str:
        num_letters = ["", "K", "M", "B", "T"]
        i = 0
        while num >= 1_000 and i < len(num_letters):
            num /= 1000
            i += 1

        return f"{num:.0f}{num_letters[i]}"


def _show_dataset_table(datasets: Dict[str, Dict[str, Any]], table_title: str):
    MAX_ROWS_PER_GROUP = 10

    table_tree = TableTree()
    for dataset_id in datasets.keys():
        dataset_metadata = datasets[dataset_id]
        dataset_node = TableTree(
            name=dataset_id,
            total_episodes=dataset_metadata["total_episodes"],
            total_steps=dataset_metadata["total_steps"],
            size=dataset_metadata["dataset_size"],
            authors=dataset_metadata.get("author"),
            count=1,
            docs_url=dataset_metadata.get("docs_url"),
        )

        table_tree.update(dataset_node)
        namespace, _, _ = parse_dataset_id(dataset_id)
        current_root = table_tree
        for ns in namespace_hierarchy(namespace):
            if ns not in current_root.sub_nodes:
                current_root.sub_nodes[ns] = TableTree(name=ns)
            current_root = current_root.sub_nodes[ns]
            current_root.update(dataset_node)
        current_root.sub_nodes[dataset_id] = dataset_node

    caption = (
        f"Number of datasets: {table_tree.count}, total size: {table_tree.print_size()}"
    )
    table = Table(title=table_title, caption=caption)
    table.add_column("Name", justify="left", style="cyan", no_wrap=True)
    table.add_column("# Episodes", justify="right", style="green", no_wrap=True)
    table.add_column("# Steps", justify="right", style="green", no_wrap=True)
    table.add_column("Size", justify="left", style="green", no_wrap=True)
    table.add_column("Author", justify="left", style="magenta", no_wrap=True)

    queue = deque(table_tree.sub_nodes.values())
    section_sentinel = object()
    while queue:
        table_node = queue.popleft()
        if table_node is section_sentinel:
            table.add_section()
        elif len(table_node.sub_nodes) == 0:
            table.add_row(*table_node.to_row())
        elif len(table_node.sub_nodes) <= MAX_ROWS_PER_GROUP:
            queue.extend(table_node.sub_nodes.values())
            queue.append(section_sentinel)
        else:
            table.add_row(*table_node.to_row())
            table.add_section()

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


@app.command("list")
def list_cmd(
    path: Annotated[
        str, typer.Argument(help="Can be `local`, `remote` or a remote path.")
    ],
    all: Annotated[
        bool, typer.Option("--all", "-a", help="Show all dataset versions.")
    ] = False,
):
    """List Minari datasets in local or remote storage."""
    if path == "local":
        datasets = local.list_local_datasets(
            latest_version=True, compatible_minari_version=not all
        )
        dataset_dir = os.environ.get(
            "MINARI_DATASETS_PATH",
            os.path.join(os.path.expanduser("~"), ".minari/datasets/"),
        )
        table_title = f"Local Minari datasets('{dataset_dir}')"
    else:
        remote_path = path
        if remote_path == "remote":
            remote_path = os.getenv("MINARI_REMOTE", DEFAULT_REMOTE)

        if "://" not in remote_path:
            print(
                Text(
                    f"Invalid remote path '{remote_path}'. Remote path should be in the format 'remote_type://remote_path'.",
                    style="red",
                )
            )
            raise typer.Abort()

        remote_type, remote_path = remote_path.split("://", maxsplit=1)
        remote_path, prefix, *_ = remote_path.split("/", maxsplit=1) + [None]
        remote_path = f"{remote_type}://{remote_path}"

        datasets = hosting.list_remote_datasets(
            remote_path=remote_path,
            prefix=prefix,
            latest_version=True,
            compatible_minari_version=not all,
        )
        table_title = f"Minari datasets in {remote_path}"

    _show_dataset_table(datasets, table_title)


@app.command()
def show(dataset_id: Annotated[str, typer.Argument()]):
    """Describe a local or remote dataset, and its environment."""
    local_datasets = local.list_local_datasets(prefix=dataset_id)
    if dataset_id in local_datasets:
        dst_metadata = local_datasets[dataset_id]
    else:
        remote_path = None
        if "://" in dataset_id:
            remote_type, remote_path = dataset_id.split("://", maxsplit=1)
            remote_path, dataset_id = remote_path.split("/", maxsplit=1)
            remote_path = f"{remote_type}://{remote_path}"
        remote_datasets = hosting.list_remote_datasets(
            remote_path=remote_path, prefix=dataset_id
        )

        if dataset_id in remote_datasets:
            dst_metadata = remote_datasets[dataset_id]
        else:
            local_dataset_path = get_dataset_path()
            print(
                Text(
                    f"""The dataset `{dataset_id}` can't be found locally """
                    f"""(at `{local_dataset_path}`) or remotely.""",
                    style="red",
                )
            )
            raise typer.Abort()

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
        local_dataset_path = get_dataset_path()
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
    dataset_id: Annotated[str, typer.Argument()],
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Perform a force download.")
    ] = False,
):
    """Download Minari datasets from Farama server."""
    local_dsts = local.list_local_datasets()

    dataset_name = dataset_id
    if "://" in dataset_name:
        _, dataset_name = dataset_name.split("://", maxsplit=1)
        _, dataset_name = dataset_name.split("/", maxsplit=1)
    if dataset_name in local_dsts and not force:
        _show_dataset_table(
            {dataset_name: local_dsts[dataset_name]}, "Download remote Minari datasets"
        )
        typer.confirm(
            f"Are you sure you want to override {dataset_id}?",
            abort=True,
        )

    hosting.download_dataset(dataset_id, force_download=True)


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
        local_dataset_path = get_dataset_path()
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
        local_dataset_path = get_dataset_path()
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

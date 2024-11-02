from __future__ import annotations

import logging
import os
import pathlib
import shutil
import subprocess
import venv
import warnings
from collections import defaultdict
from multiprocessing import Pool
from typing import OrderedDict

import generate_env_table
import generate_gif
from md_utils import dict_to_table

import minari
from minari.dataset.minari_dataset import gen_dataset_id, parse_dataset_id
from minari.namespace import download_namespace_metadata, get_namespace_metadata
from minari.utils import get_dataset_spec_dict


DATASET_FOLDER = pathlib.Path(__file__).parent.parent.joinpath("datasets")
NAMESPACE_CONTENTS = defaultdict(OrderedDict)

NO_ENV_MSG = """
```{eval-rst}

.. warning::
This dataset doesn't contain an `env_spec`, neither an `eval_env_spec` attribute. Any call to :func:`minari.MinariDataset.recover_environment` will throw an error.

```
"""
NO_TRAIN_ENV_MSG = """
```{eval-rst}

.. warning::
This dataset doesn't contain an `env_spec` attribute. Calling :func:`minari.MinariDataset.recover_environment` with `eval_env=False` will throw an error.

```
"""
NO_EVAL_ENV_MSG = """
```{{eval-rst}}

.. note::
This dataset doesn't contain an `eval_env_spec` attribute which means that the specs of the environment used for evaluation are the same as the specs of the environment used for creating the dataset. The following calls will return the same environment:

.. code-block::

        import minari

        dataset = minari.load_dataset('{}')
        env  = dataset.recover_environment()
        eval_env = dataset.recover_environment(eval_env=True)

        assert env.spec == eval_env.spec
```
"""
PRE_TRAIN_ENV_MSG = """
```{{eval-rst}}

.. note::
The following table rows correspond to the Gymnasium environment specifications used to generate the dataset.
To read more about what each parameter means you can have a look at the Gymnasium documentation https://gymnasium.farama.org/api/registry/#gymnasium.envs.registration.EnvSpec

This environment can be recovered from the Minari dataset as follows:

.. code-block::

        import minari

        dataset = minari.load_dataset('{}')
        env  = dataset.recover_environment()
```
"""
PRE_EVAL_ENV_MSG = """
```{{eval-rst}}

.. note::
This environment can be recovered from the Minari dataset as follows:

.. code-block::

        import minari

        dataset = minari.load_dataset('{}')
        eval_env  = dataset.recover_environment(eval_env=True)
```
"""


def main():
    os.environ["TQDM_DISABLE"] = "1"

    remote_datasets = minari.list_remote_datasets(latest_version=True)
    for i, (dataset_id, metadata) in enumerate(remote_datasets.items()):
        namespace, dataset_name, version = parse_dataset_id(dataset_id)
        if namespace is not None:
            DATASET_FOLDER.joinpath(namespace).mkdir(parents=True, exist_ok=True)
            ns = namespace.split("/")
            for i in range(1, len(ns)):
                parent = "/".join(ns[:i])
                sub_namespace = "/".join(ns[: i + 1])
                download_namespace_metadata(sub_namespace)
                sub_namespace_metadata = get_namespace_metadata(sub_namespace)
                NAMESPACE_CONTENTS[parent][sub_namespace] = {
                    "short_description": sub_namespace_metadata.get(
                        "description", ""
                    ).split(". ", 1)[0],
                    "file": ns[i],
                    "toctree": f"{ns[i]}/index",
                    "display_name": sub_namespace_metadata.get(
                        "display_name", ns[i][0].upper() + ns[i][1:]
                    ),
                }

            versioned_name = gen_dataset_id(None, dataset_name, version)
            NAMESPACE_CONTENTS[namespace][dataset_id] = {
                "short_description": metadata.get("description", "").split(". ", 1)[0],
                "file": versioned_name,
                "toctree": versioned_name,
                "display_name": versioned_name,
            }

    for namespace, content in NAMESPACE_CONTENTS.items():
        _generate_namespace_page(namespace, content)

    with Pool(processes=16) as pool:
        pool.map(_generate_dataset_page, remote_datasets.items())

    del os.environ["TQDM_DISABLE"]


def _generate_dataset_page(arg):
    dataset_id, metadata = arg
    _, dataset_name, version = parse_dataset_id(dataset_id)
    versioned_name = gen_dataset_id(None, dataset_name, version)

    venv_name = f"venv_{dataset_id.replace('/', '_')}"
    venv.create(venv_name, with_pip=True)
    python_path = pathlib.Path(venv_name) / "bin" / "python"
    pip_path = pathlib.Path(venv_name) / "bin" / "pip"

    requirements = [
        "minari[gcs,hdf5] @ git+https://github.com/Farama-Foundation/Minari.git",
        "imageio",
        "absl-py",
    ]
    requirements.extend(metadata.get("requirements", []))
    req_args = [pip_path, "install", *requirements]
    subprocess.check_call(req_args, stdout=subprocess.DEVNULL)
    logging.info(f"Installed requirements for {dataset_id}")

    try:
        minari.download_dataset(dataset_id)
        subprocess.check_call(
            [
                python_path,
                generate_gif.__file__,
                f"--dataset_id={dataset_id}",
                f"--path={DATASET_FOLDER}",
            ]
        )
        minari.delete_dataset(dataset_id)
        img_link_str = f'<img src="../{versioned_name}.gif" width="200" style="display: block; margin:0 auto"/>'
    except Exception as e:
        warnings.warn(f"Failed to generate gif for {dataset_id}: {e}")
        img_link_str = None

    env_docs = """"""
    env_spec = metadata.get("env_spec")
    eval_env_spec = metadata.get("eval_env_spec")

    if env_spec is None and eval_env_spec is None:
        env_docs += NO_ENV_MSG

    else:
        env_docs += "\n## Environment Specs\n"
        if env_spec is None:
            env_docs += NO_TRAIN_ENV_MSG
        else:
            env_docs += PRE_TRAIN_ENV_MSG.format(dataset_id)
            env_docs += "\n"

            train_spec_file = f"train_spec_{dataset_id.replace('/', '_')}.md"
            subprocess.check_call(
                [
                    python_path,
                    generate_env_table.__file__,
                    f"--env_spec={env_spec}",
                    f"--file_name={train_spec_file}",
                ]
            )

            env_docs += pathlib.Path(train_spec_file).read_text()
            env_docs += "\n"

        env_docs += """\n## Evaluation Environment Specs\n"""
        if eval_env_spec is None:
            env_docs += NO_EVAL_ENV_MSG.format(dataset_id)
        else:
            env_docs += PRE_EVAL_ENV_MSG.format(dataset_id)
            env_docs += "\n"

            eval_spec_file = f"eval_spec_{dataset_id.replace('/', '_')}.md"
            subprocess.check_call(
                [
                    python_path,
                    generate_env_table.__file__,
                    f"--env_spec={env_spec}",
                    f"--file_name={eval_spec_file}",
                ]
            )

            env_docs += pathlib.Path(eval_spec_file).read_text()
            env_docs += "\n"

    content = "---\nautogenerated:\n"
    content += f"title: {dataset_name.title()}\n"
    content += "---\n\n"
    content += f"# {dataset_name.title()}"
    content += "\n\n"

    if img_link_str is not None:
        content += img_link_str
        content += "\n\n"
    if "description" in metadata:
        content += "## Description"
        content += "\n\n"
        content += metadata["description"]
        content += "\n\n"

    content += "## Dataset Specs"
    content += "\n\n"
    content += dict_to_table(get_dataset_spec_dict(metadata))
    content += "\n\n"
    content += env_docs

    dataset_md_path = DATASET_FOLDER.joinpath(dataset_id + ".md")
    file = open(dataset_md_path, "w", encoding="utf-8")
    file.write(content)
    file.close()

    logging.info(f"Generated dataset page for {dataset_id}")
    shutil.rmtree(venv_name)


def _generate_namespace_page(namespace: str, namespace_content):
    namespace_path = DATASET_FOLDER.joinpath(namespace)
    download_namespace_metadata(namespace)
    namespace_metadata = get_namespace_metadata(namespace)
    title = namespace_metadata.get("display_name", namespace[0].upper() + namespace[1:])

    file_content = "---\nfirstpage:\nlastpage:\n---\n\n"
    file_content += f"# {title}\n\n"
    if "description" in namespace_metadata:
        file_content += f"{namespace_metadata['description']}\n\n"

    file_content += "## Content\n"
    file_content += "|     ID     | Description |\n"
    file_content += "| ---------- | ----------- |\n"
    for c in namespace_content.values():
        file_content += f'| <a href="{c["file"]}" title="{c["display_name"]}">{c["display_name"]}</a> | {c["short_description"]} |\n'

    file_content += "\n```{toctree}\n:hidden:\n"
    for c in namespace_content.values():
        file_content += f"{c['toctree']}\n"
    file_content += "```\n"

    namespace_file = namespace_path.joinpath("index.md")
    file = open(namespace_file, "w", encoding="utf-8")
    file.write(file_content)
    file.close()


if __name__ == "__main__":
    main()

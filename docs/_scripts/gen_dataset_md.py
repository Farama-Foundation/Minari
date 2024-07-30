import logging
import os
from collections import defaultdict
from typing import Dict

from generate_gif import generate_gif
from gymnasium.envs.registration import EnvSpec

from minari import list_remote_datasets
from minari.dataset.minari_dataset import gen_dataset_id, parse_dataset_id
from minari.storage.hosting import get_remote_dataset_versions
from minari.utils import get_dataset_spec_dict, get_env_spec_dict


def _md_table(table_dict: Dict[str, str]) -> str:
    markdown = "|    |    |\n |----|----|"
    for key, value in table_dict.items():
        markdown += f"\n| {key} | {value} |"
    return markdown


filtered_datasets = defaultdict(defaultdict)
all_remote_datasets = list_remote_datasets()

for dataset_id in all_remote_datasets.keys():
    namespace, dataset_name, version = parse_dataset_id(dataset_id)
    assert namespace is not None

    if dataset_name not in filtered_datasets[namespace]:
        max_version = get_remote_dataset_versions(
            namespace, dataset_name, latest_version=True
        )[0]
        max_version_dataset_id = gen_dataset_id(namespace, dataset_name, max_version)
        filtered_datasets[namespace][dataset_name] = all_remote_datasets[
            max_version_dataset_id
        ]

for namespace, datasets in filtered_datasets.items():
    available_datasets = """
## Available Datasets
| Dataset ID | Description |
| ---------- | ----------- |
"""

    for i, (dataset_name, dataset_spec) in enumerate(datasets.items()):
        if i == 0:
            related_pages_meta = "firstpage:\n"
        elif i == len(datasets) - 1:
            related_pages_meta = "lastpage:\n"
        else:
            related_pages_meta = ""

        dataset_id = dataset_spec["dataset_id"]

        description = None
        if "description" in dataset_spec:
            description = dataset_spec["description"]

        # Add dataset id and description to main env page
        available_datasets += f"""| <a href="../{namespace}/{dataset_name}" title="{dataset_id}">{dataset_id}</a> | {description.split('. ')[0] if description is not None else ""} |
"""

        # Generate gif
        try:
            generate_gif(dataset_id)
            path = f"../../gifs/{dataset_id}.gif"
            img_link_str = (
                f'<img src="{path}" width="200" style="display: block; margin:0 auto"/>'
            )
        except Exception as e:
            logging.warning(f"Failed to generate gif for {dataset_id}: {e}")
            img_link_str = None

        # Environment Docs
        env_docs = """"""
        env_spec = dataset_spec.get("env_spec")
        eval_env_spec = dataset_spec.get("eval_env_spec")

        if env_spec is None and eval_env_spec is None:
            env_docs += """
```{eval-rst}

.. warning::
   This dataset doesn't contain an `env_spec`, neither an `eval_env_spec` attribute. Any call to :func:`minari.MinariDataset.recover_environment` will throw an error.

```
"""
        else:
            env_docs += """
## Environment Specs
"""
            if env_spec is None:
                env_docs += """
```{eval-rst}

.. warning::
   This dataset doesn't contain an `env_spec` attribute. Calling :func:`minari.MinariDataset.recover_environment` with `eval_env=False` will throw an error.

```
"""
            else:
                env_docs += f"""
```{{eval-rst}}

.. note::
   The following table rows correspond to (in addition to the action and observation space) the Gymnasium environment specifications used to generate the dataset.
   To read more about what each parameter means you can have a look at the Gymnasium documentation https://gymnasium.farama.org/api/registry/#gymnasium.envs.registration.EnvSpec

   This environment can be recovered from the Minari dataset as follows:

   .. code-block::

        import minari

        dataset = minari.load_dataset('{dataset_id}')
        env  = dataset.recover_environment()
```

{_md_table(get_env_spec_dict(EnvSpec.from_json(env_spec)))}
"""

            env_docs += """
## Evaluation Environment Specs

"""
            if eval_env_spec is None:
                env_docs += f"""
```{{eval-rst}}

.. note::
   This dataset doesn't contain an `eval_env_spec` attribute which means that the specs of the environment used for evaluation are the same as the specs of the environment used for creating the dataset. The following calls will return the same environment:

   .. code-block::

        import minari

        dataset = minari.load_dataset('{dataset_id}')
        env  = dataset.recover_environment()
        eval_env = dataset.recover_environment(eval_env=True)

        assert env.spec == eval_env.spec
```
"""
            else:
                env_docs += f"""

```{{eval-rst}}

.. note::
   This environment can be recovered from the Minari dataset as follows:

   .. code-block::

        import minari

        dataset = minari.load_dataset('{dataset_id}')
        eval_env  = dataset.recover_environment(eval_env=True)
```

{_md_table(get_env_spec_dict(EnvSpec.from_json(eval_env_spec)))}
"""

        env_page = f"""---
autogenerated:
title: {dataset_name.title()}
{related_pages_meta}---
# {dataset_name.title()}
"""
        env_page += "\n\n"
        if img_link_str is not None:
            env_page += img_link_str
            env_page += "\n\n"
        if description is not None:
            env_page += "## Description"
            env_page += "\n\n"
            env_page += description
            env_page += "\n\n"

        env_page += "## Dataset Specs"
        env_page += "\n\n"
        env_page += _md_table(get_dataset_spec_dict(dataset_spec))
        env_page += "\n\n"
        env_page += env_docs

        dataset_doc_path = os.path.join(
            os.path.dirname(__file__), "..", "datasets", namespace
        )

        if not os.path.exists(dataset_doc_path):
            os.makedirs(dataset_doc_path)

        dataset_md_path = os.path.join(
            dataset_doc_path,
            dataset_name + ".md",
        )

        file = open(dataset_md_path, "w", encoding="utf-8")
        file.write(env_page)
        file.close()

    env_page_path = os.path.join(
        os.path.dirname(__file__), "..", "datasets", f"{namespace}.md"
    )
    file = open(env_page_path, "a", encoding="utf-8")
    file.write(available_datasets)
    file.close()

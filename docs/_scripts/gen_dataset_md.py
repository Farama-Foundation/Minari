import os
import re
from collections import defaultdict

import gymnasium as gym
from google.cloud import storage  # pyright: ignore [reportGeneralTypeIssues]
from gymnasium.envs.registration import EnvSpec

from minari import list_remote_datasets
from minari.dataset.minari_dataset import parse_dataset_id
from minari.serialization import deserialize_space
from minari.storage.hosting import get_remote_dataset_versions


def _generate_env_spec_table(env_spec: EnvSpec) -> str:
    """Create a markdown table with the environment specifications, including observation and action space."""
    env = gym.make(env_spec.id)

    action_space_table = env.action_space.__repr__().replace("\n", "")
    observation_space_table = env.observation_space.__repr__().replace("\n", "")

    return f"""
|    |    |
|----|----|
|ID| `{env_spec.id}`|
| Observation Space | `{re.sub(' +', ' ', observation_space_table)}` |
| Action Space | `{re.sub(' +', ' ', action_space_table)}` |
| entry_point | `{env_spec.entry_point}` |
| max_episode_steps | `{env_spec.max_episode_steps}` |
| reward_threshold | `{env_spec.reward_threshold}` |
| nondeterministic | `{env_spec.nondeterministic}` |
| order_enforce    | `{env_spec.order_enforce}`|
| autoreset        | `{env_spec.autoreset}` |
| disable_env_checker | `{env_spec.disable_env_checker}` |
| kwargs | `{env_spec.kwargs}` |
| additional_wrappers | `{env_spec.additional_wrappers}` |
| vector_entry_point | `{env_spec.vector_entry_point}` |

"""


filtered_datasets = defaultdict(defaultdict)
all_remote_datasets = list_remote_datasets()

for dataset_id in all_remote_datasets.keys():

    env_name, dataset_name, version = parse_dataset_id(dataset_id)

    if dataset_name not in filtered_datasets[env_name]:
        max_version = get_remote_dataset_versions(env_name, dataset_name, True)[0]
        max_version_dataset_id = "-".join([env_name, dataset_name, f"v{max_version}"])
        filtered_datasets[env_name][dataset_name] = all_remote_datasets[
            max_version_dataset_id
        ]

for env_name, datasets in filtered_datasets.items():
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

        # Dataset Specs
        dataset_id = dataset_spec["dataset_id"]
        total_timesteps = dataset_spec["total_steps"]
        total_episodes = dataset_spec["total_episodes"]
        dataset_action_space = (
            deserialize_space(dataset_spec["action_space"]).__repr__().replace("\n", "")
        )
        dataset_observation_space = (
            deserialize_space(dataset_spec["observation_space"])
            .__repr__()
            .replace("\n", "")
        )
        author = dataset_spec["author"]
        email = dataset_spec["author_email"]
        algo_name = dataset_spec["algorithm_name"]
        code = dataset_spec["code_permalink"]
        minari_version = dataset_spec["minari_version"]

        description = None
        if "description" in dataset_spec:
            description = dataset_spec["description"]

        # Add dataset id and description to main env page
        available_datasets += f"""| <a href="../{env_name}/{dataset_name}" title="{dataset_id}">{dataset_id}</a> | {description.split('. ')[0] if description is not None else ""} |
"""

        # Get image gif link if available
        img_path = f"{dataset_id}/_docs/_imgs/{dataset_id}.gif"
        storage_client = storage.Client.create_anonymous_client()
        bucket = storage_client.bucket(bucket_name="minari-datasets")

        img_exists = storage.Blob(bucket=bucket, name=img_path).exists(storage_client)

        img_link_str = None
        if img_exists:
            img_link_str = (
                f'<img src="https://storage.googleapis.com/minari-datasets/{dataset_id}/_docs/_imgs/{dataset_id}.gif" width="200" '
                'style="display: block; margin:0 auto"/>'
            )

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

{_generate_env_spec_table(EnvSpec.from_json(env_spec))}
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

{_generate_env_spec_table(EnvSpec.from_json(eval_env_spec))}
"""

        env_page = f"""---
autogenerated:
title: {dataset_name.title()}
{related_pages_meta}---
# {dataset_name.title()}

{img_link_str if img_link_str is not None else ""}

## Description

{description if description is not None else ""}

## Dataset Specs

|    |    |
|----|----|
|Total Timesteps| `{total_timesteps}`|
|Total Episodes | `{total_episodes}` |
| Dataset Observation Space | `{dataset_observation_space}` |
| Dataset Action Space | `{dataset_action_space}` |
| Algorithm           | `{algo_name}`           |
| Author              | `{author}`              |
| Email               | `{email}`               |
| Code Permalink      | <a href={code}>`{code}`</a> |
| Minari Version      | `{minari_version}`      |
| download            | `minari.download_dataset("{dataset_id}")` |

{env_docs}
"""

        dataset_doc_path = os.path.join(
            os.path.dirname(__file__), "..", "datasets", env_name
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
        os.path.dirname(__file__), "..", "datasets", f"{env_name}.md"
    )
    file = open(env_page_path, "a", encoding="utf-8")
    file.write(available_datasets)
    file.close()

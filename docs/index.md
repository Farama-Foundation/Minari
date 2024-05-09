---
hide-toc: true
firstpage:
lastpage:
---

```{toctree}
:hidden:
:caption: Introduction
content/basic_usage
content/minari_cli
content/dataset_standards
```

```{toctree}
:hidden:
:caption: API
api/minari_functions
api/minari_dataset
api/data_collector
```

```{toctree}
:hidden:
:glob:
:caption: Tutorials
tutorials/**/index
```

```{toctree}
:hidden:
:caption: Datasets
datasets/minigrid
datasets/door
datasets/hammer
datasets/relocate
datasets/pen
datasets/pointmaze
datasets/antmaze
datasets/kitchen
```

```{toctree}
:hidden:
:caption: Development

Github <https://github.com/Farama-Foundation/Minari>
release_notes/index
Contribute to the Docs <https://github.com/Farama-Foundation/Minari/tree/main/docs>
```

```{project-logo} _static/img/minari-text.png
:alt: Minari Logo
```

```{project-heading}
A dataset API for Offline Reinforcement Learning.
```

**Minari is a Python API that hosts a collection of popular Offline Reinforcement Learning datasets.** The environments from which these datasets are generated follow the [Gymnasium API](https://gymnasium.farama.org/). The datasets are publicly available in a [Farama GCP bucket](https://console.cloud.google.com/storage/browser/minari-datasets;tab=objects?forceOnBucketsSortingFiltering=false&amp;project=mcmes-345620&amp;prefix=&amp;forceOnObjectsSortingFiltering=false) and can be downloaded through the Minari CLI. Minari also provides dataset handling features such as episode sampling, filtering trajectories, as well as dataset generation utilities.

<div class="termy">

```console
// Install Minari
$ pip install minari
---> 100%

// Show remote datasets.
$ minari list remote

                 <i>Minari datasets in Farama server</i>
┌────────────────┬───────────┬────────────┬─────────┬───────────┐
│                │     <b>Total</b> │      <b>Total</b> │         │           │
│ <b>Name</b>           │  <b>Episodes</b> │      <b>Steps</b> │  <b>Author</b> │ <b>Email</b>     │
┡────────────────╇───────────╇────────────╇─────────╇───────────┩
│ <font color="#A1EFE4">door-cloned-v2</font> │      <font color="#03AC13">4356</font> │    <font color="#03AC13">1000000</font> │ <font color="#FF00FF">Farama</font>  │<font color="#FF00FF">@farama.org</font>│
│ <font color="#A1EFE4">door-expert-v2</font> │      <font color="#03AC13">5000</font> │    <font color="#03AC13">1000000</font> │ <font color="#FF00FF">Farama</font>  │<font color="#FF00FF">@farama.org</font>│
│ <font color="#A1EFE4">door-human-v2</font>  │        <font color="#03AC13">25</font> │       <font color="#03AC13">6729</font> │ <font color="#FF00FF">Farama</font>  │<font color="#FF00FF">@farama.org</font>│
└────────────────┴───────────┴────────────┴─────────┴───────────┘

// Download dataset door-cloned-v2
$ minari download door-cloned-v2
Downloading door-cloned-v2 from Farama servers...

   * Downloading data file 'door-cloned-v2/data/main_data.hdf5' ...

---> 100%

Dataset door-cloned-v2 downloaded to ~/.minari/datasets/door-cloned-v2

```
</div>

<p style="text-align: center;">
"見習い - learning by observation"
</p>

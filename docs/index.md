---
hide-toc: true
firstpage:
lastpage:
---

```{toctree}
:hidden:
:caption: Introduction
content/basic_usage
content/dataset_standards
content/minari_cli
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
datasets/door
datasets/hammer
datasets/relocate
datasets/pen
datasets/pointmaze
datasets/kitchen
```

```{toctree}
:hidden:
:caption: Development

Github <https://github.com/Farama-Foundation/Minari>
release_notes/index
Contribute to the Docs <https://github.com/Farama-Foundation/Minari/tree/main/docs>
```

```{project-logo} _static/img/minari-text.svg
:alt: Minari Logo
```

```{project-heading}
A dataset API for Offline Reinforcement Learning.
```

**Minari is a Python API that hosts a collection of popular Offline Reinforcement Learning datasets.** The environments from which these datasets are generated are standardized to follow the [Gymnasium API](https://gymnasium.farama.org/). The datasets are publicly available in a [Farama GCP bucket](https://console.cloud.google.com/storage/browser/minari-datasets;tab=objects?forceOnBucketsSortingFiltering=false&amp;project=mcmes-345620&amp;prefix=&amp;forceOnObjectsSortingFiltering=false) and can be downloaded through the Minari CLI as shown below. Minari also provides dataset handling features such as episode sampling, filtering trajectories with statiscal metrics or metadata, as well as data collection and dataset generation utilities.

```{eval-rst}

.. note::
   This library is currently under beta development and minor/major changes are expected to come along in the near future.

```

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
│ <font color="#A1EFE4">door-cloned-v0</font> │      <font color="#03AC13">4356</font> │    <font color="#03AC13">1000000</font> │ <font color="#FF00FF">Farama</font>  │<font color="#FF00FF">@farama.org</font>│
│ <font color="#A1EFE4">door-expert-v0</font> │      <font color="#03AC13">5000</font> │    <font color="#03AC13">1000000</font> │ <font color="#FF00FF">Farama</font>  │<font color="#FF00FF">@farama.org</font>│
│ <font color="#A1EFE4">door-human-v0</font>  │        <font color="#03AC13">25</font> │       <font color="#03AC13">6729</font> │ <font color="#FF00FF">Farama</font>  │<font color="#FF00FF">@farama.org</font>│
└────────────────┴───────────┴────────────┴─────────┴───────────┘ 

// Download dataset door-cloned-v0
$ minari download door-cloned-v0
Downloading door-cloned-v0 from Farama servers...

   * Downloading data file 'door-cloned-v0/data/main_data.hdf5' ...

---> 100%

Dataset door-cloned-v0 downloaded to <path-to-local-datasets>/.minari/datasets/door-cloned-v0

```
</div>

<p style="text-align: center;">
"見習い - learning by observation"
</p>

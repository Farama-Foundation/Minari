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
datasets/D4RL/index
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

**Minari is a Python API that hosts a collection of popular Offline Reinforcement Learning datasets.** The environments from which these datasets are generated follow the [Gymnasium API](https://gymnasium.farama.org/). The datasets are publicly available in a [Farama GCP bucket](https://console.cloud.google.com/storage/browser/minari-remote) and can be downloaded through the Minari CLI. Minari also provides dataset handling features such as episode sampling, filtering trajectories, as well as dataset generation utilities.

<div class="termy">

```console
// Install Minari
$ pip install "minari[all]"
---> 100%

// Show remote datasets.
$ minari list remote

                  <i>Minari datasets in Farama server</i>
┌─────────────────────┬───────────┬────────────┬───────────┬─────────┐
│                     │     <b>Total</b> │      <b>Total</b> │   <b>Dataset</b> │         │
│ <b>Name</b>                │  <b>Episodes</b> │      <b>Steps</b> │      <b>Size</b> │  <b>Author</b> │
┡─────────────────────╇───────────╇────────────╇───────────╇─────────┩
│ <font color="#A1EFE4">D4RL/door/cloned-v2</font> │      <font color="#03AC13">4356</font> │    <font color="#03AC13">1000000</font> │ <font color="#03AC13">1077.7 MB</font> │  <font color="#FF00FF">Farama</font> │
│ <font color="#A1EFE4">D4RL/door/expert-v2</font> │      <font color="#03AC13">5000</font> │    <font color="#03AC13">1000000</font> │ <font color="#03AC13">1096.4 MB</font> │  <font color="#FF00FF">Farama</font> │
│ <font color="#A1EFE4">D4RL/door/human-v2</font>  │        <font color="#03AC13">25</font> │       <font color="#03AC13">6729</font> │ <font color="#03AC13">7.1 MB</font>    │  <font color="#FF00FF">Farama</font> │
└─────────────────────┴───────────┴────────────┴───────────┴─────────┘

// Download dataset D4RL/door/cloned-v2
$ minari download D4RL/door/cloned-v2
Downloading D4RL/door/cloned-v2 from Farama servers...

   * Downloading data file 'D4RL/door/cloned-v2/data/main_data.hdf5' ...

---> 100%

Dataset D4RL/door/cloned-v2 downloaded to ~/.minari/datasets/D4RL/door/cloned-v2

```
</div>

<p style="text-align: center;">
"見習い - learning by observation"
</p>

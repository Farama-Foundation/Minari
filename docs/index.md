---
hide-toc: true
firstpage:
lastpage:
---

# Minari is a Python API for hosting and handling datasets for Offline Reinforcement Learning

```{figure} _static/img/minari-text.png
   :alt: Minari Logo
   :width: 600
```

<p style="text-align: center;">
"見習い - learning by observation"
</p>

<div class="termy">

```console
// Install Minari
$ pip install minari
---> 100%

// Check Minari version
$ minari --version
Minari version: 0.3.0

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
│ <font color="#A1EFE4">pen-cloned-v0</font>  │      <font color="#03AC13">3736</font> │     <font color="#03AC13">500000</font> │ <font color="#FF00FF">Farama</font>  │<font color="#FF00FF">@farama.org</font>│
│ <font color="#A1EFE4">pen-expert-v0</font>  │      <font color="#03AC13">4958</font> │     <font color="#03AC13">499206</font> │ <font color="#FF00FF">Farama</font>  │<font color="#FF00FF">@farama.org</font>│
│ <font color="#A1EFE4">pen-human-v0</font>   │        <font color="#03AC13">25</font> │       <font color="#03AC13">5000</font> │ <font color="#FF00FF">Farama</font>  │<font color="#FF00FF">@farama.org</font>│
└────────────────┴───────────┴────────────┴─────────┴───────────┘ 
```
</div>

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
:caption: Development

Github <https://github.com/Farama-Foundation/Minari>
```


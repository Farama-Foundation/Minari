---
layout: "contents"
title: Minari CLI
---

# Minari CLI

Minari is also packaged with some useful CLI commands. The CLI tool is build on top off [Typer](https://typer.tiangolo.com/) and it will be available after installing Minari.
The Minari CLI gives access to most of the package functions through the command-line such as listing existing datasets, downloading and deleting.

<div class="termy">

```console
// Install Minari
$ pip install minari
---> 100%

// Check Minari version
$ minari --version
Minari version: 0.5.0

// Show Minari CLI commands
$ minari --help

<b> </b><font color="#F4BF75"><b>Usage: </b></font><b>minari [OPTIONS] COMMANDS [ARGS]...                       </b>
<b>                                                     </b>
Minari is a tool for collecting and hosting Offline datasets for Reinforcement Learning environments based on the Gymnaisum API.

<font color="#A5A5A1">╭─ Options ─────────────────────────────────────────╮</font>
<font color="#A5A5A1">│ </font><font color="#A1EFE4"><b>--version</b></font>       </font><font color="#03AC13"><b>-v</b></font>            Show installed      │
<font color="#A5A5A1">│                               Minari version.     │</font>
<font color="#A5A5A1">│ </font><font color="#A1EFE4"><b>--help</b></font>                        Show this message   │
<font color="#A5A5A1">│                               and exit.           │</font>
<font color="#A5A5A1">╰───────────────────────────────────────────────────╯</font>
<font color="#A5A5A1">╭─ Commands ────────────────────────────────────────╮</font>
<font color="#A5A5A1">│ </font><font color="#A1EFE4"><b>combine</b></font>       Combine multiple datasets into a    │
<font color="#A5A5A1">│               single Minari dataset.              │</font>
<font color="#A5A5A1">│ </font><font color="#A1EFE4"><b>delete</b></font>        Delete datasets from local          │
<font color="#A5A5A1">│               database.                           │</font>
<font color="#A5A5A1">│ </font><font color="#A1EFE4"><b>download</b></font>      Download Minari datasets from       │
<font color="#A5A5A1">│               Farama server.                      │</font>
<font color="#A5A5A1">│ </font><font color="#A1EFE4"><b>list</b></font>          List Minari datasets.               │
<font color="#A5A5A1">│ </font><font color="#A1EFE4"><b>upload</b></font>        Upload Minari datasets to the       │
<font color="#A5A5A1">│               remote Farama server.               │</font>
<font color="#A5A5A1">╰───────────────────────────────────────────────────╯</font>
```
</div>

## List datasets

The `minari list COMMAND` command shows a table with the existing Minari datasets as well as some of their metadata such as number of episodes and steps in the dataset as well as the author's name and email.
This command comes with other two required sub-commands:

- `remote`: the Minari dataset table shows the datasets currently available in the remote Farama server.
- `local`: the Minari dataset table shows the datasets currently accessible in the local device.

```{eval-rst}

.. note::
   These commands will list the latest remote/local dataset versions that are compatible with your local installed Minari version. To list all the dataset versions (also incompatible) add the option :code:`--all` or :code:`-a` to the command.

```

<div class="termy">

```console
// Show local datasets.
$ minari list local

               <i>Local Minari datasets('.minari/')</i>
┌───────────────┬───────────┬───────────┬─────────┬───────────┐
│               │     <b>Total</b> │     <b>Total</b> │         │           │
│ <b>Name</b>          │  <b>Episodes</b> │     <b>Steps</b> │  <b>Author</b> │ <b>Email</b>     │
┡───────────────╇───────────╇───────────╇─────────╇───────────┩
│ <font color="#A1EFE4">pen-cloned-v2</font> │      <font color="#03AC13">3736</font> │    <font color="#03AC13">500000</font> │ <font color="#FF00FF">Rodrigo</font> │ <font color="#FF00FF">rperezvic…</font>│
│ <font color="#A1EFE4">pen-expert-v2</font> │      <font color="#03AC13">4958</font> │    <font color="#03AC13">499206</font> │ <font color="#FF00FF">Rodrigo</font> │ <font color="#FF00FF">rperezvic…</font>│
│ <font color="#A1EFE4">pen-human-v2</font>  │        <font color="#03AC13">25</font> │      <font color="#03AC13">5000</font> │ <font color="#FF00FF">Rodrigo</font> │ <font color="#FF00FF">rperezvic…</font>│
└───────────────┴───────────┴───────────┴─────────┴───────────┘

// Show remote datasets.
$ minari list remote

                 <i>Minari datasets in Farama server</i>
┌────────────────┬───────────┬────────────┬─────────┬───────────┐
│                │     <b>Total</b> │      <b>Total</b> │         │           │
│ <b>Name</b>           │  <b>Episodes</b> │      <b>Steps</b> │  <b>Author</b> │ <b>Email</b>     │
┡────────────────╇───────────╇────────────╇─────────╇───────────┩
│ <font color="#A1EFE4">door-cloned-v2</font> │      <font color="#03AC13">4356</font> │    <font color="#03AC13">1000000</font> │ <font color="#FF00FF">Rodrigo</font> │ <font color="#FF00FF">rperezvic…</font>│
│ <font color="#A1EFE4">door-expert-v2</font> │      <font color="#03AC13">5000</font> │    <font color="#03AC13">1000000</font> │ <font color="#FF00FF">Rodrigo</font> │ <font color="#FF00FF">rperezvic…</font>│
│ <font color="#A1EFE4">door-human-v2</font>  │        <font color="#03AC13">25</font> │       <font color="#03AC13">6729</font> │ <font color="#FF00FF">Rodrigo</font> │ <font color="#FF00FF">rperezvic…</font>│
│ <font color="#A1EFE4">pen-cloned-v2</font>  │      <font color="#03AC13">3736</font> │     <font color="#03AC13">500000</font> │ <font color="#FF00FF">Rodrigo</font> │ <font color="#FF00FF">rperezvic…</font>│
│ <font color="#A1EFE4">pen-expert-v2</font>  │      <font color="#03AC13">4958</font> │     <font color="#03AC13">499206</font> │ <font color="#FF00FF">Rodrigo</font> │ <font color="#FF00FF">rperezvic…</font>│
│ <font color="#A1EFE4">pen-human-v2</font>   │        <font color="#03AC13">25</font> │       <font color="#03AC13">5000</font> │ <font color="#FF00FF">Rodrigo</font> │ <font color="#FF00FF">rperezvic…</font>│
└────────────────┴───────────┴────────────┴─────────┴───────────┘
```
</div>

## Download datasets

With the command `minari download DATASET_NAME` you can download a group of datasets that are available in the remote Farama server. If the dataset name already exist locally, the Minari CLI will prompt you to override the
current content of the local dataset.

```{eval-rst}

.. note::
   The download is aborted if the remote dataset is not compatible with your local installed Minari version or through a warning if the dataset already exists locally. To perform a force download add :code:`--force` or :code:`-f` to the download command.

```

<div class="termy">

```console
// Download datasets pen-cloned-v2 and door-human-v2
$ minari download pen-cloned-v2 door-human-v2

Downloading pen-expert-v2 from Farama servers...

   * Downloading data file 'pen-expert-v2/data/main_data.hdf5' ...

---> 100%

Dataset pen-expert-v2 downloaded to <path-to-local-datasets>/.minari/datasets/pen-expert-v2

Downloading door-human-v2 from Farama servers...

   * Downloading data file 'door-human-v2/data/main_data.hdf5' ...

---> 100%

Dataset door-human-v2 downloaded to <path-to-local-datasets>/.minari/datasets/pen-expert-v2

```
</div>

## Show datasets details

You can check the details of a dataset using the command `minari show DATASET_NAME`. This command works for both local and remote datasets.

<div class="termy">

```console
// Show dataset details
$ minari show pen-expert-v2

┌───────────────────────────────────────────┐
| <center> <b> pen-expert-v2 </b> </center> |
└───────────────────────────────────────────┘

<center> <b> <u> Description </u> </b> </center>
Trajectories have expert data from a fine-tuned RL policy provided in the <a href="https://github.com/aravindr93/hand_dapg">DAPG</a> repository. The environment used to collect the dataset is <a href="https://robotics.farama.org/envs/adroit_hand/adroit_pen/"><code>AdroitHandPen-v1</code></a>.

<center> <b> <u> Dataset Specs </u> </b> </center>

┌──────────────────────┬────────────────────────┐
| <b>Total Steps</b>   │ 499206                         │
│ <b>Total Episodes</b> │ 4958                                             │
│ <b>Algorithm</b>    │ Not provided                                  │
│ <b>Author</b>    │ Rodrigo de Lazcano                                 │
│ <b>Email</b>        │ rperezvicente@farama.org                                    │
│ <b>Code Permalink</b> │ <a href="https://github.com/rodrigodelazcano/d4rl-minari-dataset-generation">https://github.com/rodrigodelazcano/d4rl-minari-dataset-generation</a>   │
│ <b>Minari Version</b> │ 0.4  (0.5.0 installed)                        │
│ <b>Download</b>    │ <code>minari download pen-expert-v2</code>  |
└────────────────────┴─────────────────────────────┘

<center> <b> <u> Environment Specs </u> </b> </center>
┌──────────────────────┬────────────────────────┐
| <b>ID</b>   │ AdroitHandPen-v1    │
│ <b>Observation Space</b> │ <code> Box(-inf, inf, (45,), float64) </code>           │
│ <b>Action Space</b>    │ <code> Box(-1.0, 1.0, (24,), float32) </code>                                  │
│ <b>entry_point</b>    │ <code>gymnasium_robotics.envs.adroit_hand.adroit_pen:AdroitHandPenEnv</code>    │
│ <b>max_episode_steps</b>        │ 100                              │
│ <b>reward_threshold </b> │  None  │
│ <b>nondeterministic</b> │ <code>False</code>    │
│ <b>order_enforce</b> │  <code>True</code>   │
│ <b>autoreset</b> │  <code>False</code>   │
│ <b>disable_env_checker</b> │  <code>False</code>  │
│ <b>kwargs</b> │  <code>{'reward_type': 'dense'}</code>  │
│ <b>additional_wrappers</b> │  <code>()</code>  │
│ <b>vector_entry_point</b> │   <code>None</code> │
└────────────────────┴─────────────────────────────┘
```
</div>

## Delete local datasets

Local Minari datasets can be deleted by instantiating the following command, `minari delete DATASETS`. This command will also prompt a confirmation message to proceed with the deletion of the given datasets.

<div class="termy">

```console
// Delete datasets pen-cloned-v2 and door-human-v2
$ minari delete pen-cloned-v2 door-human-v2

                   <i>Delete local Minari datasets</i>
┌────────────────┬───────────┬────────────┬─────────┬───────────┐
│                │     <b>Total</b> │      <b>Total</b> │         │           │
│ <b>Name</b>           │  <b>Episodes</b> │      <b>Steps</b> │  <b>Author</b> │ <b>Email</b>     │
┡────────────────╇───────────╇────────────╇─────────╇───────────┩
│ <font color="#A1EFE4">door-human-v2</font>  │        <font color="#03AC13">25</font> │       <font color="#03AC13">6729</font> │ <font color="#FF00FF">Rodrigo</font> │ <font color="#FF00FF">rperezvic…</font>│
│ <font color="#A1EFE4">pen-cloned-v2</font>  │      <font color="#03AC13">3736</font> │     <font color="#03AC13">500000</font> │ <font color="#FF00FF">Rodrigo</font> │ <font color="#FF00FF">rperezvic…</font>│
└────────────────┴───────────┴────────────┴─────────┴───────────┘

# Are you sure you want to delete these local datasets? [y/N]:$ y

Dataset door-human-v2 deleted!
Dataset pen-cloned-v2 deleted!
```

</div>

## Combine datasets

Minari datasets can also be merged together into a single dataset with the following command, `minari combine DATASETS --dataset-name NEW_DATASET_NAME`.

<div class="termy">

```console
// Combine datasets pen-cloned-v2, pen-expert-v2 and pen-human-v2 into pen-all-v2.
$ minari combine pen-cloned-v2 pen-expert-v2 pen-human-v2 --dataset-name pen-all-v2

The datasets <font color="#03AC13">['pen-cloned-v2', 'pen-expert-v2', 'pen-human-v2']</font> were successfully combined into <font color="#A1EFE4">pen-all-v2</font>!
```
</div>

## Upload datasets

If you would like to upload your Minari dataset to a remote server, you can use the command the command `minari upload DATASETS --key-path PATH_STRING/KEY_FILE.json`.

```{eval-rst}
.. note::
   The progress bar shown in the example below is not currently implemented in the Minari package.
```

<div class="termy">

```console
// Upload datasets pen-cloned-v2 and door-human-v2 to remote server.
$ minari upload pen-cloned-v2 door-human-v2 --key-path /path-to-key/file-name.json

---> 100%
Dataset door-human-v2 uploaded!

---> 100%
Dataset pen-cloned-v2 uploaded!
```
</div>

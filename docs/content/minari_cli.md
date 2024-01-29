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
Minari version: 0.3.0

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
│ <font color="#A1EFE4">pen-cloned-v0</font> │      <font color="#03AC13">3736</font> │    <font color="#03AC13">500000</font> │ <font color="#FF00FF">Rodrigo</font> │ <font color="#FF00FF">rperezvic…</font>│
│ <font color="#A1EFE4">pen-expert-v0</font> │      <font color="#03AC13">4958</font> │    <font color="#03AC13">499206</font> │ <font color="#FF00FF">Rodrigo</font> │ <font color="#FF00FF">rperezvic…</font>│
│ <font color="#A1EFE4">pen-human-v0</font>  │        <font color="#03AC13">25</font> │      <font color="#03AC13">5000</font> │ <font color="#FF00FF">Rodrigo</font> │ <font color="#FF00FF">rperezvic…</font>│
└───────────────┴───────────┴───────────┴─────────┴───────────┘

// Show remote datasets.
$ minari list remote

                 <i>Minari datasets in Farama server</i>
┌────────────────┬───────────┬────────────┬─────────┬───────────┐
│                │     <b>Total</b> │      <b>Total</b> │         │           │
│ <b>Name</b>           │  <b>Episodes</b> │      <b>Steps</b> │  <b>Author</b> │ <b>Email</b>     │
┡────────────────╇───────────╇────────────╇─────────╇───────────┩
│ <font color="#A1EFE4">door-cloned-v0</font> │      <font color="#03AC13">4356</font> │    <font color="#03AC13">1000000</font> │ <font color="#FF00FF">Rodrigo</font> │ <font color="#FF00FF">rperezvic…</font>│
│ <font color="#A1EFE4">door-expert-v0</font> │      <font color="#03AC13">5000</font> │    <font color="#03AC13">1000000</font> │ <font color="#FF00FF">Rodrigo</font> │ <font color="#FF00FF">rperezvic…</font>│
│ <font color="#A1EFE4">door-human-v0</font>  │        <font color="#03AC13">25</font> │       <font color="#03AC13">6729</font> │ <font color="#FF00FF">Rodrigo</font> │ <font color="#FF00FF">rperezvic…</font>│
│ <font color="#A1EFE4">pen-cloned-v0</font>  │      <font color="#03AC13">3736</font> │     <font color="#03AC13">500000</font> │ <font color="#FF00FF">Rodrigo</font> │ <font color="#FF00FF">rperezvic…</font>│
│ <font color="#A1EFE4">pen-expert-v0</font>  │      <font color="#03AC13">4958</font> │     <font color="#03AC13">499206</font> │ <font color="#FF00FF">Rodrigo</font> │ <font color="#FF00FF">rperezvic…</font>│
│ <font color="#A1EFE4">pen-human-v0</font>   │        <font color="#03AC13">25</font> │       <font color="#03AC13">5000</font> │ <font color="#FF00FF">Rodrigo</font> │ <font color="#FF00FF">rperezvic…</font>│
└────────────────┴───────────┴────────────┴─────────┴───────────┘
```
</div>

## Download datasets

With the command `minari download DATASETS` you can download a group of datasets that are available in the remote Farama server. If the dataset name already exist locally, the Minari CLI will prompt you to override the
current content of the local dataset.

```{eval-rst}

.. note::
   The download is aborted if the remote dataset is not compatible with your local installed Minari version or through a warning if the dataset already exists locally. To perform a force download add :code:`--force` or :code:`-f` to the download command.

```

<div class="termy">

```console
// Download datasets pen-cloned-v0 and door-human-v0
$ minari download pen-cloned-v0 door-human-v0

Downloading pen-expert-v0 from Farama servers...

   * Downloading data file 'pen-expert-v0/data/main_data.hdf5' ...

---> 100%

Dataset pen-expert-v0 downloaded to <path-to-local-datasets>/.minari/datasets/pen-expert-v0

Downloading door-human-v0 from Farama servers...

   * Downloading data file 'door-human-v0/data/main_data.hdf5' ...

---> 100%

Dataset door-human-v0 downloaded to <path-to-local-datasets>/.minari/datasets/pen-expert-v0

```
</div>

## Delete local datasets

Local Minari datasets can be deleted by instantiating the following command, `minari delete DATASETS`. This command will also prompt a confirmation message to proceed with the deletion of the given datasets.

<div class="termy">

```console
// Delete datasets pen-cloned-v0 and door-human-v0
$ minari delete pen-cloned-v0 door-human-v0

                   <i>Delete local Minari datasets</i>
┌────────────────┬───────────┬────────────┬─────────┬───────────┐
│                │     <b>Total</b> │      <b>Total</b> │         │           │
│ <b>Name</b>           │  <b>Episodes</b> │      <b>Steps</b> │  <b>Author</b> │ <b>Email</b>     │
┡────────────────╇───────────╇────────────╇─────────╇───────────┩
│ <font color="#A1EFE4">door-human-v0</font>  │        <font color="#03AC13">25</font> │       <font color="#03AC13">6729</font> │ <font color="#FF00FF">Rodrigo</font> │ <font color="#FF00FF">rperezvic…</font>│
│ <font color="#A1EFE4">pen-cloned-v0</font>  │      <font color="#03AC13">3736</font> │     <font color="#03AC13">500000</font> │ <font color="#FF00FF">Rodrigo</font> │ <font color="#FF00FF">rperezvic…</font>│
└────────────────┴───────────┴────────────┴─────────┴───────────┘

# Are you sure you want to delete these local datasets? [y/N]:$ y

Dataset door-human-v0 deleted!
Dataset pen-cloned-v0 deleted!
```

</div>

## Upload datasets

If you would like to upload your own Minari dataset to the remote server please get in touch with the Farama team at [contact@farama.org](mailto:contact@farama.org). We will share with you a json encryption key file to give you permission to upload data to our GCP bucket. Then you can upload your dataset with the command `minari upload DATASETS --key-path PATH_STRING/KEY_FILE.json`.

```{eval-rst}
.. note::
   The progress bar shown in the example below is not currently implemented in the Minari package.
```

<div class="termy">

```console
// Upload datasets pen-cloned-v0 and door-human-v0 to Farama server.
$ minari upload pen-cloned-v0 door-human-v0 --key-path /path-to-key/file-name.json

---> 100%
Dataset door-human-v0 uploaded!

---> 100%
Dataset pen-cloned-v0 uploaded!
```
</div>

## Combine datasets

Minari datasets can also be merged together into a single dataset with the following command, `minari combine DATASETS --dataset-name NEW_DATASET_NAME`.

<div class="termy">

```console
// Combine datasets pen-cloned-v0, pen-expert-v0 and pen-human-v0 into pen-all-v0.
$ minari combine pen-cloned-v0 pen-expert-v0 pen-human-v0 --dataset-name pen-all-v0

The datasets <font color="#03AC13">['pen-cloned-v0', 'pen-expert-v0', 'pen-human-v0']</font> were successfully combined into <font color="#A1EFE4">pen-all-v0</font>!
```
</div>

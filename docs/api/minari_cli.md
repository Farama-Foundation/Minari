---
title: Minari CLI
---

# Minari CLI

Minari is also packaged with some useful CLI commands. The CLI tool is build on top off [Typer](https://typer.tiangolo.com/) and it will be available when installing Minari.


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


The Minari CLI gives access to most of the package functions through the command-line
## List datasets

```
minari list local
```

```
minari list remote
```
## Download datasets

## Delete local datasets

```
minari delete door
```

## Upload datasets
## Combine datasets
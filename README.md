[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) 


<p align="center">
    <img src="minari-text.png" width="500px"/>
</p>

Minari is a Python library for conducting research in offline reinforcement learning, akin to an offline version of Gymnasium or an offline RL version of HuggingFace's datasets library. This library is currently in beta.

The documentation website is at [minari.farama.org](https://minari.farama.org/main/). We also have a public discord server (which we use for Q&A and to coordinate development work) that you can join here: https://discord.gg/bnJ6kubTg6.

Note: Minari was previously developed under the name Kabuki.


## Installation
To install Minari from [PyPI](https://pypi.org/project/minari/):
```
pip install minari
```

Note that currently Minari is under a beta release. If you'd like to start testing or contribute to Minari please install this project from source with: 

```bash
git clone https://github.com/Farama-Foundation/Minari.git
cd Minari
pip install -e .
```

## Getting Started

For an introduction to Minari, see [Basic Usage](https://minari.farama.org/main/content/basic_usage/). To create new datasets using Minari, see our [Pointmaze D4RL Dataset](https://minari.farama.org/main/tutorials/dataset_creation/point_maze_dataset/) tutorial, which re-creates the Maze2D datasets from [D4RL](https://github.com/Farama-Foundation/D4RL).

## API 

To check available remote datasets:

```python
import minari

minari.list_remote_datasets()
```

To check available local datasets:

```python
import minari

minari.list_local_datasets()
```

To download a dataset:

```python
import minari

minari.download_dataset("door-cloned-v1")
```

To load a dataset:

```python
import minari

dataset = minari.load_dataset("door-cloned-v1")
```

## Project Maintainers
Main Contributors: [Rodrigo Perez-Vicente](https://github.com/rodrigodelazcano), [Omar Younis](https://github.com/younik), [John Balis](https://github.com/balisujohn)

Maintenance for this project is also contributed by the broader Farama team: [farama.org/team](https://farama.org/team).

___

_Minari is a shortening of Minarai, the Japanese word for "learning by observation"._

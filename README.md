[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


<p align="center">
    <img src="minari-text.png" width="500px"/>
</p>

Minari is a Python library for conducting research in offline reinforcement learning, akin to an offline version of Gymnasium or an offline RL version of HuggingFace's datasets library.

The documentation website is at [minari.farama.org](https://minari.farama.org/main/). We also have a public discord server (which we use for Q&A and to coordinate development work) that you can join here: https://discord.gg/bnJ6kubTg6.


## Installation
To install Minari from [PyPI](https://pypi.org/project/minari/):
```bash
pip install minari
```

This will install the minimum required dependencies. Additional dependencies will be prompted for installation based on your use case. To install all dependencies at once, use:
```bash
pip install "minari[all]"
```

If you'd like to start testing or contribute to Minari please install this project from source with:

```
git clone https://github.com/Farama-Foundation/Minari.git
cd Minari
pip install -e ".[all]"
```

## Command Line API

To check available remote datasets:

```bash
minari list remote
```

To download a dataset:

```bash
minari download D4RL/door/human-v2
```

To check available local datasets:

```bash
minari list local
```
To show the details of a dataset:

```bash
minari show D4RL/door/human-v2
```

For the list of commands:
```bash
minari --help
```

## Basic Usage

### Reading a Dataset

```python
import minari

dataset = minari.load_dataset("D4RL/door/human-v2")

for episode_data in dataset.iterate_episodes():
    observations = episode_data.observations
    actions = episode_data.actions
    rewards = episode_data.rewards
    terminations = episode_data.terminations
    truncations = episode_data.truncations
    infos = episode_data.infos
    ...
```

### Writing a Dataset

```python
import minari
import gymnasium as gym
from minari import DataCollector


env = gym.make('FrozenLake-v1')
env = DataCollector(env)

for _ in range(100):
    env.reset()
    done = False
    while not done:
        action = env.action_space.sample()  # <- use your policy here
        obs, rew, terminated, truncated, info = env.step(action)
        done = terminated or truncated

dataset = env.create_dataset("frozenlake/test-v0")
```

For other examples, see [Basic Usage](https://minari.farama.org/main/content/basic_usage/). For a complete tutorial on how to create new datasets using Minari, see our [Pointmaze D4RL Dataset](https://minari.farama.org/main/tutorials/dataset_creation/point_maze_dataset/) tutorial, which re-creates the Maze2D datasets from [D4RL](https://github.com/Farama-Foundation/D4RL).

## Training Libraries Integrating Minari

 - [TorchRL](https://github.com/pytorch/rl)
 - [d3rlpy](https://github.com/takuseno/d3rlpy)
 - [AgileRL](https://github.com/AgileRL/AgileRL)


## Citation
If you use Minari, please consider citing it:

```
@software{minari,
	author = {Younis, Omar G. and Perez-Vicente, Rodrigo and Balis, John U. and Dudley, Will and Davey, Alex and Terry, Jordan K},
	doi = {10.5281/zenodo.13767625},
	month = sep,
	publisher = {Zenodo},
	title = {Minari},
	url = {https://doi.org/10.5281/zenodo.13767625},
	version = {0.5.0},
	year = 2024,
	bdsk-url-1 = {https://doi.org/10.5281/zenodo.13767625}
}
```



___

_Minari is a shortening of Minarai, the Japanese word for "learning by observation"._

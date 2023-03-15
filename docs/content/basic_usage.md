---
title: Basic Usage
---
# Basic Usage

Minari is a standard dataset hosting interface for Offline Reinforcement Learning applications. Minari is compatible with most of the RL environments that follow the Gymnasium API and facilitates Offline RL dataset handling by providing data collection, dataset hosting, and dataset sampling capabilities.
## Collecting Data

Minari abstracts the data collection process from the environment. This is achieved by using the `DataCollectorV0` wrapper which stores the environment data in internal memory buffers before storing the dataset into disk. The `DataCollectorV0` wrapper can also perform caching by scheduling the amount of episodes or steps that are stored in-memory before saving the data in a temporary Minari dataset.

The wrapper can be initialized as follows:

```python
import minari
import gymnasium as gym

env = gym.make('EnvID')
env = minari.DataCollectorV0(env, record_infos=True, max_buffer_steps=100000)
```

## Create Minari Dataset

### Checkpoint Minari Dataset
### Combine Minari Datasets

## Using Minari Datasets
### Recover Gymnasium Environment

### Sampling Episodes from Minari Dataset


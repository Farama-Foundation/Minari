<p align="center">
    <img src="minari-text.png" width="500px"/>
</p>

Minari is the new name of this library. Minari used to be called Kabuki.

Minari is intended to be a Python library for conducting research in offline reinforcement learning, akin to an offline version of Gymnasium or an offline RL version of HuggingFace's datasets library. The goal is to release a fully working beta in late November or early December.

We have a public discord server (which we also use to coordinate development work) that you can join here: https://discord.gg/jfERDCSw.


## Installation
`pip install numpy cython`

`pip install git+https://github.com/Farama-Foundation/Minari.git`

## Downloading datasets

```python
import minari

dataset = minari.download_dataset("LunarLander_v2_remote-test-dataset")
```

## Recreating Gymnasium environments (Coming very soon!)

```python
import gymnasium as gym

env = gym.make(deserialise_spec_stack(json.loads(dataset.environment_stack)))
```

## Uploading datasets

```python
dataset.save(
    ".datasets/LunarLander-v2-test_dataset.hdf5"
)  # todo: abstract away parent directory and hdf5 extension
dataset = minari.upload_dataset("LunarLander_v2_test-dataset")
```


## Saving to dataset format
It is not the aim of Minari to insist that you use a certain buffer implementation. However, in order to maintain standardisation across the library, we have a standardised format, the `MinariDataset` class, for saving replay buffers to file. 

This converter will have tests to ensure formatting standards

## Checking available remote datasets

```python
import minari

minari.list_remote_datasets()
```

## Checking available local datasets
```python
import minari
minari.list_local_datasets()
```



___

_Minari is a shortening of Minarai, the Japanese word for "learning by observation"._

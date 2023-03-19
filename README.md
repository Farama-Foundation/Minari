<p align="center">
    <img src="minari-text.png" width="500px"/>
</p>

Minari is the new name of this library. Minari used to be called Kabuki.

Minari is intended to be a Python library for conducting research in offline reinforcement learning, akin to an offline version of Gymnasium or an offline RL version of HuggingFace's datasets library. This library is currently in beta.

More details about the features that Minari supports can be read in the documentation at https://minari.farama.org/. We also have a public discord server (which we use for Q&A and to coordinate development work) that you can join here: https://discord.gg/jfERDCSw.


## Installation

Currently the beta release is under development. If you'd like to start testing or contribute to Minari please install this project from source with: 

```bash
git clone https://github.com/Farama-Foundation/Minari.git
cd Minari
pip install -e .
```


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

## Downloading datasets

```python
import minari

dataset = minari.download_dataset("LunarLander_v2_remote-test-dataset")
```
___

_Minari is a shortening of Minarai, the Japanese word for "learning by observation"._

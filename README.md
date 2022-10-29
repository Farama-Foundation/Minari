<p align="center">
    <img src="kabuki-text.png" width="500px"/>
</p>
Kabuki is intended to be a Python library for conducting research in offline reinforcement learning, akin to an offline version of Gymnasium or an offline RL version of HuggingFace's datasets library. The goal is to release a fully working beta in late November or early December.

We have a public discord server (which we also use to coordinate development work) that you can join here: https://discord.gg/jfERDCSw if you're interested in it.


## Installation
`pip install numpy cython`

`pip install git+https://github.com/Farama-Foundation/Kabuki.git@WD/MDPDataset`

## Downloading datasets
```python
import kabuki
dataset = kabuki.retrieve_dataset("LunarLander-v2-test_dataset.hdf5")
```

or



## Saving to MDPDatasets
It is not the aim of Kabuki to insist that you use a certain buffer implementation. However, in order to maintain standardisation across the library, we have a standardised format, the `MDPDataset` class, for saving replay buffers to file. 


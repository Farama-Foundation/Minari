# Kabuki Dataset generation.

This repo serves to demonstrate a workflow for creating
offline datasets compatible with Kabuki.

It uses dm-acme for online RL training. The data
is logged via EnvLogger, and the logged data is then post-processed
into an HDF5 file. However, this is only a proof of concept, and 
neither dm-acme nor EnvLogger constitutes part of the specification.

## Spec for the new HDF5 dataset in Kabuki
We describe the standard used in the HDF5 dataset used in D4RL-V2.
Unlike the old datasets used in previous versions of the D4RL datasets,
we now formalize a new standard for the new datasets.

The new datasets will continue to use HDF5 as the storage format. Although HDF5 files
do not naturally have the mechanisms for storing distinct episodes as separate entries,
it is a widely adopted a standard format that can be easily used in different
frameworks and languages.

The previous iterations of the D4RL datasets have some outstanding issues.
One notable issue is that terminal observations are not captured.
While the omission of terminal observations are not really problematic
for offline actor-critic algorithms such as CQL/BCQ/IQL etc.,
they pose issues for researchers 
who would like to work on offline imitation learning research.
It is known that proper handling of terminal transitions can significantly influence the 
performance of imitation learning algorithms.
Therefore, the terminal observations will be recorded in the dataset in the new version. The new datasets should capture as much information from the original environment as possible whenever possible.

In the new version, the dataset will follow the convention introduced by the RLDS project.
The agent's experience is stored in the dataset as a sequence of episodes consisting of a variable number of steps. The steps are stored as a flattened dictionary of arrays
in the state-action-reward (SAR) alignment. Concretely each step consists of

* is_first, is_last: indicating the observation for the step is the first/the last step of an episode.
* observation: observation for the step
* action: action taken after observing the `observation` of the step
* reward: reward obtained after applying the action in the step.
* is_terminal: indicating whether the observation is terminal (is_terminal = False indicates that the episode is truncated.)
* discount: discount factor at this step. This may be unfamiliar to gym.Env users but 
is consistent with the discount used in dm_env. In particular, discount = 0 indicates that
the *next* step is terminal and 1.0 otherwise.

Refer to https://github.com/google-research/rlds for a more detailed description.

## Generating datasets.
While HDF5 is used as the final format for storing benchmark datasets in D4RL,
HDF5 is not used as the format during the data collection process. In this repo,
we demonstrate using EnvLogger for recording the interactions made by an
RL agent during online learning. The logged experience will then be post-processed
(and potentially stitched with other datasets) to produce the final HDF5 files.
We provide `convert_dataset.py` to show how this can be done by converting
from EnvLogger's Riegeli file formats to a single HDF5 file.
Alternatively, we can also use EnvLogger's RLDS backend to generate an RLDS-compatible TensorFlow dataset and convert that to the HDF5 file.
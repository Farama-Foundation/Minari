# pyright: basic, reportGeneralTypeIssues=false

import base64
import json
import os

import gymnasium as gym
from gymnasium.utils.serialize_spec_stack import deserialise_spec_stack

import minari

dataset = minari.download_dataset("LunarLander_v2_remote-test-dataset")

print("*" * 60, " Dataset Structure")
print(f"Dataset attributes: {dataset.__dir__()}\n")
print(f"Episode attributes: {dataset.episodes[0].__dir__()}\n")
print(f"Transition attributes: {dataset.episodes[0].transitions[0].__dir__()}\n")

print("*" * 60, " Examples")
print(f"Shape of observations: {dataset.observations.shape}\n")
print(f"Return of third episode: {dataset.episodes[2].compute_return()}\n")
print(f"21st action of fifth episode: {dataset.episodes[4].transitions[20].action}\n")

reconstructed_environment = gym.make(
    deserialise_spec_stack(json.loads(dataset.environment_stack))
)

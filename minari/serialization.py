from __future__ import annotations

import json
import os
import warnings
from typing import Dict, List, Optional, Union

import gymnasium as gym
import h5py
import numpy as np
from gymnasium.envs.registration import EnvSpec




def serialize_space(space: gym.spaces.Space, to_string=True) -> Union[Dict, str]:
    if isinstance(space, gym.spaces.Box):
        result = {}
        result["type"] = "box"
        result["dtype"] = str(space.dtype)
        result["shape"] = list(space.shape)
        result[
            "low"
        ] = (
            space.low.tolist()
        )  # we have to use python float type to serialze the np.float32 types
        result["high"] = space.high.tolist()
    elif isinstance(space, gym.spaces.Discrete):
        result = {}
        result["type"] = "discrete"
        result["dtype"] = "int64"  # this seems to be hardcoded in Gymnasium
        result["start"] = int(
            space.start
        )  # we need to cast from np.int64 to python's int type in order to serialize
        result["n"] = int(space.n)
    elif isinstance(space, gym.spaces.Dict):
        result = {"type": "dict", "subspaces": {}}
        for key in space.spaces.keys():
            result["subspaces"][key] = serialize_space(
                space.spaces[key], to_string=False
            )
    elif isinstance(space, gym.spaces.Tuple):
        result = {"type": "tuple", "subspaces": []}
        for subspace in space.spaces:
            result["subspaces"].append(serialize_space(subspace, to_string=False))
    if to_string:
        return json.dumps(result)
    else:
        return result


def deserialize_space(space_dict, from_string=True):
    if from_string:
        space_dict = json.loads(space_dict)

    assert type(space_dict) == dict
    if space_dict["type"] == "tuple":
        subspaces = tuple(
            [
                deserialize_space(subspace, from_string=False)
                for subspace in space_dict["subspaces"]
            ]
        )
        return gym.spaces.Tuple(subspaces)
    elif space_dict["type"] == "dict":
        subspaces = {
            key: deserialize_space(space_dict["subspaces"][key], from_string=False)
            for key in space_dict["subspaces"]
        }
        return gym.spaces.Dict(subspaces)
    elif space_dict["type"] == "box":
        shape = tuple(space_dict["shape"])
        low = np.array(space_dict["low"])
        high = np.array(space_dict["high"])
        dtype = np.dtype(space_dict["dtype"])
        return gym.spaces.Box(low=low, high=high, shape=shape, dtype=dtype) # type: ignore
    elif space_dict["type"] == "discrete":
        n = space_dict["n"]
        start = space_dict["start"]
        return gym.spaces.Discrete(n=n, start=start)
    else:
        assert False, "big problem"

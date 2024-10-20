from __future__ import annotations

import json
from collections import defaultdict
from functools import singledispatch
from typing import Dict, Union

import numpy as np
from gymnasium import spaces


@singledispatch
def serialize_space(space: spaces.Space, to_string=True) -> Union[Dict, str]:
    raise NotImplementedError(f"No serialization method available for {space}")


@serialize_space.register(spaces.Box)
def _serialize_box(space: spaces.Box, to_string=True) -> Union[Dict, str]:
    result = {}
    result["type"] = "Box"
    result["dtype"] = str(space.dtype)
    result["shape"] = list(space.shape)
    # we have to use python float type to serialize the np.float32 types
    result["low"] = space.low.tolist()
    result["high"] = space.high.tolist()

    if to_string:
        result = json.dumps(result)
    return result


@serialize_space.register(spaces.Discrete)
def _serialize_discrete(space: spaces.Discrete, to_string=True) -> Union[Dict, str]:
    result = {}
    result["type"] = "Discrete"
    result["dtype"] = "int64"  # this seems to be hardcoded in Gymnasium
    # we need to cast from np.int64 to python's int type in order to serialize
    result["start"] = int(space.start)
    result["n"] = int(space.n)

    if to_string:
        result = json.dumps(result)
    return result


@serialize_space.register(spaces.MultiDiscrete)
def _serialize_multi_discrete(
    space: spaces.MultiDiscrete, to_string=True
) -> Union[Dict, str]:
    result = {}
    result["type"] = "MultiDiscrete"
    result["dtype"] = str(space.dtype)
    result["nvec"] = space.nvec.tolist()
    result["start"] = space.start.tolist()

    if to_string:
        result = json.dumps(result)
    return result


@serialize_space.register(spaces.MultiBinary)
def _serialize_multi_binary(
    space: spaces.MultiBinary, to_string=True
) -> Union[Dict, str]:
    result = {"type": "MultiBinary", "n": space.n}

    if to_string:
        result = json.dumps(result)
    return result


@serialize_space.register(spaces.Dict)
def _serialize_dict(space: spaces.Dict, to_string=True) -> Union[Dict, str]:
    result = {"type": "Dict", "subspaces": {}}

    for key in space.spaces.keys():
        result["subspaces"][key] = serialize_space(space.spaces[key], to_string=False)

    if to_string:
        result = json.dumps(result)
    return result


@serialize_space.register(spaces.Tuple)
def _serialize_tuple(space: spaces.Tuple, to_string=True) -> Union[Dict, str]:
    result = {"type": "Tuple", "subspaces": []}

    for subspace in space.spaces:
        result["subspaces"].append(serialize_space(subspace, to_string=False))

    if to_string:
        return json.dumps(result)
    else:
        return result


@serialize_space.register(spaces.Text)
def _serialize_text(space: spaces.Text, to_string=True) -> Union[Dict, str]:
    result = {
        "type": "Text",
        "max_length": space.max_length,
        "min_length": space.min_length,
        "charset": space.characters,
    }

    if to_string:
        return json.dumps(result)
    else:
        return result


class type_value_dispatch:
    def __init__(self, func) -> None:
        self.registry = defaultdict(func)

    def register(self, type: str):
        def decorator(method):
            self.registry[type] = method
            return method

        return decorator

    def __call__(self, space_dict: Union[Dict, str]) -> spaces.Space:
        if not isinstance(space_dict, Dict):
            space_dict = json.loads(space_dict)

        assert isinstance(space_dict, Dict)
        return self.registry[space_dict["type"]](space_dict)


@type_value_dispatch
def deserialize_space(space_dict: Dict) -> spaces.Space:
    raise NotImplementedError(
        f"No deserialization method available for {space_dict['type']}"
    )


@deserialize_space.register("Tuple")
def _deserialize_tuple(space_dict: Dict) -> spaces.Tuple:
    assert space_dict["type"] == "Tuple"
    subspaces = tuple(
        deserialize_space(subspace) for subspace in space_dict["subspaces"]
    )
    return spaces.Tuple(subspaces)


@deserialize_space.register("Dict")
def _deserialize_dict(space_dict: Dict) -> spaces.Dict:
    assert space_dict["type"] == "Dict"
    subspaces = {
        key: deserialize_space(space_dict["subspaces"][key])
        for key in space_dict["subspaces"]
    }
    return spaces.Dict(subspaces)


@deserialize_space.register("Box")
def _deserialize_box(space_dict: Dict) -> spaces.Box:
    assert space_dict["type"] == "Box"
    shape = tuple(space_dict["shape"])
    dtype = space_dict["dtype"]
    low = np.array(space_dict["low"], dtype=dtype)
    high = np.array(space_dict["high"], dtype=dtype)
    return spaces.Box(low=low, high=high, shape=shape, dtype=dtype)


@deserialize_space.register("Discrete")
def _deserialize_discrete(space_dict: Dict) -> spaces.Discrete:
    assert space_dict["type"] == "Discrete"
    n = space_dict["n"]
    start = space_dict["start"]
    return spaces.Discrete(n=n, start=start)


@deserialize_space.register("MultiDiscrete")
def _deserialize_multi_discrete(space_dict: Dict) -> spaces.MultiDiscrete:
    assert space_dict["type"] == "MultiDiscrete"
    return spaces.MultiDiscrete(
        nvec=space_dict["nvec"],
        dtype=space_dict["dtype"],
        start=space_dict["start"],
    )


@deserialize_space.register("MultiBinary")
def _deserialize_multi_binary(space_dict: Dict) -> spaces.MultiBinary:
    assert space_dict["type"] == "MultiBinary"
    return spaces.MultiBinary(n=space_dict["n"])


@deserialize_space.register("Text")
def _deserialize_text(space_dict: Dict) -> spaces.Text:
    assert space_dict["type"] == "Text"
    return spaces.Text(
        max_length=space_dict["max_length"],
        min_length=space_dict["min_length"],
        charset=space_dict["charset"],
    )

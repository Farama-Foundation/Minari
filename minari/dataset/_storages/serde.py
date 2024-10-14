from __future__ import annotations

import json
from functools import partial
from typing import Dict, Any

import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()

        return super().default(obj)

try:
    import orjson
    dumps = partial(orjson.dumps, option=orjson.OPT_SERIALIZE_NUMPY)
    loads = orjson.loads
except ImportError:
    loads = json.loads
    dumps = partial(json.dumps,cls=NumpyEncoder)

def serialize_dict(data: Dict[str, Any]) -> bytes:
    return dumps(data)


def deserialize_dict(serialized_data: str) -> Dict[str, Any]:
    return orjson.loads(serialized_data)

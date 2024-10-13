from __future__ import annotations

import json
from typing import Dict, Any

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return super().default(obj)


def serialize_dict(data: Dict[str, Any]) -> str:
    return json.dumps(data, cls=NumpyEncoder)


def deserialize_dict(serialized_data: str) -> Dict[str, Any]:
    return json.loads(serialized_data)

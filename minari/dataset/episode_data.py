from dataclasses import dataclass
from typing import Any, Dict, List, Union

import numpy as np


@dataclass(frozen=True)
class EpisodeData:
    """Contains the datasets data for a single episode."""

    id: int
    observations: Any
    actions: Any
    rewards: np.ndarray
    terminations: np.ndarray
    truncations: np.ndarray
    infos: Union[Dict, List]  # dict is for backwards compatibility

    def __len__(self) -> int:
        return len(self.rewards)

    def __repr__(self) -> str:
        if isinstance(self.infos, dict):
            infos_repr = f"infos=dict with the following keys: {list(self.infos.keys())}"
        elif isinstance(self.infos, list):
            infos_repr = (f"infos=list of dicts with the following keys: "
                          f"{set(key for d in self.infos for key in d.keys())}")
        elif self.infos is None:
            infos_repr = "infos=None"
        else:
            raise ValueError(f"Unexpected type for infos: {type(self.infos)}")
        return (
            "EpisodeData("
            f"id={self.id}, "
            f"total_steps={len(self)}, "
            f"observations={EpisodeData._repr_space_values(self.observations)}, "
            f"actions={EpisodeData._repr_space_values(self.actions)}, "
            f"rewards=ndarray of {len(self.rewards)} floats, "
            f"terminations=ndarray of {len(self.terminations)} bools, "
            f"truncations=ndarray of {len(self.truncations)} bools, "
            f"{infos_repr}"            
            ")"
        )

    @staticmethod
    def _repr_space_values(value):
        if isinstance(value, np.ndarray):
            return f"ndarray of shape {value.shape} and dtype {value.dtype}"
        elif isinstance(value, dict):
            reprs = [
                f"{k}: {EpisodeData._repr_space_values(v)}" for k, v in value.items()
            ]
            dict_repr = ", ".join(reprs)
            return "{" + dict_repr + "}"
        elif isinstance(value, tuple):
            reprs = [EpisodeData._repr_space_values(v) for v in value]
            values_repr = ", ".join(reprs)
            return "(" + values_repr + ")"
        else:
            return repr(value)

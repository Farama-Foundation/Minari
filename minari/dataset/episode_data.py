from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True)
class EpisodeData:
    """Contains the datasets data for a single episode."""

    id: int
    seed: Optional[int]
    total_steps: int
    observations: Any
    actions: Any
    rewards: np.ndarray
    terminations: np.ndarray
    truncations: np.ndarray
    infos: dict

    def __repr__(self) -> str:
        return (
            "EpisodeData("
            f"id={repr(self.id)}, "
            f"seed={repr(self.seed)}, "
            f"total_steps={self.total_steps}, "
            f"observations={EpisodeData._repr_space_values(self.observations)}, "
            f"actions={EpisodeData._repr_space_values(self.actions)}, "
            f"rewards=ndarray of {len(self.rewards)} floats, "
            f"terminations=ndarray of {len(self.terminations)} bools, "
            f"truncations=ndarray of {len(self.truncations)} bools, "
            f"infos=dict with the following keys: {list(self.infos.keys())}"
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

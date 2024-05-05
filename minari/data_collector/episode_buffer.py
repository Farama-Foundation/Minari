from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

import jax.tree_util as jtu

from minari.data_collector import StepData


@dataclass(frozen=True)
class EpisodeBuffer:
    """Contains the data of a single episode."""

    id: Optional[int] = None
    seed: Optional[int] = None
    observations: Union[None, list, dict, tuple] = None
    actions: Union[None, list, dict, tuple] = None
    rewards: list = field(default_factory=list)
    terminations: list = field(default_factory=list)
    truncations: list = field(default_factory=list)
    infos: Optional[dict] = None

    def add_step_data(self, step_data: StepData) -> EpisodeBuffer:
        """Add step data dictionary to episode buffer.

        Args:
            step_data (StepData): dictionary with data for a single step

        Returns:
            EpisodeBuffer: episode buffer with appended data
        """

        def _append(data, buffer):
            if isinstance(buffer, list):
                buffer.append(data)
                return buffer
            else:
                return [buffer, data]

        observations = step_data["observations"]
        if self.observations is not None:
            observations = jtu.tree_map(
                _append, step_data["observations"], self.observations
            )
        actions = step_data["actions"]
        if self.actions is not None:
            actions = jtu.tree_map(_append, step_data["actions"], self.actions)
        infos = step_data["infos"]
        if self.infos is not None:
            infos = jtu.tree_map(_append, step_data["infos"], self.infos)
        self.rewards.append(step_data["rewards"])
        self.terminations.append(step_data["terminations"])
        self.truncations.append(step_data["truncations"])

        return EpisodeBuffer(
            id=self.id,
            seed=self.seed,
            observations=observations,
            actions=actions,
            rewards=self.rewards,
            terminations=self.terminations,
            truncations=self.truncations,
            infos=infos,
        )

    def __len__(self) -> int:
        return len(self.rewards)

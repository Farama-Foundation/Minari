from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

from minari.dataset.step_data import StepData


@dataclass(frozen=True)
class EpisodeBuffer:
    """Contains the data of a single episode."""

    id: Optional[int] = None
    seed: Optional[int] = None
    options: Optional[dict] = None
    observations: Union[None, list, dict, tuple] = None
    actions: Union[None, list, dict, tuple] = None
    rewards: list = field(default_factory=list)
    terminations: list = field(default_factory=list)
    truncations: list = field(default_factory=list)
    infos: Optional[Union[dict, list]] = None

    def add_step_data(self, step_data: StepData, infos_format=None) -> EpisodeBuffer:
        """Add step data dictionary to episode buffer.

        Args:
            step_data (StepData): dictionary with data for a single step
            infos_format (str): format of the infos data. Can be "dict" or "list"

        Returns:
            EpisodeBuffer: episode buffer with appended data
        """
        try:
            import jax.tree_util as jtu
        except ImportError:
            raise ImportError(
                'jax is not installed. Please install it using `pip install "minari[create]"`'
            )

        def _append(data, buffer):
            if isinstance(buffer, list):
                buffer.append(data)
                return buffer
            else:
                return [buffer, data]

        observations = step_data["observation"]
        if self.observations is not None:
            observations = jtu.tree_map(
                _append, step_data["observation"], self.observations
            )

        if self.actions is None:
            actions = jtu.tree_map(lambda x: [x], step_data["action"])
        else:
            actions = jtu.tree_map(_append, step_data["action"], self.actions)

        infos_format = infos_format or "dict"
        if self.infos is None:
            infos = jtu.tree_map(lambda x: [x], step_data["info"]) if infos_format == "dict" else [step_data["info"]]
        else:
            if isinstance(self.infos, dict):
                infos = jtu.tree_map(_append, step_data["info"], self.infos)
            elif isinstance(self.infos, list):
                self.infos.append(step_data["info"])
                infos = self.infos
            else:
                raise ValueError(f"Unexpected type for infos: {type(self.infos)}")


        self.rewards.append(step_data["reward"])
        self.terminations.append(step_data["termination"])
        self.truncations.append(step_data["truncation"])

        return EpisodeBuffer(
            id=self.id,
            seed=self.seed,
            options=self.options,
            observations=observations,
            actions=actions,
            rewards=self.rewards,
            terminations=self.terminations,
            truncations=self.truncations,
            infos=infos,
        )

    def __len__(self) -> int:
        """Buffer length."""
        return len(self.rewards)

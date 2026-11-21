from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Union

import numpy as np

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
    infos: Optional[dict] = None

    def add_step_data(self, step_data: StepData) -> EpisodeBuffer:
        """Add step data dictionary to episode buffer.

        The info dictionary can have different keys from one step to another. The
        values that a step does not provide are filled with ``NaN``, so that all
        the infos of the episode keep the same number of entries.

        Args:
            step_data (StepData): dictionary with data for a single step

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

        if self.infos is None:
            infos = jtu.tree_map(lambda x: [x], step_data["info"])
        else:
            info, buffered_infos = step_data["info"], self.infos
            if not _have_same_keys(info, buffered_infos):
                info, buffered_infos = self._pad_missing_infos(info, buffered_infos)
            infos = jtu.tree_map(_append, info, buffered_infos)

        self.rewards.append(step_data["reward"])
        self.terminations.append(step_data["terminated"])
        self.truncations.append(step_data["truncated"])

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

    def _pad_missing_infos(
        self, info: Any, buffer: Any, path: str = ""
    ) -> tuple[Any, Any]:
        """Give the info of a step and the buffered infos the same dictionary keys.

        A key that the step does not provide gets a ``NaN`` entry for this step,
        and a key that the buffer does not know yet gets a ``NaN`` entry for each
        info that is already stored.
        """
        if not (isinstance(info, dict) and isinstance(buffer, dict)):
            return info, buffer

        import jax.tree_util as jtu

        def is_series(x):
            return isinstance(x, list)

        # Until the first step is added, the buffer holds the reset info as it is.
        has_series = len(self.rewards) > 0
        info, buffer = dict(info), dict(buffer)
        for key in [*buffer, *(key for key in info if key not in buffer)]:
            key_path = f"{path}/{key}" if path else str(key)
            if key not in info:
                if has_series:
                    info[key] = jtu.tree_map(
                        lambda series: _missing_like(series[-1], key_path),
                        buffer[key],
                        is_leaf=is_series,
                    )
                else:
                    info[key] = jtu.tree_map(
                        lambda value: _missing_like(value, key_path), buffer[key]
                    )
            elif key not in buffer:
                num_entries = 1
                if has_series:
                    # One info is stored for each observation, unless the infos
                    # that are already buffered show a different length.
                    series = jtu.tree_leaves(self.infos, is_leaf=is_series)
                    num_entries = len(series[0]) if series else len(self.rewards) + 1
                buffer[key] = jtu.tree_map(
                    lambda value: [_missing_like(value, key_path)] * num_entries,
                    info[key],
                )
            else:
                info[key], buffer[key] = self._pad_missing_infos(
                    info[key], buffer[key], key_path
                )

        return info, buffer

    def __len__(self) -> int:
        """Buffer length."""
        return len(self.rewards)


def _have_same_keys(info: Any, buffer: Any) -> bool:
    """Check that two nested dictionaries have the same keys at every level."""
    if not (isinstance(info, dict) and isinstance(buffer, dict)):
        return True
    return info.keys() == buffer.keys() and all(
        _have_same_keys(value, buffer[key]) for key, value in info.items()
    )


def _missing_like(value: Any, key_path: str) -> Any:
    """Return the ``NaN`` filling for a step in which ``value`` is not provided."""
    dtype = np.asarray(value).dtype
    if dtype.kind in "biu":
        # Integers and booleans can't hold NaN.
        dtype = np.dtype(np.float64)
    elif dtype.kind not in "fc":
        raise ValueError(
            f"The info key '{key_path}' is not provided at every step and its "
            f"values of type {type(value).__name__} can't be filled with NaN. "
            "Only numeric infos can change from one step to another; use a "
            "StepDataCallback to give the infos the same structure at every step."
        )
    missing = np.full(np.shape(value), np.nan, dtype=dtype)
    return missing if isinstance(value, np.ndarray) else missing[()]

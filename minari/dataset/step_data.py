from typing import Any, Dict, Optional, SupportsFloat, TypedDict


class StepData(TypedDict):
    """Object containing data of a single environment step."""

    observation: Any
    action: Optional[Any]
    reward: Optional[SupportsFloat]
    terminated: Optional[bool]
    truncated: Optional[bool]
    info: Dict[str, Any]

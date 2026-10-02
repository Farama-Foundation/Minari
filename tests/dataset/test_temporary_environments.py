import json

import gymnasium as gym
import pytest

from minari.dataset.minari_storage import MinariStorage
from minari.serialization import serialize_space


class TemporaryEnvironment(gym.Env):
    observation_space = gym.spaces.Box(-1, 1, (1,))
    action_space = gym.spaces.Discrete(2)
    closed = []

    def close(self):
        self.closed.append(True)


@pytest.mark.parametrize("operation", ["new", "read"])
@pytest.mark.parametrize("missing", ["observation", "action", "both"])
def test_space_inference_closes_temporary_environment(
    tmp_path, monkeypatch, operation, missing
):
    closed = []
    monkeypatch.setattr(TemporaryEnvironment, "closed", closed)
    env_id = "MinariTemporaryEnvironment-v0"
    gym.register(
        env_id,
        entry_point=f"{__name__}:TemporaryEnvironment",
        order_enforce=False,
        disable_env_checker=True,
    )
    try:
        observation = (
            None
            if missing in ["observation", "both"]
            else TemporaryEnvironment.observation_space
        )
        action = (
            None if missing in ["action", "both"] else TemporaryEnvironment.action_space
        )
        if operation == "new":
            storage = MinariStorage.new(
                tmp_path,
                observation_space=observation,
                action_space=action,
                env_spec=gym.spec(env_id),
                data_format="arrow",
            )
        else:
            metadata = {"env_spec": gym.spec(env_id).to_json(), "data_format": "arrow"}
            if observation is not None:
                metadata["observation_space"] = serialize_space(observation)
            if action is not None:
                metadata["action_space"] = serialize_space(action)
            (tmp_path / "metadata.json").write_text(json.dumps(metadata))
            storage = MinariStorage.read(tmp_path)
        assert storage.observation_space == TemporaryEnvironment.observation_space
        assert storage.action_space == TemporaryEnvironment.action_space
        assert closed == [True]
    finally:
        gym.registry.pop(env_id, None)

import gymnasium as gym
import pytest
from gymnasium import spaces

from minari.utils import _generate_dataset_metadata, get_env_spec_dict


ENV_ID = "UtilsCloseCounting-v0"


class CloseCountingEnv(gym.Env):
    closed = 0

    def __init__(self):
        self.observation_space = spaces.Box(-1, 1, shape=(2,))
        self.action_space = spaces.Discrete(3)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 1.0, True, False, {}

    def close(self):
        CloseCountingEnv.closed += 1


@pytest.fixture
def close_counting_env_spec():
    gym.register(ENV_ID, entry_point=f"{__name__}:CloseCountingEnv")
    CloseCountingEnv.closed = 0
    try:
        yield gym.spec(ENV_ID)
    finally:
        gym.registry.pop(ENV_ID, None)


def test_get_env_spec_dict_closes_env(close_counting_env_spec):
    spec_dict = get_env_spec_dict(close_counting_env_spec)
    assert CloseCountingEnv.closed == 1
    assert "Observation Space" in spec_dict
    assert "Action Space" in spec_dict


@pytest.mark.parametrize(
    "scores",
    [
        {"ref_max_score": 10.0},
        {"expert_policy": lambda obs: 0},
    ],
)
def test_reference_score_env_is_closed(close_counting_env_spec, scores):
    metadata = _generate_dataset_metadata(
        dataset_id="dummy/test-v0",
        env_spec=close_counting_env_spec,
        eval_env=None,
        algorithm_name="test",
        author="test",
        author_email="test@example.com",
        code_permalink="https://example.com",
        ref_min_score=None,
        ref_max_score=scores.get("ref_max_score"),
        expert_policy=scores.get("expert_policy"),
        num_episodes_average_score=2,
        description="test",
        requirements=None,
    )
    assert CloseCountingEnv.closed == 1
    assert metadata["ref_min_score"] == 1.0
    expected_max_score = scores.get("ref_max_score", 1.0)
    assert metadata["ref_max_score"] == expected_max_score

from typing import Any, Dict, Optional
from typing_extensions import TypedDict

import gymnasium as gym
from gymnasium import spaces


class StepData(TypedDict):
    observations: Any
    actions: Optional[Any]
    rewards: Optional[Any]
    terminations: Optional[bool]
    truncations: Optional[bool]
    infos: Dict[str, Any]


STEP_DATA_KEYS = {
    "actions",
    "observations",
    "rewards",
    "truncations",
    "terminations",
}


class StepDataCallback:
    """Callback to create step data dictionary from the return data of each Gymnasium environment step.

    The current callback automatically detects observation/action spaces that need
    to be flatten before saving to HDF5 file (currently only supports Dict or Tuple
    Gymnasium spaces. Text, Sequence, and Graph are currently not compatible with
    Minari).

    This callback can be overridden to add extra environment information in each step or
    edit the observation, action, reward, termination, truncation, or info returns.
    """

    def __init__(self, env: gym.Env):
        self.env = env

        def check_flatten_space(space: gym.Space):
            """Check if space needs to be flatten or if it's not supported by Minari.

            Args:
                space: the Gymnasium space to be checked

            Returns:
                bool: True if space needs to be flatten before storing in HDF5 dataset. False otherwise.

            ValueError: If space is/contains Text, Sequence, or Graph space types
            """
            if isinstance(space, spaces.Dict):
                for s in space.spaces.values():
                    check_flatten_space(s)
                return True
            elif isinstance(space, spaces.Tuple):
                for s in space.spaces:
                    check_flatten_space(s)
                return True
            elif isinstance(
                self.env.observation_space, (spaces.Text, spaces.Sequence, spaces.Graph)
            ):
                ValueError(f"Minari doesn't support space of type {space}")
            else:
                return False

        # check if observation/action need to be flatten before saving to HDF5
        self.flatten_observation = check_flatten_space(self.env.observation_space)
        self.flatten_action = check_flatten_space(self.env.action_space)

    def __call__(
        self,
        env: gym.Env,
        obs: Any,
        info: Dict[str, Any],
        action: Optional[Any] = None,
        rew: Optional[Any] = None,
        terminated: Optional[bool] = None,
        truncated: Optional[bool] = None,
    ) -> StepData:
        """Callback method.

        The input arguments belong to a Gymnasium stepping transition: `obs, rew, terminated, truncated, info = env.step(action)`.
        Override this method to add additional keys or edit each environment's step returns. Additional nested dictionaries can be added to the returned step dictionary
        as follows:

        .. code::

            class CustomStepDataCallback(StepDataCallback):
                def __call__(self, env, **kwargs):
                    step_data = super().__call__(env, **kwargs)
                    step_data['environment_states'] = {}
                    step_data['environment_states']['pose'] = {}
                    step_data['environment_states']['pose']['position'] = env.position
                    step_data['environment_states']['pose']['orientation'] = env.orientation
                    step_data['environment_states']['velocity'] = env.velocity

                    return step_data

        The episode groups in the HDF5 file of this Minari dataset will contain a subgroup called `environment_states` with dataset `velocity` and another subgroup called `pose`
        with datasets `position` and `orientation`

        Args:
            env (gym.Env): current Gymnasium environment.
            obs (Any): observation returned by `env.step(action)`
            info (Dict): information dictionary returned by `env.step(action)`
            action (Optional[Any], optional): stepping action in `env.step(action)`. Defaults to None.
            rew (Optional[Any], optional): reward returned by `env.step(action)`. Defaults to None.
            terminated (Optional[Any], optional): terminated returned by `env.step(action)`. Defaults to None.
            truncated (Optional[Any], optional): truncated returned by `env.step(action)`. Defaults to None.

        Returns:
            Dict: dictionary step data. Must contain the keys in STEP_DATA_KEYS = {'actions', 'observations',
                    'rewards', 'terminations', 'truncations', 'infos'}. Additional key's can be added with nested dictionaries
        """
        if action is not None:
            # Flatten the actions
            if self.flatten_action:
                action = spaces.flatten(self.env.action_space, action)
        # Flatten the observations
        if self.flatten_observation:
            obs = spaces.flatten(self.env.observation_space, obs)

        step_data: StepData = {
            "actions": action,
            "observations": obs,
            "rewards": rew,
            "terminations": terminated,
            "truncations": truncated,
            "infos": info,
        }

        return step_data

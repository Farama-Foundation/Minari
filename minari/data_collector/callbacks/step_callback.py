from typing import Any, Dict, Optional

import gymnasium as gym

from minari.dataset.step_data import StepData


class StepDataCallback:
    """Callback to create step data dictionary from the return data of each Gymnasium environment step.

    This callback can be overridden to add extra environment information in each step or
    edit the observation, action, reward, termination, truncation, or info returns.
    """

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

        The Minari dataset will contain a dictionary called `environment_states` with `velocity` value and another dictionary `pose`
        with `position` and `orientation`

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
        step_data: StepData = {
            "action": action,
            "observation": obs,
            "reward": rew,
            "termination": terminated,
            "truncation": truncated,
            "info": info,
        }

        return step_data

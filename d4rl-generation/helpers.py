"""Shared helpers for rl_continuous experiments."""
from typing import Optional
from acme import wrappers
import dm_env
import gym
from acme.utils import loggers as acme_loggers
from absl import logging

_VALID_TASK_SUITES = ("gym", "control")


def make_environment(suite: str, task: str, seed=None) -> dm_env.Environment:
    """Makes the requested continuous control environment.
    Args:
      suite: One of 'gym' or 'control'.
      task: Task to load. If `suite` is 'control', the task must be formatted as
        f'{domain_name}:{task_name}'
    Returns:
      An environment satisfying the dm_env interface expected by Acme agents.
    """

    if suite not in _VALID_TASK_SUITES:
        raise ValueError(
            f"Unsupported suite: {suite}. Expected one of {_VALID_TASK_SUITES}"
        )

    if suite == "gym":
        env = gym.make(task)
        env.seed(seed)
        # Make sure the environment obeys the dm_env.Environment interface.
        env = wrappers.GymWrapper(env)

    elif suite == "control":
        # Load dm_suite lazily not require Mujoco license when not using it.
        from dm_control import suite as dm_suite  # pylint: disable=g-import-not-at-top

        domain_name, task_name = task.split(":")
        env = dm_suite.load(domain_name, task_name, task_kwargs={'random': seed})
        env = wrappers.ConcatObservationWrapper(env)

    # Wrap the environment so the expected continuous action spec is [-1, 1].
    # Note: this is a no-op on 'control' tasks.
    env = wrappers.CanonicalSpecWrapper(env, clip=True)
    env = wrappers.SinglePrecisionWrapper(env)
    return env


def get_default_logger_factory(workdir: str, save_data=True, time_delta: float = 1.0):
    """Create a custom logger factory for use in the experiment."""

    def logger_factory(label: str, steps_key: Optional[str] = None, task_id: int = 0):
        del steps_key, task_id

        print_fn = logging.info
        terminal_logger = acme_loggers.TerminalLogger(label=label, print_fn=print_fn)

        loggers = [terminal_logger]

        if save_data:
            loggers.append(acme_loggers.CSVLogger(workdir, label=label))

        # Dispatch to all writers and filter Nones and by time.
        logger = acme_loggers.Dispatcher(loggers, acme_loggers.to_numpy)
        logger = acme_loggers.NoneFilter(logger)
        logger = acme_loggers.TimeFilter(logger, time_delta)

        return logger

    return logger_factory

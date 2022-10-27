"""Example running SAC on continuous control tasks and generating data via EnvLogger."""

from absl import flags
from acme import specs
import tensorflow as tf
from acme.agents.jax import sac
from acme.agents.jax.sac import builder
import helpers
from absl import app
from acme.jax import experiments
import logged_experiment
import envlogger
from acme.utils import paths

FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "gym:HalfCheetah-v4", "What environment to run")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_integer("num_steps", 1_000_0, "Number of env steps to run.")
flags.DEFINE_integer("eval_every", 5_000, "How often to run evaluation.")
flags.DEFINE_integer("evaluation_episodes", 10, "Evaluation episodes.")
flags.DEFINE_string("workdir", None, "Evaluation episodes.")
flags.DEFINE_string("dataset_dir", None, "Where to save logged interaction")
flags.mark_flag_as_required("workdir")


def build_experiment_config():
    """Builds SAC experiment config which can be executed in different ways."""
    # Create an environment, grab the spec, and use it to create networks.

    suite, task = FLAGS.env_name.split(":", 1)
    environment = helpers.make_environment(suite, task)
    logger_factory = helpers.get_default_logger_factory(FLAGS.workdir)

    environment_spec = specs.make_environment_spec(environment)
    network_factory = lambda spec: sac.make_networks(
        spec, hidden_layer_sizes=(256, 256, 256)
    )

    # Construct the agent.
    config = sac.SACConfig(
        learning_rate=3e-4,
        n_step=1,
        min_replay_size=1000,
        target_entropy=sac.target_entropy_from_env_spec(environment_spec),
    )
    sac_builder = builder.SACBuilder(config)

    return experiments.ExperimentConfig(
        builder=sac_builder,
        environment_factory=lambda seed: helpers.make_environment(suite, task, seed),
        network_factory=network_factory,
        seed=FLAGS.seed,
        max_num_actor_steps=FLAGS.num_steps,
        logger_factory=logger_factory,
    )


def wrap_with_envlogger(env, dataset_dir):
    paths.process_path(dataset_dir, add_uid=False)
    return envlogger.EnvLogger(env, data_directory=dataset_dir)


def main(_):
    tf.config.set_visible_devices([], "GPU")
    config = build_experiment_config()
    if FLAGS.dataset_dir:
        make_envlogger = lambda env: wrap_with_envlogger(env, FLAGS.dataset_dir)
    else:
        make_envlogger = None
    logged_experiment.run_experiment(
        experiment=config,
        eval_every=FLAGS.eval_every,
        num_eval_episodes=FLAGS.evaluation_episodes,
        make_envlogger=make_envlogger,
    )


if __name__ == "__main__":
    app.run(main)

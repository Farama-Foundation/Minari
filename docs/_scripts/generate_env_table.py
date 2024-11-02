from absl import app, flags
from gymnasium.envs.registration import EnvSpec
from md_utils import dict_to_table

from minari.utils import get_env_spec_dict


FLAGS = flags.FLAGS
flags.DEFINE_string("env_spec", None, "Environment spec json file")
flags.DEFINE_string("file_name", None, "File name to save the md file")


def main(argv):
    del argv
    env_spec_dict = get_env_spec_dict(EnvSpec.from_json(FLAGS.env_spec))
    md_table = dict_to_table(env_spec_dict)
    with open(FLAGS.file_name, "w") as f:
        f.write(md_table)


if __name__ == "__main__":
    app.run(main)

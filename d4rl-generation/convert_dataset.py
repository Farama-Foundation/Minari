"""Convert dataset logged by the EnvLogger Riegelli backend to HDF5."""
from envlogger import reader
import numpy as np
import h5py
import tree
from absl import flags
from absl import app

_DATASET_DIR = flags.DEFINE_string("dataset_dir", None, "")
_OUTPUT_FILE = flags.DEFINE_string("output_file", "dataset.hdf5", "")
flags.mark_flag_as_required("dataset_dir")


def _convert_envlogger_episode_to_rlds_steps(episode):
    """Convert an episode of envlogger.StepData to RLDS compatible steps."""
    observations = np.stack([step.timestep.observation for step in episode])
    # RLDS uses the SAR alignment while envlogger uses ARS.
    # The following lines handle converting from the ARS to SAR alignment.
    actions = np.stack([step.action for step in episode[1:]])
    # Add dummy action to the last step containing the terminal observation.
    actions = np.concatenate(
        [actions, np.expand_dims(np.zeros_like(actions[0]), axis=0)]
    )
    # Add dummy reward to the last step containing the terminal observation.
    rewards = np.stack([step.timestep.reward for step in episode[1:]])
    rewards = np.concatenate(
        [rewards, np.expand_dims(np.zeros_like(rewards[0]), axis=0)]
    )
    # Add dummy discounts to the last step containing the terminal observation.
    discounts = np.stack([step.timestep.reward for step in episode[1:]])
    discounts = np.concatenate(
        [discounts, np.expand_dims(np.zeros_like(discounts[0]), axis=0)]
    )
    # the is_first/last/terminal flags are already aligned in ARS alignment.
    is_first = np.array([step.timestep.first() for step in episode])
    is_last = np.array([step.timestep.last() for step in episode])
    is_terminal = np.array(
        [step.timestep.last() and step.timestep.discount == 0.0 for step in episode]
    )
    return {
        "observation": observations,
        "action": actions,
        "reward": rewards,
        "discounnts": discounts,
        "is_first": is_first,
        "is_last": is_last,
        "is_terminal": is_terminal,
    }


def write_to_hdf5_file(episodes, filename):
    """Write episodes in EnvLogger format to an HDF5 file."""
    all_steps = []
    for episode in episodes:
        all_steps.append(_convert_envlogger_episode_to_rlds_steps(episode))
    all_steps = tree.map_structure(lambda *xs: np.concatenate(xs), *all_steps)
    f = h5py.File(filename, "w")
    for key in all_steps.keys():
        f.create_dataset(key, data=all_steps[key], compression="gzip")
    f.close()


def main(_):
    output_file = _OUTPUT_FILE.value
    with reader.Reader(data_directory=_DATASET_DIR.value) as r:
        print(r.observation_spec())
        print(r.metadata())
        write_to_hdf5_file(r.episodes, output_file)
    # Inspecting the created HDF5 file
    f = h5py.File(output_file, "r")
    for k in f:
        print(k, f[k].shape)
    f.close()


if __name__ == "__main__":
    app.run(main)

import os

import imageio
from absl import app, flags

import minari


FLAGS = flags.FLAGS
flags.DEFINE_string("dataset_id", None, "Dataset ID")
flags.DEFINE_string("path", None, "Path to save the gif")
flags.DEFINE_integer("num_frames", 512, "Number of frames in the gif")
flags.DEFINE_integer("fps", 32, "Frames per second in the gif")


def _space_at(values, index):
    if isinstance(values, dict):
        return {k: _space_at(v, index) for k, v in values.items()}
    elif isinstance(values, tuple):
        return (_space_at(v, index) for v in values)
    else:
        return values[index]


def generate_gif(dataset_id, path, num_frames=512, fps=32):
    dataset = minari.load_dataset(dataset_id)
    env = dataset.recover_environment(render_mode="rgb_array")
    images = []

    metadatas = dataset.storage.get_episode_metadata(dataset.episode_indices)
    for episode, episode_metadata in zip(dataset.iterate_episodes(), metadatas):
        seed = episode_metadata.get("seed")
        if episode.id == 0 and seed is None:
            raise ValueError("Cannot reproduce episodes with unknown seed.")

        env.reset(seed=seed, options=episode_metadata.get("options"))
        images.append(env.render())
        for step_id in range(len(episode)):
            act = _space_at(episode.actions, step_id)
            env.step(act)
            images.append(env.render())
            if len(images) > num_frames:
                env.close()
                gif_file = os.path.join(path, f"{dataset_id}.gif")
                imageio.mimsave(gif_file, images, fps=fps)
                return gif_file

    raise ValueError("There are not enough steps in the dataset.")


def main(argv):
    del argv
    generate_gif(FLAGS.dataset_id, FLAGS.path, FLAGS.num_frames, FLAGS.fps)


if __name__ == "__main__":
    app.run(main)

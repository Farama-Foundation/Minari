import os

import imageio

import minari


def _space_at(values, index):
    if isinstance(values, dict):
        return {k: _space_at(v, index) for k, v in values.items()}
    elif isinstance(values, tuple):
        return (_space_at(v, index) for v in values)
    else:
        return values[index]


def generate_gif(dataset_name, num_frames=256, fps=16):
    dataset = minari.load_dataset(dataset_name, download=True)
    env = dataset.recover_environment(render_mode="rgb_array")
    images = []
    ep_id = 0

    while len(images) < num_frames:
        episode = dataset[ep_id]
        if episode.seed is None:
            raise ValueError("Cannot reproduce episodes with unknown seed.")

        env.reset(seed=episode.seed)
        images.append(env.render())
        for step_id in range(episode.total_steps):
            act = _space_at(episode.actions, step_id)
            env.step(act)
            images.append(env.render())

        ep_id += 1

    gif_dir = os.path.join(os.path.dirname(__file__), "..", "datasets", "gifs")
    if not os.path.exists(gif_dir):
        os.makedirs(gif_dir)

    gif_path = os.path.join(gif_dir, f"{dataset_name}.gif")
    imageio.mimsave(gif_path, images, fps=fps)
    return gif_path

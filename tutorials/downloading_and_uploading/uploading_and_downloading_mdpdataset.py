import gymnasium as gym
from kabuki.dataset import MDPDataset
import numpy as np
import requests
import os
from google.cloud import storage

# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../../../../.config/gcloud/credentials.db"

num_episodes = 2

env = gym.make("LunarLander-v2", render_mode="rgb_array")
observation, info = env.reset(seed=42)

replay_buffer = {
    "episode": np.array([]),
    "observation": np.array([]),
    "action": np.array([]),
    "reward": np.array([]),
    "done": np.array([]),
}

for episode in range(num_episodes):
    observation, info = env.reset()
    terminated = False
    truncated = False
    while not terminated and not truncated:
        action = env.action_space.sample()  # User-defined policy function
        observation, reward, terminated, truncated, info = env.step(action)
        np.append(replay_buffer["episode"], episode)
        np.append(replay_buffer["observation"], observation)
        np.append(replay_buffer["action"], action)
        np.append(replay_buffer["reward"], reward)
        np.append(replay_buffer["done"], terminated)

env.close()

ds = MDPDataset(
    observations=replay_buffer["observation"],
    actions=replay_buffer["action"],
    rewards=replay_buffer["reward"],
    terminals=replay_buffer["done"],
)

ds.dump("test_dataset.hdf5")


from google.cloud import storage


def upload_blob(
    source_blob_name, project_id="dogwood-envoy-367012", bucket_name="kabuki-datasets"
):
    """Uploads a file to the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID of your GCS object

    storage_client = storage.Client(project_id)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.upload_from_filename(source_blob_name)

    print(f"File {source_blob_name} uploaded to {source_blob_name}.")


upload_blob("test_dataset.hdf5", "test_dataset.hdf5")

print("Uploaded to GCS")


def download_blob(
    source_blob_name,
    project_id="dogwood-envoy-367012",
    bucket_name="kabuki-datasets",
):
    """Downloads a blob from the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"

    # The ID of your GCS object
    # source_blob_name = "storage-object-name"

    # The path to which the file should be downloaded

    storage_client = storage.Client(project=project_id)

    bucket = storage_client.bucket(bucket_name)

    # Construct a client side representation of a blob.
    # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
    # any content from Google Cloud Storage. As we don't need additional data,
    # using `Bucket.blob` is preferred here.
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(source_blob_name)

    print(
        "Downloaded storage object {} from bucket {} to local file {}.".format(
            source_blob_name, bucket_name, source_blob_name
        )
    )


download_blob("test_dataset.hdf5", "test_dataset_received.hdf5")

print("Downloaded from GCS")

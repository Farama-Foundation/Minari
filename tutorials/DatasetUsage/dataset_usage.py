import base64
import os

import minari

GCP_DATASET_ADMIN = os.environ["GCP_DATASET_ADMIN"]

credentials_json = base64.b64decode(GCP_DATASET_ADMIN).decode("utf8").replace("'", '"')
with open("credentials.json", "w") as f:
    f.write(credentials_json)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./credentials.json"

dataset = minari.download_dataset("LunarLander_v2_remote-test-dataset")

print("*" * 60, " Dataset Structure")
print(f"Dataset attributes: {dataset.__dir__()}\n")
print(f"Episode attributes: {dataset.episodes[0].__dir__()}\n")
print(f"Transition attributes: {dataset.episodes[0].transitions[0].__dir__()}\n")

print("*" * 60, " Examples")
print(f"Shape of observations: {dataset.observations.shape}\n")
print(f"Return of third episode: {dataset.episodes[2].compute_return()}\n")
print(f"21st action of fifth episode: {dataset.episodes[4].transitions[20].action}\n")

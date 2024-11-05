import os
import minari
from minari.namespace import list_remote_namespaces

name = "D4RL/kithcen/complete-v1"
minari.download_dataset(name)

os.environ["MINARI_REMOTE"] = "hf://farama-minari"

try:

    minari.upload_dataset(name, token="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")

    print(minari.list_remote_datasets(latest_version=True).keys())
    print(list_remote_namespaces())

finally:
    del os.environ["MINARI_REMOTE"]

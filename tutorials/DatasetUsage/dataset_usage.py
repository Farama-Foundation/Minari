import kabuki

dataset = kabuki.download_dataset("LunarLander-v2-remote_test_dataset")

print(f"{dataset}")
print(f"{dataset.observations}")
print(f"{dataset.episodes}")
print(f"{dataset.episodes[0].transitions}")

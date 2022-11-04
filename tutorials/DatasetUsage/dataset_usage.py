import kabuki

dataset = kabuki.download_dataset("LunarLander-v2-remote_test_dataset")

print("*" * 60, " Dataset Structure")
print(f"Dataset attributes: {dataset.__dir__()}\n")
print(f"Episode attributes: {dataset.episodes[0].__dir__()}\n")
print(f"Transition attributes: {dataset.episodes[0].transitions[0].__dir__()}\n")

print("*" * 60, " Examples")
print(f"Shape of observations: {dataset.observations.shape}\n")
print(f"Return of third episode: {dataset.episodes[2].compute_return()}\n")
print(f"21st action of fifth episode: {dataset.episodes[5].transitions[21].action}\n")

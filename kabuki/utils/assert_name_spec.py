import re


def test_and_return_name(dataset_path):
    parsed = re.search(
        r"(\\*|\/*)([\w\-]*)\.*(\w*)$", dataset_path
    )  # https://regex101.com/r/cg5fK7/1
    if not parsed:
        raise ValueError(
            f"Invalid dataset path {dataset_path}. Please raise an issue on GitHub."
        )
    filename = parsed.group(2)
    extension = parsed.group(3)
    assert (
        extension == "hdf5"
    ), f"File extension must be .hdf5, not {extension}. (Are you trying to use a period in your dataset name?)"
    if not re.match(r"\w+-v\d-\w+$", filename):
        raise ValueError(
            f"Invalid dataset name {filename}. Must be in the format <environment_name>-v<environment_version>-<description>."
        )
    return filename

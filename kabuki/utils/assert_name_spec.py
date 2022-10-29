import re


def test_and_return_name(dataset_path):
    _assert_hdf5(dataset_path)
    _assert_name_spec(dataset_path)
    return _get_dataset_name(dataset_path)


def _get_dataset_name(dataset_path):
    return re.match(r"/[^\\/]+?(?=\.\w+$)/", dataset_path).group(0)


def _assert_name_spec(dataset_path):
    # https://stackoverflow.com/a/58412900
    filename = _get_dataset_name(dataset_path)

    if not re.match(r"\w+-v\d-\w+", filename):
        raise ValueError(f"Invalid dataset name {filename}")


def _assert_hdf5(dataset_path):
    assert dataset_path.endswith(".hdf5"), "Dataset must be in HDF5 format"

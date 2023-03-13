---
title: Dataset Format
---

# Dataset Information


## Storage

The local root path where the Minari datasets are stored is set by default to `.minari/datasets/`, this path can be overridden by setting the environment variable `MINARI_DATASET_LOCATION`. The directory tree structure has in the first level. The Minari dataset id standard has the dataset name followed by the version of the dataset, for dataset `my_dataset` and version `0`, `my_dataset-v0`

<div class="only-light">
<ul class="directory-list">
<li class="folder">minari_root
    <ul>
    <li class="folder">dataset_name-v0
        <ul>
        <li class="folder">data
            <ul>
            <li class="file">main_data.hdf5</li>
            <li class="file">additional_data_0.hdf5</li>
            <li class="file">additional_data_1.hdf5</li>
            </ul>
        </li>
        </ul>
    </li>
    <li class="folder">dataset_name-v1
        <ul>
        <li class="folder">data
            <ul>
            <li class="file">main_data.hdf5</li>
            </ul>
        </li>
        </ul>
    </li>
    <li class="folder-closed">other_dataset_name-v0</li>
    </ul>
</li>
</ul>
</div>

<div class="only-dark">
<ul class="directory-list">
<li class="folder-white">minari_root
    <ul class="white">
    <li class="folder-white">dataset_name-v0
        <ul class="white">
        <li class="folder-white">data
            <ul class="white">
            <li class="file-white">main_data.hdf5</li>
            <li class="file-white">additional_data_0.hdf5</li>
            <li class="file-white">additional_data_1.hdf5</li>
            </ul>
        </li>
        </ul>
    </li>
    <li class="folder-white">dataset_name-v1
        <ul class="white">
        <li class="folder-white">data
            <ul class="white">
            <li class="file-white">main_data.hdf5</li>
            </ul>
        </li>
        </ul>
    </li>
    <li class="folder-white-closed">other_dataset_name-v0</li>
    </ul>
</li>
</ul>
</div>

## Dataset File Format

required groups and datasets found for every Minari dataset created with a Gymnasium environment. rewards, actions, observations, terminations, truncations. Optional store the infos group or you can even add extra data to the one returned each step and create nested dictionaries of data.

### HDF5 file structure

<div class="only-light">
<ul class="directory-list">
<li class="file">main_data.hdf5
    <ul>
        <li class="folder">episode_0
            <ul>
                <li class="dataset">observations</li>
                <li class="dataset">actions</li>
                <li class="dataset">terminations</li>
                <li class="dataset">truncations</li>
                <li class="dataset">rewards</li>
                <li class="folder">infos
                <ul>
                    <li class="dataset">infos_datasets</li>
                    <li class="folder">infos_subgroup
                    <ul>
                        <li class="dataset">more_datasets</li>
                    </ul>
                    </li>
                </ul>
                </li>
                <li class="folder">additional_groups
                    <ul>
                        <li class="dataset">additional_datasets</li>
                    </ul>
                </li>
            </ul>
        </li>
        <li class="folder-closed">episode_1</li>
        <li class="folder-closed">episode_2</li>
        <ul><br></ul>
        <li class="folder-closed">episode_id</li>
    </ul>
</li>
</ul>
</div>

<div class="only-dark">
<ul class="directory-list">
    <li class="file-white" style="color:white">main_data.hdf5
        <ul class="white">
            <li class="folder-white">episode_0
                <ul>
                    <li class="dataset-white">observations</li>
                    <li class="dataset-white">actions</li>
                    <li class="dataset-white">terminations</li>
                    <li class="dataset-white">truncations</li>
                    <li class="dataset-white">rewards</li>
                    <li class="folder-white">infos
                    <ul class="white">
                        <li class="dataset-white">infos_datasets</li>
                        <li class="folder-white">infos_subgroup
                        <ul>
                            <li class="dataset-white">more_datasets</li>
                        </ul>
                        </li>
                    </ul>
                    </li>
                    <li class="folder-white">additional_groups
                        <ul class="white">
                            <li class="dataset-white">additional_datasets</li>
                        </ul>
                    </li>
                </ul>
            </li>
            <li class="folder-white-closed">episode_1</li>
            <li class="folder-white-closed">episode_2</li>
            <ul class="white"><br></ul>
            <li class="folder-white-closed">episode_id</li>
        </ul>
    </li>
</ul>
</div>

### Default dataset metadata
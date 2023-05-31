import h5py
import numpy as np


class EpisodeMetadataCallback:
    """Callback to full episode after saving to hdf5 file as a group.

    This callback can be overridden to add extra metadata attributes or statistics to
    each HDF5 episode group in the Minari dataset. The custom callback can then be
    passed to the DataCollectorV0 wrapper to the `episode_metadata_callback` argument.

    TODO: add more default statistics to episode datasets
    """

    def __call__(self, eps_group: h5py.Group):
        """Callback method.

        Override this method to add custom attribute metadata to the episode group.

        Args:
            eps_group (h5py.Group): the HDF5 group that contains an episode's datasets
        """
        eps_group["rewards"].attrs["sum"] = np.sum(eps_group["rewards"])
        eps_group["rewards"].attrs["mean"] = np.mean(eps_group["rewards"])
        eps_group["rewards"].attrs["std"] = np.std(eps_group["rewards"])
        eps_group["rewards"].attrs["max"] = np.max(eps_group["rewards"])
        eps_group["rewards"].attrs["min"] = np.min(eps_group["rewards"])

        eps_group.attrs["total_steps"] = eps_group["rewards"].shape[0]

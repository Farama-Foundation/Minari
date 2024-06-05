from typing import Dict


class EpisodeMetadataCallback:
    """Callback to full episode after saving to hdf5 file as a group.

    This callback can be overridden to add extra metadata attributes or statistics to
    each episode in the Minari dataset. The custom callback can then be
    passed to the DataCollector wrapper to the `episode_metadata_callback` argument.
    """

    def __call__(self, episode: Dict) -> Dict:
        """Callback method.

        Override this method to add custom attribute metadata to the episode group.

        Args:
            episode (dict): the dict that contains an episode's data
        Returns:
            Dict: episode metadata.
        """
        return {
            "rewards_sum": float(episode["rewards"].sum()),
            "rewards_mean": float(episode["rewards"].mean()),
            "rewards_std": float(episode["rewards"].std()),
            "rewards_max": float(episode["rewards"].max()),
            "rewards_min": float(episode["rewards"].min()),
        }

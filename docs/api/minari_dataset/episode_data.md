# EpisodeData

## minari.EpisodeData

```{eval-rst}
.. autoclass:: minari.EpisodeData
```

### Attributes

```{eval-rst}
.. autoattribute:: minari.EpisodeData.id

    The ID of the episode in the Minari dataset.

.. autoattribute:: minari.EpisodeData.seed

    The seed used to reset this episode in the Gymnasium API.

.. autoattribute:: minari.EpisodeData.total_steps

    The number of steps contained in this episode.

.. autoattribute:: minari.EpisodeData.observations

    Stacked observations of the episode. The initial and final observations are included meaning that the number
    of observations will be increased by one compared to the number of steps.

.. autoattribute:: minari.EpisodeData.actions

    Stacked actions taken in each episode step.

.. autoattribute:: minari.EpisodeData.terminations

    The ``terminated`` value after each environment step.

.. autoattribute:: minari.EpisodeData.truncations

    The ``truncated`` value after each environment step.

.. autoattribute:: minari.EpisodeData.infos

    The stacked ``infos`` of the episodes. As for the observations, this attribute contains one more element compared to the number of steps.
```

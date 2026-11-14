---
title: Data Collection
---

# DataCollector

```{toctree}
:hidden:
data_collector/episode_buffer
data_collector/step_data_callback
data_collector/episode_metadata_callback
```

## minari.DataCollector

```{eval-rst}
.. autoclass:: minari.DataCollector
```

### Methods

```{eval-rst}
.. autofunction:: minari.DataCollector.step
.. autofunction:: minari.DataCollector.reset
.. autofunction:: minari.DataCollector.create_dataset
.. autofunction:: minari.DataCollector.add_to_dataset
.. autofunction:: minari.DataCollector.close
```

## Recorded data ownership

`DataCollector` retains references to recorded arrays by default, avoiding copies
on the collection path. Until the episode is written, callers, policies and
`StepDataCallback` implementations must not modify these arrays. In particular,
mutating an observation returned by the collector or reusing a policy's action
array can change buffered data. Gymnasium environments must independently return
fresh observation and info objects from `reset()` and `step()`.

For pipelines that mutate these arrays, enable snapshotting explicitly:

```python
import gymnasium as gym
import minari

collector = minari.DataCollector(gym.make("CartPole-v1"), copy_data=True)
# Collect transitions as usual, then close the collector when finished.
collector.close()
```

This deep-copies the recorded observation, action, reward and (when enabled) info
**after** the step callback, including the initial observation and recorded info
from `reset()`. Changes made later cannot alter those snapshots. The environment's
return objects are still passed to the caller unchanged. Unrecorded infos and
unused callback fields are not copied. Snapshotting costs time and memory
proportional to the recorded payload; leave it disabled when the producer already
provides stable data. It does not repair mutations inside `env.step()` before the
callback runs, nor replace Gymnasium's fresh-output requirement.

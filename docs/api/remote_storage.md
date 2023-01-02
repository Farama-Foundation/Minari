---
title: "Remote Usage"
---

# Remote Storage
## API Overview

Minari's API easily lets you list datasets available to download, and download datasets to local storage. With permissions, it is also easy to upload your own datasets.
```{eval-rst}
.. literalinclude:: ../../tutorials/RemoteStorage/remote_storage.py
   :language: python
   :lines: 100-128
   :emphasize-lines: 12, 15, 24
```

## Complete Working Example
### Environment Setup
To run the full code below, you will need to install the dependencies shown below. It is recommended to use a newly-created virtual environment to avoid dependency conflicts.
```{eval-rst}
.. literalinclude:: ../../tutorials/RemoteStorage/requirements.txt
   :language: text
```
Then you need to `pip install -e .` from the root of the repository.

### Full Code
```{eval-rst}
.. literalinclude:: ../../tutorials/RemoteStorage/remote_storage.py
   :language: python
```
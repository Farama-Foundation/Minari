from minari.storage.hosting import (
    download_dataset,
    list_remote_datasets,
    upload_dataset,
)
from minari.storage.local import delete_dataset, list_local_datasets, load_dataset

from . import dataset

__version__ = "0.0.1"

try:
    import sys
    from farama_notifications import notifications

    if "minari" in notifications and __version__ in notifications["minari"]:
        print(notifications["minari"][__version__], file=sys.stderr)
except Exception:  # nosec
    pass

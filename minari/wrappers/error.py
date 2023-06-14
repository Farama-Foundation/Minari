"""Set of Error classes for gymnasium."""
import warnings


class Error(Exception):
    """Error superclass."""


# Local errors
class ExistingDataSet(Error):
    """Raised when the dataset already exists and force_download=False."""


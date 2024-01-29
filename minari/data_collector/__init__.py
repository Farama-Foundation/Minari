from minari.data_collector.data_collector import DataCollector


__all__ = ["DataCollector"]


def __getattr__(name):
    if name == "DataCollectorV0":
        from minari.data_collector.data_collector import DataCollectorV0
        return DataCollectorV0
    else:
        raise ImportError(f"cannot import name '{name}' from '{__name__}' ({__file__})")

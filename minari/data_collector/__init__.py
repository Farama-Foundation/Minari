from minari.data_collector.data_collector import DataCollector


__all__ = ["DataCollector"]


def __getattr__(name):
    if name == "DataCollectorV0":
        from minari.data_collector.data_collector import DataCollectorV0
        return DataCollectorV0

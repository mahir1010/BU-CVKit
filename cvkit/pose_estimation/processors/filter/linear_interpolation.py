from cvkit.pose_estimation.processors.processor_interface import Processor, ProcessorMetaData


class LinearInterpolationFilter(Processor):
    REQUIRES_STATS = True
    PROCESSOR_NAME = "Linear Interpolation"
    PROCESSOR_ID = "cvkit_interpolation"
    META_DATA = {'target_column': ProcessorMetaData('Target Part', ProcessorMetaData.BODY_PART),
                 'threshold': ProcessorMetaData('Threshold', ProcessorMetaData.FLOAT, 0.6, 0.0, 1.0),
                 'max_cluster_size': ProcessorMetaData('Maximum Window Size', ProcessorMetaData.INT, 10, 1)}
    PROCESSOR_SUMMARY = "Performs linear interpolation to fill missing body parts."
    DISTRIBUTED = True

    def __init__(self, target_column, threshold=0.6, max_cluster_size=10):
        super(LinearInterpolationFilter, self).__init__()
        self.target_column = target_column
        self.threshold = threshold
        self.max_cluster_size = max_cluster_size

    def process(self, data_store):
        self._data_store = data_store
        self._data_ready = False
        self._progress = 0
        if not self._data_store.verify_stats():
            raise Exception("This process requires data-frame statistics."
                            "\nPlease run ClusterAnalysis before this one")
        for index, candidate in enumerate(self._data_store.stats.iter_na_clusters(self.target_column)):
            self._progress = int(index / len(self._data_store) * 100)
            if candidate['begin'] == 0 or candidate['end'] == len(self._data_store) - 1:
                continue
            if candidate['end'] - candidate['begin'] < self.max_cluster_size:
                begin = self._data_store.get_part(candidate['begin'] - 1, self.target_column)
                end = self._data_store.get_part(candidate['end'] + 1, self.target_column)
                vector = (end - begin) / (candidate['end'] - candidate['begin'] + 2)
                current = begin + vector
                current.likelihood = self.threshold
                for i in range(candidate['begin'], candidate['end'] + 1):
                    self._data_store.set_part(i, current)
                    current += vector
        self._data_ready = True
        self._progress = 100

    def get_output(self):
        if self._data_ready:
            return self._data_store
        else:
            return None

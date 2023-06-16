from cvkit.pose_estimation.data_readers import DataStoreStats
from cvkit.pose_estimation.processors.processor_interface import Processor, ProcessorMetaData


class ClusterAnalysis(Processor):
    PROCESSOR_NAME = "Data Analysis"
    PROCESSOR_ID = "cvkit_cluster_analysis"
    META_DATA = {'threshold': ProcessorMetaData('Threshold', ProcessorMetaData.FLOAT, 0.6, 0.0, 1.0)}
    PROCESSOR_SUMMARY = "Generates meta-statistics of 2D/3D pose data. Provides list of clusters of accurate or missing data."

    def __init__(self, threshold=0.6):
        self.threshold = threshold
        super(ClusterAnalysis, self).__init__()

    def process(self, data_store):
        self._data_store = data_store
        self._data_ready = False
        self._progress = 0
        if not self._data_store.verify_stats():
            body_parts = self._data_store.body_parts
            stats = DataStoreStats(body_parts)
            for index, skeleton in self._data_store.row_iterator():
                self._progress = int(index / len(self._data_store) * 100)
                if self.PRINT and self._progress % 10 == 0:
                    print(f'\r {self.PROCESS_NAME} {self._progress}% complete', end='')
                accurate = True
                acc_count = len(data_store.body_parts)
                for part in body_parts:
                    if skeleton[part] < self.threshold:
                        stats.update_cluster_info(index, part)
                        accurate = False
                        acc_count -= 1
                stats.add_occupancy_data(acc_count / len(data_store.body_parts))
                if accurate:
                    stats.update_cluster_info(index, '', True)
            if self.PRINT and self._progress % 10 == 0:
                print(f'\r {self.PROCESS_NAME} {self._progress}% complete', end='')
            self._data_store.set_stats(stats)
        self._data_ready = True
        self._progress = 100

    def get_output(self):
        if self._data_ready:
            return self._data_store
        else:
            return None

from cvkit.pose_estimation.data_readers import DataStoreStats
from cvkit.pose_estimation.post_processors.post_prcessor_interface import PostProcessor


class ClusterAnalysis(PostProcessor):
    PROCESS_NAME = "Data Analysis"

    def __init__(self, threshold=0.6):
        self.threshold = threshold
        super(ClusterAnalysis, self).__init__(None)

    def process(self, data_store):
        self.data_store = data_store
        self.data_ready = False
        self.progress = 0
        body_parts = self.data_store.body_parts
        stats = DataStoreStats(body_parts)
        for index, skeleton in self.data_store.row_iterator():
            self.progress = int(index / len(self.data_store) * 100)
            if self.PRINT and self.progress % 10 == 0:
                print(f'\r {self.PROCESS_NAME} {self.progress}% complete', end='')
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
        if self.PRINT and self.progress % 10 == 0:
            print(f'\r {self.PROCESS_NAME} {self.progress}% complete', end='')
        self.data_store.set_stats(stats)
        self.data_ready = True
        self.progress = 100

    def get_output(self):
        if self.data_ready:
            return self.data_store
        else:
            return None

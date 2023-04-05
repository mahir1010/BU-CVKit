from cvkit.pose_estimation.post_processors.post_prcessor_interface import PostProcessor


class LinearInterpolationFilter(PostProcessor):
    REQUIRES_STATS = True
    PROCESS_NAME = "Linear Interpolation"

    def __init__(self, target_column, threshold=0.6, max_cluster_size=10):
        super(LinearInterpolationFilter, self).__init__(target_column)
        self.threshold = threshold
        self.max_cluster_size = max_cluster_size

    def process(self, data_store):
        self.data_store = data_store
        self.data_ready = False
        self.progress = 0
        if not self.data_store.verify_stats():
            raise Exception("This process requires data-frame statistics."
                            "\nPlease run ClusterAnalysis before this one")
        for index, candidate in enumerate(self.data_store.stats.iter_na_clusters(self.target_column)):
            self.progress = int(index / len(self.data_store) * 100)
            if candidate['begin'] == 0 or candidate['end'] == len(self.data_store) - 1:
                continue
            if candidate['end'] - candidate['begin'] < self.max_cluster_size:
                begin = self.data_store.get_part(candidate['begin'] - 1, self.target_column)
                end = self.data_store.get_part(candidate['end'] + 1, self.target_column)
                vector = (end - begin) / (candidate['end'] - candidate['begin'] + 2)
                current = begin + vector
                current.likelihood = self.threshold
                for i in range(candidate['begin'], candidate['end'] + 1):
                    self.data_store.set_part(i, current)
                    current += vector
        self.data_ready = True
        self.progress = 100

    def get_output(self):
        if self.data_ready:
            return self.data_store
        else:
            return None

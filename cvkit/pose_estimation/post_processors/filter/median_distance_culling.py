import numpy as np

from cvkit.pose_estimation.post_processors.post_prcessor_interface import PostProcessor


class MedianDistanceFilter(PostProcessor):
    PROCESS_NAME = "Median Distance Culling"

    def process(self, data_store):
        self.data_store = data_store
        self.data_ready = False
        self.progress = 0
        for index, skeleton in self.data_store.row_iterator():
            self.progress = int(index / len(self.data_store) * 100)
            for part in self.data_store.body_parts:
                point = skeleton[part]
                if point.likelihood >= self.threshold:
                    distances = np.array(
                        [np.linalg.norm(skeleton[part] - skeleton[p]) if skeleton[p] >= self.threshold else 0
                         for p in self.data_store.body_parts])
                    distances = distances[distances != 0]
                    if len(distances) != 0 and np.median(distances) > self.distance_threshold:
                        self.data_store.delete_part(index, part)
        self.data_ready = True
        self.progress = 100

    def get_output(self):
        if self.data_ready:
            return self.data_store
        else:
            return None

    def __init__(self, threshold=0.5, distance_threshold=400):
        self.threshold = threshold
        self.distance_threshold = distance_threshold
        super(MedianDistanceFilter, self).__init__(None)

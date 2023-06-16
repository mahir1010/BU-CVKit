import numpy as np

from cvkit.pose_estimation.processors.processor_interface import Processor, ProcessorMetaData


class MedianDistanceFilter(Processor):
    PROCESSOR_NAME = "Median Distance Culling"
    PROCESSOR_ID = "cvkit_median_distance_culling"
    META_DATA = {'threshold': ProcessorMetaData('Threshold', ProcessorMetaData.FLOAT, 0.6, 0.0, 1.0),
                 'distance_threshold': ProcessorMetaData('Distance Threshold', ProcessorMetaData.FLOAT, 400)}
    PROCESSOR_SUMMARY = "Computes distance matrix among all body parts and filters outliers based on median distance."

    def process(self, data_store):
        self._data_store = data_store
        self._data_ready = False
        self._progress = 0
        for index, skeleton in self._data_store.row_iterator():
            self._progress = int(index / len(self._data_store) * 100)
            for part in self._data_store.body_parts:
                point = skeleton[part]
                if point.likelihood >= self.threshold:
                    distances = np.array(
                        [np.linalg.norm(skeleton[part] - skeleton[p]) if skeleton[p] >= self.threshold else 0
                         for p in self._data_store.body_parts])
                    distances = distances[distances != 0]
                    if len(distances) != 0 and np.median(distances) > self.distance_threshold:
                        self._data_store.delete_part(index, part)
        self._data_ready = True
        self._progress = 100

    def get_output(self):
        if self._data_ready:
            return self._data_store
        else:
            return None

    def __init__(self, threshold=0.6, distance_threshold=400):
        self.threshold = threshold
        self.distance_threshold = distance_threshold
        super(MedianDistanceFilter, self).__init__()

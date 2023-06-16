import numpy as np

from cvkit.pose_estimation.processors.processor_interface import Processor, ProcessorMetaData
from cvkit.pose_estimation.utils import compute_distance_matrix


class DistanceStatisticsFilter(Processor):
    PROCESSOR_NAME = "Distance Statistics Filter"
    PROCESSOR_ID = "cvkit_distance_stats"
    META_DATA = {'distance_matrix_mean': ProcessorMetaData('Distance - Mean', ProcessorMetaData.NUMPY_ARRAY),
                 'distance_matrix_sd': ProcessorMetaData('Distance - SD', ProcessorMetaData.NUMPY_ARRAY),
                 'threshold': ProcessorMetaData('Threshold', ProcessorMetaData.FLOAT, 0.5, 0.0, 1.0),
                 'sd_factor': ProcessorMetaData('SD Scale Factor', ProcessorMetaData.FLOAT, 1.25, 0.0)}
    PROCESSOR_SUMMARY = "Uses Mean and Standard deviation of distance among body parts to filter outliers"

    def __init__(self, distance_matrix_mean, distance_matrix_sd, threshold=0.5, sd_factor=1.25):
        super(DistanceStatisticsFilter, self).__init__()
        assert sd_factor >= 0
        self.distance_matrix_mean = distance_matrix_mean
        self.distance_matrix_sd = distance_matrix_sd
        self.threshold = threshold
        self.sd_factor = sd_factor

    def process(self, data_store):
        self._data_store = data_store
        self._data_ready = False
        self._progress = 0
        distance_matrix_mean = np.load(self.distance_matrix_mean)
        distance_matrix_sd = np.load(self.distance_matrix_sd) * self.sd_factor
        body_parts = data_store.body_parts
        removed_count = [0] * len(data_store.body_parts)
        for index, skeleton in self._data_store.row_iterator():
            self._progress = int(index / len(self._data_store) * 100)
            if self.PRINT and self._progress % 10 == 0:
                print(f'\r {self.PROCESS_NAME} {self._progress}% complete', end='')
            distance_matrix = compute_distance_matrix(skeleton)
            difference = np.absolute(distance_matrix_mean - distance_matrix)
            max_score = len(skeleton) + distance_matrix.trace()

            if max_score == 0:
                continue
            scores = np.array([max_score] * len(skeleton))
            for i in range(len(skeleton)):
                if distance_matrix[i][i] != -1:
                    for j in range(len(skeleton)):
                        if distance_matrix[i][j] != -1 and difference[i][j] < distance_matrix_sd[i][j]:
                            scores[i] -= 1
            scores = scores / max_score
            for i, score in enumerate(scores):
                if distance_matrix[i][i] != -1 and self.threshold < score:
                    removed_count[i] += 1
                    data_store.delete_part(index, body_parts[i])
        self._data_ready = True
        print("\nremoved: ", {body_parts[k]: removed_count[k] for k in range(len(data_store.body_parts))})
        self._progress = 100

    def get_output(self):
        if self._data_ready:
            return self._data_store
        else:
            return None

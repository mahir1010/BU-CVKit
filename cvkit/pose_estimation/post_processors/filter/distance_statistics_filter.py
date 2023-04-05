import numpy as np

from cvkit.pose_estimation.post_processors.post_prcessor_interface import PostProcessor
from cvkit.pose_estimation.utils import compute_distance_matrix


class DistanceStatisticsFilter(PostProcessor):
    PROCESS_NAME = "Distance Statistics Filter"

    def __init__(self, distance_matrix_stats, threshold=0.5, sd_factor=1.25):
        super(DistanceStatisticsFilter, self).__init__()
        assert sd_factor >= 0
        self.distance_matrix_mean = distance_matrix_stats[0]
        self.distance_matrix_sd = distance_matrix_stats[1] * sd_factor
        self.threshold = threshold

    def process(self, data_store):
        self.data_store = data_store
        self.data_ready = False
        self.progress = 0
        body_parts = data_store.body_parts
        removed_count = [0] * len(data_store.body_parts)
        for index, skeleton in self.data_store.row_iterator():
            self.progress = int(index / len(self.data_store) * 100)
            if self.PRINT and self.progress % 10 == 0:
                print(f'\r {self.PROCESS_NAME} {self.progress}% complete', end='')
            distance_matrix = compute_distance_matrix(skeleton)
            difference = np.absolute(self.distance_matrix_mean - distance_matrix)
            max_score = len(skeleton) + distance_matrix.trace()

            if max_score == 0:
                continue
            scores = np.array([max_score] * len(skeleton))
            for i in range(len(skeleton)):
                if distance_matrix[i][i] != -1:
                    for j in range(len(skeleton)):
                        if distance_matrix[i][j] != -1 and difference[i][j] < self.distance_matrix_sd[i][j]:
                            scores[i] -= 1
            scores = scores / max_score
            for i, score in enumerate(scores):
                if distance_matrix[i][i] != -1 and self.threshold < score:
                    removed_count[i] += 1
                    data_store.delete_part(index, body_parts[i])
        self.data_ready = True
        print("\nremoved: ", {body_parts[k]: removed_count[k] for k in range(len(data_store.body_parts))})
        self.progress = 100

    def get_output(self):
        if self.data_ready:
            return self.data_store
        else:
            return None

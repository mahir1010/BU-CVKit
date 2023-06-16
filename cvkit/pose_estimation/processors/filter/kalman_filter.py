import numpy as np
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter as Filter

from cvkit.pose_estimation import Skeleton
from cvkit.pose_estimation.processors.processor_interface import Processor, ProcessorMetaData


class KalmanFilter(Processor):
    PROCESSOR_NAME = "Kalman Filtering"
    PROCESSOR_ID = "cvkit_kalman_filter"
    META_DATA = {'target_column': ProcessorMetaData('Target Part', ProcessorMetaData.BODY_PART),
                 'framerate': ProcessorMetaData('Framerate', ProcessorMetaData.FLOAT),
                 'skip': ProcessorMetaData('Skip Invalids', ProcessorMetaData.BOOLEAN, True),
                 'threshold': ProcessorMetaData('Threshold', ProcessorMetaData.FLOAT, 0.6, 0.0, 1.0)}
    PROCESSOR_SUMMARY = "Constant acceleration Kalman Filter"

    DISTRIBUTED = True

    def __init__(self, target_column, framerate, skip=True, threshold=0.6):
        super(KalmanFilter, self).__init__()
        self.target_column = target_column
        self.framerate = framerate
        self.threshold = threshold
        self.skip = skip
        self.dt = float(1 / framerate)

    def process(self, data_store):
        self._data_store = data_store
        self._data_ready = False
        self._progress = 0
        tracker = None
        for index, point in self._data_store.part_iterator(self.target_column):
            self._progress = int(index / len(self._data_store) * 100)
            if self.skip:
                if point < self.threshold:
                    # new_tracker = Tracker(point, self.dt)
                    # new_tracker.tracker.R = tracker.tracker.R.copy()
                    # new_tracker.tracker.Q = tracker.tracker.Q.copy()
                    # new_tracker.tracker.P = tracker.tracker.P.copy()
                    del tracker
                    # tracker = new_tracker
                    tracker = None
                else:
                    if tracker is None:
                        tracker = Tracker(point, self.dt)
                        # self.data.append(point.tolist())
                    else:
                        # self.data.append(tracker.update(point).tolist())
                        point[:3] = tracker.update(point).tolist()
                        self._data_store.set_part(index, point)
            else:
                if point < self.threshold:
                    if tracker is not None:
                        p = tracker.get_next_pred()
                        p = tracker.update(p).tolist()
                        # self.data.append(p)
                        point[:3] = p
                        self._data_store.set_part(index, point)
                    else:
                        self.data.append(None)
                else:
                    if tracker is None:
                        tracker = Tracker(point, self.dt)
                        # self.data.append(point.tolist())
                    else:
                        # self.data.append(tracker.update(point).tolist())
                        point[:3] = tracker.update(point).tolist()
                        self._data_store.set_part(index, point)
        self._data_ready = True
        self._progress = 100

    def get_output(self):
        if self._data_ready:
            return self._data_store
        else:
            return None


def generate_F_matrix(dimension, dt):
    assert dimension > 1

    arr = []
    for i in range(dimension):
        row = [0] * dimension * 3
        row[i] = 1
        row[i + dimension] = dt
        row[i + 2 * dimension] = 0.5 * dt ** 2
        arr.append(row)
    for i in range(dimension):
        row = [0] * dimension * 3
        row[i] = 0
        row[i + dimension] = 1
        row[i + 2 * dimension] = dt
        arr.append(row)
    for i in range(dimension):
        row = [0] * dimension * 3
        row[i] = 0
        row[i + dimension] = 0
        row[i + 2 * dimension] = 1
        arr.append(row)
    return np.array(arr)


class Tracker:
    def __init__(self, data, dt):
        self.dt = dt
        assert np.ndim(data) == 1
        self.dimensions = np.shape(data)[-1]
        self.tracker = Tracker.get_kalman_filter(data, self.dimensions, self.dt)

    @staticmethod
    def get_kalman_filter(data, dimensions, dt):
        kalman = Filter(dimensions * 3, dimensions)
        kalman.x = np.hstack((data, [0.0] * dimensions * 2)).astype(np.float32)
        kalman.F = generate_F_matrix(dimensions, dt)
        kalman.H = np.array([[0] * dimensions * 3] * dimensions)
        kalman.H[list(range(dimensions)), list(range(dimensions))] = 1
        kalman.P *= 100
        kalman.R *= 0.8
        kalman.Q = Q_discrete_white_noise(dimensions, dt=dt, block_size=3, order_by_dim=False)
        kalman.B = 0
        return kalman

    def get_next_pred(self):
        return self.tracker.H @ self.tracker.get_prediction()[0]

    def update(self, data, likelihood=1.0, threshold=0.9):
        self.tracker.predict()
        if likelihood < threshold:
            self.tracker.update(None)
        else:
            self.tracker.update(np.array(data))
        return self.tracker.x[:self.dimensions]


class SkeletonTracker:
    def __init__(self, skeleton: Skeleton, dt):
        self.parts = {}
        self.dt = dt
        for part in skeleton.parts.keys():
            self.parts[part] = Tracker(skeleton[part])

    def get_next_pred(self, part):
        return self.parts[part].get_next_pred()

    def update(self, skeleton: Skeleton, threshold=.80):
        for part in self.parts:
            skeleton[part] = self.parts[part].update(skeleton[part], skeleton.partsLikelihood[part], threshold)
        return skeleton

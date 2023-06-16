import numpy as np

from cvkit.pose_estimation.processors.processor_interface import Processor, ProcessorMetaData


class MovingAverageFilter(Processor):
    PROCESSOR_NAME = "Moving Average"
    PROCESSOR_ID = "cvkit_moving_average"
    META_DATA = {'target_column': ProcessorMetaData('Target Part', ProcessorMetaData.BODY_PART),
                 'threshold': ProcessorMetaData('Threshold', ProcessorMetaData.FLOAT, 0.6, 0.0, 1.0),
                 'window_size': ProcessorMetaData('Window Size', ProcessorMetaData.INT, min_val=1)}
    PROCESSOR_SUMMARY = "Runs a moving average filter to reduce noise."
    DISTRIBUTED = True

    def __init__(self, target_column, window_size, threshold=0.6):
        super(MovingAverageFilter, self).__init__()
        self.target_column = target_column
        self.threshold = threshold
        self.window_size = window_size

    def process(self, data_store):
        self._data_store = data_store
        average_window = []
        self._data_ready = False
        self._progress = 0
        for index, point in self._data_store.part_iterator(self.target_column):
            self._progress = int(index / len(self._data_store) * 100)
            if point < self.threshold:
                average_window.clear()
            else:
                average_window.append(point)
                if len(average_window) > self.window_size:
                    average_window.pop(0)
                point[:] = np.average(average_window, weights=np.square(list(range(1, len(average_window) + 1))),
                                      axis=0)
                self._data_store.set_part(index, point)
        self._data_ready = True
        self._progress = 100

    def get_output(self):
        if self._data_ready:
            return self._data_store
        else:
            return None

import numpy as np

from cvkit.pose_estimation.processors.processor_interface import Processor, ProcessorMetaData
from cvkit.pose_estimation.utils import magnitude


class VelocityFilter(Processor):
    REQUIRES_STATS = True
    PROCESSOR_NAME = "Velocity Filter"
    PROCESSOR_ID = "cvkit_velocity_filter"
    META_DATA = {'target_column': ProcessorMetaData('Target Part', ProcessorMetaData.BODY_PART),
                 'threshold': ProcessorMetaData('likelihood Threshold', ProcessorMetaData.FLOAT),
                 'threshold_velocity': ProcessorMetaData('Velocity Threshold', ProcessorMetaData.FLOAT),
                 'framerate': ProcessorMetaData('Framerate', ProcessorMetaData.FLOAT)}
    PROCESSOR_SUMMARY = "Filters body parts with velocity higher than the threshold."
    DISTRIBUTED = True

    def __init__(self, target_column, threshold, framerate, threshold_velocity):
        super(VelocityFilter, self).__init__()
        self.target_column = target_column
        self.threshold = threshold
        self.framerate = framerate
        self.dt = 1 / framerate
        self.threshold_velocity = threshold_velocity

    def process(self, data_store):
        self._data_store = data_store
        self._data_ready = False
        self._progress = 0
        self._removed = 0
        previous_point = None
        previous_index = -1
        for index, point in self._data_store.part_iterator(self.target_column):
            self._progress = int(index / len(self._data_store) * 100)
            if self.PRINT and self._progress % 10 == 0:
                print(f'\r {self.PROCESSOR_NAME} {self._progress}% complete', end='')
            if point > self.threshold:
                if previous_point is not None:
                    velocity = np.subtract(point, previous_point) / ((index - previous_index) * self.dt)
                else:
                    velocity = [0, 0, 0]
                if magnitude(velocity) <= self.threshold_velocity:
                    previous_point = point.copy()
                    previous_index = index
                else:
                    data_store.delete_part(index, self.target_column, True)
                    self._removed+=1
        if self.PRINT and self._progress % 10 == 0:
            print(f'\r {self.PROCESSOR_NAME} {self._progress}% complete', end='')
        self._data_ready = True
        self._progress = 100

    def get_output(self):
        if self._data_ready:
            return self._data_store
        else:
            return None

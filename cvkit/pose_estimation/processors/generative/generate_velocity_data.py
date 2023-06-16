import numpy as np

from cvkit import MAGIC_NUMBER
from cvkit.pose_estimation import Part
from cvkit.pose_estimation.processors.processor_interface import Processor, ProcessorMetaData


class GenerateVelocity(Processor):
    PROCESSOR_NAME = "Generate Velocity"
    PROCESSOR_ID = "cvkit_generate_velocity"
    META_DATA = {'target_column': ProcessorMetaData('Target Part', ProcessorMetaData.BODY_PART),
                 'threshold': ProcessorMetaData('Threshold', ProcessorMetaData.FLOAT, 0.6, 0.0, 1.0),
                 'velocity_threshold': ProcessorMetaData('Velocity Threshold', ProcessorMetaData.FLOAT),
                 'framerate': ProcessorMetaData('Framerate', ProcessorMetaData.FLOAT)}

    PROCESSOR_SUMMARY = "Computes velocity data for target column and stores in empty datastore"

    def __init__(self, target_column, framerate, velocity_threshold, threshold=0.6):
        super(GenerateVelocity, self).__init__()
        self.target_column = target_column
        self.framerate = framerate
        self.dt = 1 / framerate
        self.threshold = threshold
        self.velocity_threshold = velocity_threshold

    def process(self, data_store, empty_datastore):
        self._data_store = data_store
        self._data_ready = False
        self._progress = 0
        previous_point = None
        previous_index = -1
        for index, point in self._data_store.part_iterator(self.target_column):
            self._progress = int(index / len(self._data_store) * 100)
            if self.PRINT and self._progress % 10 == 0:
                print(f'\r {self.PROCESS_NAME} {self._progress}% complete', end='')
            velocity = [MAGIC_NUMBER, MAGIC_NUMBER, MAGIC_NUMBER]
            flag = False
            if point > self.threshold:
                if previous_point is not None:
                    velocity = np.subtract(point, previous_point) / ((index - previous_index) * self.dt)
                else:
                    velocity = [0, 0, 0]
                if max(np.abs(velocity).tolist()) <= self.velocity_threshold:
                    flag = True
                    previous_point = point.copy()
                    previous_index = index
                else:
                    velocity = [MAGIC_NUMBER, MAGIC_NUMBER, MAGIC_NUMBER]
            empty_datastore.set_part(index, Part(velocity, self.target_column, float(flag)))
        if self.PRINT:
            print(f'\r {self.PROCESS_NAME} 100% complete', end='')
        self._data_ready = True
        self._progress = 100

    def get_output(self):
        if self._data_ready:
            return self._data_store
        else:
            return None

from cvkit.pose_estimation.data_readers import initialize_datastore_reader
from cvkit.pose_estimation.processors.processor_interface import Processor, ProcessorMetaData
from cvkit.pose_estimation.utils import magnitude


class VelocityFilter(Processor):
    REQUIRES_STATS = True
    PROCESSOR_NAME = "Velocity Filter"
    PROCESSOR_ID = "cvkit_velocity_filter"
    META_DATA = {
        'velocity_ds_dict': ProcessorMetaData('Velocity Data Store', ProcessorMetaData.DATA_STORE, serialize=False),
        'global_config': ProcessorMetaData('Global Config', ProcessorMetaData.GLOBAL_CONFIG),
        'threshold_velocity': ProcessorMetaData('Velocity Threshold', ProcessorMetaData.FLOAT)}
    PROCESSOR_SUMMARY = "Filters body parts with velocity higher than the threshold."

    def __init__(self, global_config, velocity_ds_dict, threshold_velocity):
        super(VelocityFilter, self).__init__()
        self.global_config = global_config
        self.velocity_ds_dict = velocity_ds_dict
        self._velocity_data_store = initialize_datastore_reader(self.global_config.body_parts,
                                                                self.velocity_ds_dict['path'],
                                                                self.velocity_ds_dict['type'])
        self.threshold_velocity = threshold_velocity

    def process(self, data_store):
        self._data_store = data_store
        self._data_ready = False
        self._progress = 0
        for index in range(len(data_store)):
            self._progress = int(index / len(self._data_store) * 100)
            if self.PRINT and self._progress % 10 == 0:
                print(f'\r {self.PROCESS_NAME} {self._progress}% complete', end='')
            for body_part in data_store.body_parts:
                if magnitude(self._velocity_data_store.get_part(index, body_part)) > self.threshold_velocity:
                    data_store.delete_part(index, body_part, True)
        if self.PRINT and self._progress % 10 == 0:
            print(f'\r {self.PROCESS_NAME} {self._progress}% complete', end='')
        self._data_ready = True
        self._progress = 100

    def get_output(self):
        if self._data_ready:
            return self._data_store
        else:
            return None

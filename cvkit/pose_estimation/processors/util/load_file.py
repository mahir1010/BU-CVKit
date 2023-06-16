from cvkit.pose_estimation.config import PoseEstimationConfig
from cvkit.pose_estimation.data_readers import DataStoreInterface, initialize_datastore_reader
from cvkit.pose_estimation.processors.processor_interface import Processor, ProcessorMetaData


class LoadFile(Processor):
    PROCESSOR_NAME = "File Loader"
    PROCESSOR_ID = "cvkit_load_file"
    META_DATA = {'data_store_dict': ProcessorMetaData('Input File', ProcessorMetaData.DATA_STORE, serialize=False),
                 'global_config': ProcessorMetaData('Global Config', ProcessorMetaData.GLOBAL_CONFIG), }
    PROCESSOR_SUMMARY = "Utility processor for loading initial data file."

    def process(self, data_store: DataStoreInterface):
        self._data_store = initialize_datastore_reader(self.global_config.body_parts, self.data_store_dict['path'],
                                                       self.data_store_dict['type'])
        self._progress = 100

    def get_output(self):
        return self._data_store

    def __init__(self, global_config: PoseEstimationConfig, data_store_dict):
        super(LoadFile, self).__init__()
        self.global_config = global_config
        self.data_store_dict = data_store_dict

from cvkit.pose_estimation.data_readers import DataStoreInterface
from cvkit.pose_estimation.processors.processor_interface import Processor, ProcessorMetaData


class SaveFile(Processor):
    PROCESSOR_NAME = "Save File"
    PROCESSOR_ID = "cvkit_save_file"
    META_DATA = {'path': ProcessorMetaData('File Path', ProcessorMetaData.FILE_PATH, regex='*.csv', serialize=False)}
    PROCESSOR_SUMMARY = "Utility processor for saving the final data file."

    def process(self, data_store: DataStoreInterface):
        self._data_store = data_store
        data_store.save_file(self.path)
        self._progress = 100

    def get_output(self):
        return self._data_store

    def __init__(self, path):
        super(SaveFile, self).__init__()
        self.path = path

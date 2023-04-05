from cvkit.pose_estimation.data_readers import DataStoreInterface
from cvkit.pose_estimation.post_processors.post_prcessor_interface import PostProcessor


class SaveFile(PostProcessor):
    PROCESS_NAME = "Save File"

    def process(self, data_store: DataStoreInterface):
        self.data_store = data_store
        data_store.save_file(self.path)
        self.progress = 100

    def get_output(self):
        return self.data_store

    def __init__(self, path):
        super(SaveFile, self).__init__(None)
        self.path = path

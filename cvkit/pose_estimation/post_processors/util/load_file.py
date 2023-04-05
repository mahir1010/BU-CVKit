from cvkit.pose_estimation.data_readers import DataStoreInterface
from cvkit.pose_estimation.post_processors.post_prcessor_interface import PostProcessor


class LoadFile(PostProcessor):
    PROCESS_NAME = "Load File"

    def process(self, data_store: DataStoreInterface):
        self.progress = 100
        pass

    def get_output(self):
        return self.data_store

    def __init__(self, data_store):
        super(LoadFile, self).__init__(None)
        self.data_store = data_store

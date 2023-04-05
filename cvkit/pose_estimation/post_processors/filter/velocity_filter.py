from cvkit.pose_estimation.post_processors.post_prcessor_interface import PostProcessor
from cvkit.pose_estimation.utils import magnitude


class VelocityFilter(PostProcessor):
    REQUIRES_STATS = True
    PROCESS_NAME = "Velocity Filter"

    def __init__(self, velocity_data_store, threshold_velocity):
        super(VelocityFilter, self).__init__(None)
        self.velocity_data_store = velocity_data_store
        self.threshold_velocity = threshold_velocity

    def process(self, data_store):
        self.data_store = data_store
        self.data_ready = False
        self.progress = 0
        for index in range(len(data_store)):
            self.progress = int(index / len(self.data_store) * 100)
            if self.PRINT and self.progress % 10 == 0:
                print(f'\r {self.PROCESS_NAME} {self.progress}% complete', end='')
            for body_part in data_store.body_parts:
                if magnitude(self.velocity_data_store.get_part(i, body_part)) > self.threshold_velocity:
                    data_store.delete_part(index, body_part, True)
        if self.PRINT and self.progress % 10 == 0:
            print(f'\r {self.PROCESS_NAME} {self.progress}% complete', end='')
        self.data_ready = True
        self.progress = 100

    def get_output(self):
        if self.data_ready:
            return self.data_store
        else:
            return None

from cvkit.pose_estimation.post_processors.post_prcessor_interface import PostProcessor


class RegionFilter2D(PostProcessor):
    REQUIRES_STATS = True
    PROCESS_NAME = "2D Region Filter"

    def __init__(self, uncertainty_regions: list):
        super(RegionFilter2D, self).__init__(None)
        self.uncertainty_region = uncertainty_regions

    def process(self, data_store):
        self.data_store = data_store
        self.data_ready = False
        self.progress = 0
        for index, skeleton in data_store.row_iterator():
            self.progress = int(index / len(self.data_store) * 100)
            if self.PRINT and self.progress % 10 == 0:
                print(f'\r {self.PROCESS_NAME} {self.progress}% complete', end='')
            for part in data_store.body_parts:
                for uncertainty_region in self.uncertainty_regions:
                    if uncertainty_region[0][0] < skeleton[part][0] < uncertainty_region[0][1] and \
                            uncertainty_region[1][
                                0] < skeleton[part][1] < uncertainty_region[1][1]:
                        data_store.delete_part(index, part, force_remove=True)
                        break
        if self.PRINT and self.progress % 10 == 0:
            print(f'\r {self.PROCESS_NAME} {self.progress}% complete', end='')
        self.data_ready = True
        self.progress = 100

    def get_output(self):
        if self.data_ready:
            return self.data_store
        else:
            return None

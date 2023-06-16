import numpy as np

from cvkit.pose_estimation.processors.processor_interface import Processor, ProcessorMetaData


class RegionFilter2D(Processor):
    REQUIRES_STATS = True
    PROCESSOR_NAME = "2D Region Filter"
    PROCESSOR_ID = "cvkit_2d_region_filter"
    META_DATA = {'uncertainty_regions': ProcessorMetaData('Uncertain Regions', ProcessorMetaData.NUMPY_ARRAY)}
    PROCESSOR_SUMMARY = "Deletes body parts lying in provided 2D regions of uncertainty."

    def __init__(self, uncertainty_regions):
        super(RegionFilter2D, self).__init__()
        self.uncertainty_regions = uncertainty_regions

    def process(self, data_store):
        self._data_store = data_store
        self._data_ready = False
        self._progress = 0
        uncertainty_regions = np.load(self.uncertainty_regions)
        for index, skeleton in data_store.row_iterator():
            self._progress = int(index / len(self._data_store) * 100)
            if self.PRINT and self._progress % 10 == 0:
                print(f'\r {self.PROCESS_NAME} {self._progress}% complete', end='')
            for part in data_store.body_parts:
                for uncertainty_region in uncertainty_regions:
                    if uncertainty_region[0][0] < skeleton[part][0] < uncertainty_region[0][1] and \
                            uncertainty_region[1][
                                0] < skeleton[part][1] < uncertainty_region[1][1]:
                        data_store.delete_part(index, part, force_remove=True)
                        break
        if self.PRINT and self._progress % 10 == 0:
            print(f'\r {self.PROCESS_NAME} {self._progress}% complete', end='')
        self._data_ready = True
        self._progress = 100

    def get_output(self):
        if self._data_ready:
            return self._data_store
        else:
            return None

from cvkit.pose_estimation.processors.processor_interface import Processor, ProcessorMetaData
from cvkit.utils import build_intrinsic
import cv2
import numpy as np

class UndistortFilter(Processor):
    PROCESSOR_NAME = "Undistort Points"
    PROCESSOR_ID = "cvkit_undistort_pts"
    META_DATA = {'global_config': ProcessorMetaData('Global Config', ProcessorMetaData.GLOBAL_CONFIG),
                 'threshold': ProcessorMetaData('Threshold', ProcessorMetaData.FLOAT, 0.6, 0.0, 1.0),
                 'source_view': ProcessorMetaData('Source Views', ProcessorMetaData.VIEWS,min_val=1,max_val=1)}
    PROCESSOR_SUMMARY = "Undistorts 2D points using provided distortion coefficients."

    def __init__(self,global_config,source_view,threshold=0.6):
        super(UndistortFilter, self).__init__()
        self.global_config = global_config
        self.threshold = threshold
        self.source_view = source_view

    def process(self, data_store):
        self._data_store = data_store
        self._data_ready = False
        self._progress = 0
        camera = self.global_config.views[self.source_view]
        matrix = build_intrinsic(camera.f_px,camera.principal_point)
        distortion = camera.distortion
        for index, skeleton in data_store.row_iterator():
            self._progress = int(index / len(self._data_store) * 100)
            if self.PRINT and self._progress % 10 == 0:
                print(f'\r {self.PROCESSOR_NAME} {self._progress}% complete', end='')
            points = skeleton.numpy()[:,:2].reshape(-1,1,2)
            points = cv2.undistortPoints(points,matrix,distortion,None,matrix)
            for part,point in zip(skeleton.body_parts,points):
                skeleton[part][:2] = point[0]
            data_store.set_skeleton(index,skeleton)
        if self.PRINT:
            print(f'\r {self.PROCESSOR_NAME} 100% complete', end='')
        self._data_ready = True
        self._progress = 100

    def get_output(self):
        if self._data_ready:
            return self._data_store
        else:
            return None

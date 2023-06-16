import numpy as np

from cvkit.pose_estimation import Skeleton
from cvkit.pose_estimation.config import PoseEstimationConfig
from cvkit.pose_estimation.data_readers.cvkit_datastore import CVKitDataStore3D
from cvkit.pose_estimation.processors.processor_interface import Processor, ProcessorMetaData
from cvkit.pose_estimation.reconstruction.DLT import DLTrecon
from cvkit.pose_estimation.utils import rotate


class DLTReconstruction(Processor):
    PROCESSOR_NAME = "Reconstruction"
    PROCESSOR_ID = "cvkit_3d_reconstruct"
    META_DATA = {'global_config': ProcessorMetaData('Global Config', ProcessorMetaData.GLOBAL_CONFIG),
                 'source_views': ProcessorMetaData('Source Views', ProcessorMetaData.VIEWS),
                 'data_readers': ProcessorMetaData('DataReaders', ProcessorMetaData.FILE_MAP),
                 'file_prefix': ProcessorMetaData('File Prefix', ProcessorMetaData.TEXT),
                 'threshold': ProcessorMetaData('Threshold', ProcessorMetaData.FLOAT, 0.6, 0.0, 1.0)}
    PROCESSOR_SUMMARY = "Performs 3D reconstruction from selected source views."

    def __init__(self, global_config: PoseEstimationConfig, source_views, data_readers, threshold):
        super(DLTReconstruction, self).__init__()
        self.global_config = global_config
        self.threshold = threshold
        self.source_views = source_views
        self.data_readers = [data_readers[source_view] for source_view in source_views]
        self._out_csv = None

    def process(self, data_store):
        self._out_csv = CVKitDataStore3D(self.global_config.body_parts, None)
        dlt_coefficients = np.array([self.global_config.views[view].dlt_coefficients for view in self.source_views])
        rotation_matrix = np.array(self.global_config.rotation_matrix)
        scale = self.global_config.computed_scale
        translation_matrix = np.array(self.global_config.translation_matrix) * scale
        length = len(min(self.data_readers, key=lambda x: len(x)))
        self._data_ready = False
        self._progress = 0
        for iterator in range(length):
            self._progress = int(iterator / length * 100)
            skeleton_2D = [reader.get_skeleton(iterator) for reader in self.data_readers]
            recon_data = {}
            prob_data = {}
            for name in self.global_config.body_parts:
                subset = [sk[name] for sk in skeleton_2D]
                dlt_subset = dlt_coefficients.copy()
                indices = [subset[i].likelihood >= self.threshold for i in range(len(subset))]
                if (self.global_config.reconstruction_algorithm == "auto_subset" and sum(indices) >= 2) or sum(
                        indices) == len(
                    self.data_readers):
                    dlt_subset = dlt_subset[indices, :]
                    subset = [element for i, element in enumerate(subset) if indices[i]]
                    recon_data[name] = rotate(DLTrecon(3, len(subset), dlt_subset, subset), rotation_matrix,
                                              scale,
                                              multiplier=self.global_config.axis_rotation_3D) + translation_matrix
                    prob_data[name] = min(subset, key=lambda x: x.likelihood).likelihood
            skeleton_3D = Skeleton(self.global_config.body_parts, recon_data, prob_data)
            self._out_csv.set_skeleton(iterator, skeleton_3D)
        self._progress = 100
        self._data_ready = True

    def get_output(self):
        if self._data_ready:
            return self._out_csv
        else:
            return None

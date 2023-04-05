import numpy as np

from cvkit.pose_estimation.config import PoseEstimationConfig
from cvkit.pose_estimation import Skeleton
from cvkit.pose_estimation.data_readers.cvkit_datastore import CVKitDataStore3D
from cvkit.pose_estimation.post_processors.post_prcessor_interface import PostProcessor
from cvkit.pose_estimation.reconstruction.DLT import DLTrecon
from cvkit.pose_estimation.utils import rotate


class DLTReconstruction(PostProcessor):
    PROCESS_NAME = "Reconstruction"

    def __init__(self, global_config: PoseEstimationConfig, source_views, data_readers, threshold,
                 reconstruction_algorithm,
                 scale):
        super(DLTReconstruction, self).__init__(None)
        self.body_parts = global_config.body_parts
        self.threshold = threshold
        self.data_readers = data_readers
        self.dlt_coefficients = np.array([global_config.views[view].dlt_coefficients for view in source_views])
        self.reconstruction_algorithm = reconstruction_algorithm
        self.rotation_matrix = np.array(global_config.rotation_matrix)
        assert self.rotation_matrix.shape == (3, 3)
        self.scale = scale
        self.translation_matrix = np.array(global_config.translation_matrix) * self.scale
        assert self.translation_matrix.shape == (3,)
        self.out_csv=None

    def process(self, data_store):
        self.out_csv = CVKitDataStore3D(self.body_parts, None)
        length = len(min(self.data_readers, key=lambda x: len(x)))
        self.data_ready = False
        self.progress = 0
        for iterator in range(length):
            self.progress = int(iterator / length * 100)
            skeleton_2D = [reader.get_skeleton(iterator) for reader in self.data_readers]
            recon_data = {}
            prob_data = {}
            for name in self.body_parts:
                subset = [sk[name] for sk in skeleton_2D]
                dlt_subset = self.dlt_coefficients
                indices = [subset[i].likelihood >= self.threshold for i in range(len(subset))]
                if (self.reconstruction_algorithm == "auto_subset" and sum(indices) >= 2) or sum(indices) == len(
                        self.data_readers):
                    dlt_subset = dlt_subset[indices, :]
                    subset = [element for i, element in enumerate(subset) if indices[i]]
                    recon_data[name] = rotate(DLTrecon(3, len(subset), dlt_subset, subset), self.rotation_matrix,
                                              self.scale) + self.translation_matrix
                    prob_data[name] = min(subset, key=lambda x: x.likelihood).likelihood
            skeleton_3D = Skeleton(self.body_parts, recon_data, prob_data)
            self.out_csv.set_skeleton(iterator, skeleton_3D)
        self.progress = 100
        self.data_ready = True

    def get_output(self):
        if self.data_ready:
            return self.out_csv
        else:
            return None

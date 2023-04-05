import os

import numpy as np

from cvkit.pose_estimation.config import PoseEstimationConfig
from cvkit.pose_estimation import Part
from cvkit.pose_estimation.data_readers import DeeplabcutDataStore
from cvkit.pose_estimation.post_processors.post_prcessor_interface import PostProcessor
from cvkit.pose_estimation.reconstruction.DLT import DLTdecon
from cvkit.pose_estimation.utils import rotate


class DLTDeconstruction(PostProcessor):
    PROCESS_NAME = "Deconstruction Process"

    def __init__(self, global_config: PoseEstimationConfig, target_views, scale, file_prefix=None):
        super(DLTDeconstruction, self).__init__(None)
        self.body_parts = global_config.body_parts
        self.dlt_coefficients = np.array([global_config.views[view].dlt_coefficients for view in target_views])
        self.target_views = target_views
        self.rotation_matrix = np.linalg.inv(np.array(global_config.rotation_matrix))
        assert self.rotation_matrix.shape == (3, 3)
        self.scale = scale
        self.translation_matrix = np.array(global_config.translation_matrix) * self.scale
        self.file_prefix = file_prefix
        if file_prefix is not None:
            self.file_prefix = os.path.join(global_config.output_folder, file_prefix)
        assert self.translation_matrix.shape == (3,)

    def process(self, data_store):
        out_files = [DeeplabcutDataStore(data_store.body_parts,
                                         f'{data_store.base_file_path if self.file_prefix is None else self.file_prefix}_{target_view}.csv')
                     for
                     target_view in self.target_views]
        self.data_ready = False
        self.progress = 0
        for index, skeleton in data_store.row_iterator():
            self.progress = int(index / len(data_store) * 100)
            if self.PRINT and self.progress % 10 == 0:
                print(f'\r{self.progress}% complete', end='')
            for part in data_store.body_parts:
                if skeleton[part] > 0:
                    raw_part_3d = rotate(np.array(skeleton[part]) - self.translation_matrix, self.rotation_matrix,
                                         self.scale, True)
                    parts_2d = np.round(DLTdecon(self.dlt_coefficients, raw_part_3d, 3, len(self.target_views)))[0,
                               :].reshape(len(self.target_views), 2)
                    for part_2d, data_store_2d in zip(parts_2d, out_files):
                        data_store_2d.set_part(index, Part(part_2d, part, 1.0))
        self.progress = 100
        for file in out_files:
            file.save_file()
        self.data_ready = True

    def get_output(self):
        return None

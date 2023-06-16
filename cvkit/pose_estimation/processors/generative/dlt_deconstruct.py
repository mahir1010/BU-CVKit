import os

import numpy as np

from cvkit.pose_estimation import Part
from cvkit.pose_estimation.config import PoseEstimationConfig
from cvkit.pose_estimation.data_readers import DeeplabcutDataStore
from cvkit.pose_estimation.processors.processor_interface import Processor, ProcessorMetaData
from cvkit.pose_estimation.reconstruction.DLT import DLTdecon
from cvkit.pose_estimation.utils import rotate


class DLTDeconstruction(Processor):
    PROCESSOR_NAME = "Deconstruction Process"
    PROCESSOR_ID = "cvkit_3d_deconstruct"
    META_DATA = {'global_config': ProcessorMetaData('Global Config', ProcessorMetaData.GLOBAL_CONFIG),
                 'target_views': ProcessorMetaData('Target Views', ProcessorMetaData.VIEWS),
                 'file_prefix': ProcessorMetaData('File Prefix', ProcessorMetaData.TEXT)}
    PROCESSOR_SUMMARY = "Re-projects 3D data back to target 2D views."

    def __init__(self, global_config: PoseEstimationConfig, target_views, file_prefix=None):
        super(DLTDeconstruction, self).__init__()
        self.global_config = global_config
        self.target_views = target_views
        self.file_prefix = file_prefix
        if file_prefix is not None:
            self.file_prefix = os.path.join(global_config.output_folder, file_prefix)

    def process(self, data_store):
        out_files = [DeeplabcutDataStore(data_store.body_parts,
                                         f'{data_store.base_file_path if self.file_prefix is None else self.file_prefix}_{target_view}.csv')
                     for
                     target_view in self.target_views]
        dlt_coefficients = np.array([self.global_config.views[view].dlt_coefficients for view in self.target_views])
        rotation_matrix = np.linalg.inv(np.array(self.global_config.rotation_matrix))
        scale = self.global_config.computed_scale
        translation_matrix = np.array(self.global_config.translation_matrix) * scale
        self._data_ready = False
        self._progress = 0
        for index, skeleton in data_store.row_iterator():
            self._progress = int(index / len(data_store) * 100)
            if self.PRINT and self._progress % 10 == 0:
                print(f'\r{self._progress}% complete', end='')
            for part in data_store.body_parts:
                if skeleton[part] > 0:
                    raw_part_3d = rotate(np.array(skeleton[part]) - translation_matrix, rotation_matrix,
                                         scale, True, multiplier=[1, 1, -1])
                    parts_2d = np.round(DLTdecon(dlt_coefficients, raw_part_3d, 3, len(self.target_views)))[0,
                               :].reshape(len(self.target_views), 2)
                    for part_2d, data_store_2d in zip(parts_2d, out_files):
                        data_store_2d.set_part(index, Part(part_2d, part, 1.0))
        self._progress = 100
        for file in out_files:
            file.save_file()
        self._data_ready = True

    def get_output(self):
        return None

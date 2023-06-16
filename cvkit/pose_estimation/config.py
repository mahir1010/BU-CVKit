import os

import numpy as np
import yaml as yml

DEFAULT_THRESHOLD = 0.6


class CameraViews:
    def __init__(self, data_dictionary, framerate):
        self.axes = data_dictionary.get('axes', {})
        self.dlt_coefficients = np.array(data_dictionary.get('dlt_coefficients', []))
        assert self.dlt_coefficients.shape == (12,)
        self.framerate = framerate
        self.pos = np.array(data_dictionary.get('pos', []))
        self.resolution = np.array(data_dictionary.get('resolution', []))
        self.principal_point = np.array(data_dictionary.get('principal_point', []))
        self.f_px = data_dictionary.get('f_px', -1)

    def export_dict(self):
        return {
            'axes': self.axes,
            'dlt_coefficients': self.dlt_coefficients.tolist(),
            'pos': self.pos.tolist(),
            'resolution': self.resolution.tolist(),
            'f_px': self.f_px,
            'principal_point': self.principal_point.tolist()
        }


class AnnotationConfig:
    def __init__(self, data_dictionary):
        self.view = data_dictionary.get('view', None)
        self.annotation_file = data_dictionary['annotation_file']
        self.annotation_file_flavor = data_dictionary['annotation_file_flavor']
        self.video_file = data_dictionary['video_file']
        try:
            assert os.path.isfile(self.video_file)
        except:
            print(self.video_file, "not found!")
            exit()
        self.video_reader = data_dictionary['video_reader']

    def export_dict(self):
        return {'view': self.view,
                'annotation_file': self.annotation_file,
                'annotation_file_flavor': self.annotation_file_flavor,
                'video_file': self.video_file,
                'video_reader': self.video_reader
                }


class PoseEstimationConfig:
    ENABLE_FLOW_STYLE = ['name', 'output_folder', 'threshold', 'reprojection_toolbox', 'behaviours', 'body_parts',
                         'skeleton']

    def __init__(self, path):
        self.path = path
        self.data_dictionary = yml.safe_load(open(path, 'r'))
        assert 0 <= DEFAULT_THRESHOLD < 1.0
        self.project_name = self.data_dictionary.get('name', 'unnamed')
        self.output_folder = self.data_dictionary['output_folder']
        assert os.path.exists(self.output_folder)
        self.threshold = float(self.data_dictionary['Reconstruction'].get('threshold', DEFAULT_THRESHOLD))
        self.axis_rotation_3D = np.array(
            self.data_dictionary['Reconstruction'].get('axis_rotation_3D', np.array([1, 1, 1])))
        if np.any(np.abs(self.axis_rotation_3D) != 1):
            print(f"Resetting {self.axis_rotation_3D} to [1,1,1]")
            self.axis_rotation_3D = np.array([1, 1, 1])
        self.body_parts = self.data_dictionary['body_parts']
        self.num_parts = len(self.body_parts)
        self.skeleton = self.data_dictionary['skeleton']
        self.colors = list(self.data_dictionary.get('colors', []))
        self.framerate = self.data_dictionary['Reconstruction']['framerate']
        self.annotation_views = {}
        if 'annotation' in self.data_dictionary:
            for annotation_view in self.data_dictionary['annotation']:
                assert annotation_view != 'Reconstruction' and annotation_view != 'Sync'
                data = self.data_dictionary['annotation'][annotation_view]
                self.annotation_views[annotation_view] = AnnotationConfig(data)
        self.views = {}
        if 'views' in self.data_dictionary:
            for view in self.data_dictionary['views']:
                self.views[view] = CameraViews(self.data_dictionary['views'][view], self.framerate)
        self.rotation_matrix = np.array(self.data_dictionary['Reconstruction'].get('rotation_matrix', np.identity(3)),
                                        dtype=np.float32)
        assert self.rotation_matrix.shape == (3, 3)
        self.scale = float(self.data_dictionary['Reconstruction'].get('scale', 1.0))
        self.computed_scale = self.data_dictionary['Reconstruction'].get('computed_scale', self.scale)
        self.translation_matrix = np.array(self.data_dictionary['Reconstruction'].get('translation_matrix', [0, 0, 0]),
                                           dtype=np.float32)
        self.reconstruction_algorithm = self.data_dictionary.get('reconstruction_algorithm', 'default')

    def export_dict(self):
        return {'name': self.project_name,
                'output_folder': self.output_folder,
                'body_parts': self.body_parts,
                'skeleton': self.skeleton,
                'annotation': {view: self.annotation_views[view].export_dict() for view in self.annotation_views},
                'colors': self.colors,
                'views': {view: self.views[view].export_dict() for view in self.views},
                'Reconstruction': {
                    'threshold': self.threshold,
                    'scale': self.scale,
                    'framerate': self.framerate,
                    'rotation_matrix': self.rotation_matrix.tolist(),
                    'translation_matrix': self.translation_matrix.tolist(),
                    'reconstruction_algorithm': self.reconstruction_algorithm,
                    'computed_scale': float(self.computed_scale),
                    'axis_rotation_3D': self.axis_rotation_3D.tolist()
                }
                }


def save_config(path, data_dict):
    out_file = ''
    for key in data_dict:
        if key in PoseEstimationConfig.ENABLE_FLOW_STYLE:
            out_file += yml.dump({key: data_dict[key]}) + '\n'
        else:
            out_file += yml.dump({key: data_dict[key]}, default_flow_style=None) + '\n'
    save_file = open(path, 'w')
    save_file.write(out_file)

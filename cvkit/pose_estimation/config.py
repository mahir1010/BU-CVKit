import os

import numpy as np
import yaml as yml

DEFAULT_THRESHOLD = 0.6 #: Default likelihood threshold value


class CameraViews:
    """ Stores metadata of the camera setup.

    :param data_dictionary: Dictionary containing the metadata for the recording setup.
    :type data_dictionary: dict
    :param framerate: Project level Framerate. (Assumes equal framerate for all views)
    :type framerate: float
    """

    def __init__(self, data_dictionary, framerate):

        self.axes = data_dictionary.get('axes', {}) #: Contains 2D x_max, y_max, and origin. This can be used to create a coordinate system for the reconstructed data.
        self.dlt_coefficients = np.array(data_dictionary.get('dlt_coefficients', [])) #: DLT co-efficients generated by the EasyWand package.
        self.framerate = framerate
        self.pos = np.array(data_dictionary.get('pos', [])) #: Extrinsic data: Position of the camera in world coordinates.
        self.resolution = np.array(data_dictionary.get('resolution', [-1,-1])) #: Intrinsic Data: Resolution of the captured video.
        self.principal_point = np.array(data_dictionary.get('principal_point', [-1,-1])) #: Principal point of the camera lens.
        self.distortion = np.array(data_dictionary.get('distortion',np.zeros((1,5))))
        if self.distortion.ndim ==1 :
            self.distortion = np.expand_dims(self.distortion,0)
        self.f_px = data_dictionary.get('f_px', -1) #: Focal length in pixels
    def is_dlt_valid(self):
        return self.dlt_coefficients.shape == (12,)
    
    def export_dict(self):
        return {
            'axes': self.axes,
            'dlt_coefficients': self.dlt_coefficients.tolist(),
            'pos': self.pos.tolist(),
            'resolution': self.resolution.tolist(),
            'f_px': self.f_px,
            'principal_point': self.principal_point.tolist(),
            'distortion': self.distortion.tolist()
        }


class AnnotationConfig:
    """ Stores information about data files for each view of the project.

    :param data_dictionary: dictionary containing annotation meta-data.
    :type data_dictionary: dict
    """
    def __init__(self, name,data_dictionary):

        self.view = name #: Name of the camera
        self.annotation_file = data_dictionary['annotation_file'] #: Path of the annotation data file.
        self.annotation_file_flavor = data_dictionary['annotation_file_flavor'] #: Flavor of the data file. Refer :py:attr::py:attr:`cvkit.pose_estimation.data_readers.datastore_interface.DataStoreInterface.FLAVOR`
        self.video_file = data_dictionary['video_file'] #: Path to the video file
        try:
            assert os.path.isfile(self.video_file)
        except:
            print(self.video_file, "not found!")
            exit()
        self.video_reader = data_dictionary['video_reader'] #: Flavor of the video file. Refer :py:attr:`cvkit.video_readers.video_reader_interface.BaseVideoReaderInterface.FLAVOR`

    def export_dict(self):
        return {'annotation_file': self.annotation_file,
                'annotation_file_flavor': self.annotation_file_flavor,
                'video_file': self.video_file,
                'video_reader': self.video_reader
                }


class PoseEstimationConfig:
    """This class is used to read and write pose estimation configuration files. It contains basic information about the experiments such as the
    body parts of the tracked subject, their connectivity, camera setup, data folder, data files, reconstruction parameters, and so on.

    .. highlight:: YAML
    .. code-block:: YAML

        # Project Name
        name: unnamed_project

        # Valid path to output folder
        output_folder: ''
        # List of body parts
        body_parts:
          - part_1
          - part_2
          - part_3
          - part_4
        # List of lists defining body part connections
        skeleton:
          - - part_1
            - part_3
          - - part_2
            - part_3
          - - part_3
            - part_4
        # List of colors (R,G,B). If enough colors are not provided, others will be randomly generated.
        colors: #Optional
            - [ 230, 25, 75 ] # Color for part_1
            - [ 60, 180, 75 ] # Color for part_2
            - [ 255, 225, 25 ] # Color for part_3

        # Reconstruction configuration parameters
        Reconstruction:

            # Project level framerate. We currently only support videos with equal      #
            # framerate                                                                 #
            framerate: 60

            # Unscaled length of the x-axis
            x_len: <length>

            # Unscaled length of the y-axis
            y_len: <length>

            # Reconstruction algorithm, accepts 'default' or 'auto_subset'              #
            # default: Reconstructs if likelihood is higher than the threshold for all  #
            # views.                                                                    #
            # auto_subset: Automatically creates a subset of 'accurate' viewpoints      #
            # based on the threshold value. The reconstruction is performed if the      #
            # number of viewpoints is more than 2.                                      #
            reconstruction_algorithm: default # Optional

            # Rotation Matrix to align 3D reconstructed data. It will be multiplied     #
            # after initial reconstruction.                                             #
            rotation_matrix: # Optional
            - [ 1.0, 0.0, 0.0 ]
            - [ 0.0, 1.0, 0.0 ]
            - [ 0.0, 0.0, 1.0 ]

            # The desired scale factor for converting reconstructed data's units.       #
            # Example: If reconstructed data is in meters, scale can be set to 1000 to  #
            # to generate data in millimeters.                                          #
            scale: 1.0

            # Scale factor that can be computed through update_alignment_matrices.       #
            # This uses pre-known distances on the arena to adjust the desired scaling   #
            # factor for mitigating reconstruction noise.                                #
            computed_scale: [1.0,1.0,1.0] # Optional, defaults to scale

            # Project level likelihood threshold value.
            threshold: 0.75

            # Static translation vector. Added after scaling.                           #
            # Used for moving origin to desired location.                               #
            # Note: The translation vector has to be scaled before adding               #
            translation_vector: [ 0, 0, 0 ]

            # Axis Alignment vector. Used to flip targeted axis.                        #
            # Only accepts either 1 or -1, indicating whether the corresponding axis    #
            # will be flipped.                                                          #
            # [-1,1,-1] Flips x and z axes.                                             #
            axis_rotation_3D: [1,1, 1]
        annotation:
            VIEW_NAME_1:
                annotation_file: '' # Path to datastore containing pose data for the view
                annotation_file_flavor: <flavor> # DataStoreInterface Flavor
                video_file: '' # Path to the video file for the view
                video_reader: <flavor> # BaseVideoReaderInterface Flavor
                # Corresponding Camera ID. Use None for importing video data not        #
                # corresponding to any cameras.                                         #
                view: None

            # Repeat for each annotated views

        views:
            Cam_id_1:
                axes:
                    origin: [-1, -1 ] # 2D position of the origin for this camera view
                    x_max: [ -1, -1 ] # 2D position of the x max location for this camera view
                    y_max: [ -1, -1 ] # 2D position of the y max location for this camera view
                dlt_coefficients: <list of 12 numbers representing the DLT co-efficients for this camera view>
                f_px: -1 # Focal length of the camera in px
                pos: [ ] # Position of the camera in world coordinates.
                principal_point: [ ] # Principal point of the camera.
                resolution: [ ] # Resolution of the captured frames.
                
            # Repeat for each camera.

    :param path: The path of the yaml file
    :type path: str
    """

    ENABLE_FLOW_STYLE = ['name', 'output_folder', 'threshold', 'reprojection_toolbox', 'behaviours', 'body_parts',
                         'skeleton']

    def __init__(self, path):

        self.path = path
        self.data_dictionary = yml.safe_load(open(path, 'r'))
        assert 0 <= DEFAULT_THRESHOLD < 1.0
        self.project_name = self.data_dictionary.get('name', 'unnamed') #: The name of the Project
        self.output_folder = self.data_dictionary['output_folder'] #: Path for the output directory
        assert os.path.exists(self.output_folder)
        self.threshold = float(self.data_dictionary['Reconstruction'].get('threshold', DEFAULT_THRESHOLD)) #: Threshold value for the project.
        self.axis_rotation_3D = np.array(
            self.data_dictionary['Reconstruction'].get('axis_rotation_3D', np.array([1, 1, 1]))) #: 3 dimensional list where all the elements are either 1 or -1. This can be used to flip desired axis.
        if np.any(np.abs(self.axis_rotation_3D) != 1):
            print(f"Resetting {self.axis_rotation_3D} to [1,1,1]")
            self.axis_rotation_3D = np.array([1, 1, 1])
        self.body_parts = self.data_dictionary['body_parts'] #: List of body parts
        self.num_parts = len(self.body_parts) #: Number of body parts
        self.skeleton = self.data_dictionary['skeleton'] #: Defines connectivity among the body parts.
        self.colors = list(self.data_dictionary.get('colors', []))#: Custom colors for each body part. Colors are randomly generated if not explicitly provided.
        self.framerate = self.data_dictionary['Reconstruction']['framerate']#: Project level framerate
        self.annotation_views = {} #: A dictionary mapping views to its corresponding data files - :py:class:`~cvkit.pose_estimation.config.AnnotationConfig`.
        if 'annotation' in self.data_dictionary:
            for annotation_view in self.data_dictionary['annotation']:
                assert annotation_view != 'Reconstruction' and annotation_view != 'Sync'
                data = self.data_dictionary['annotation'][annotation_view]
                self.annotation_views[annotation_view] = AnnotationConfig(annotation_view,data)
        self.views = {} #: A dictionary mapping view names to camera information - :py:class:`~cvkit.pose_estimation.config.CameraViews`.
        if 'views' in self.data_dictionary:
            for view in self.data_dictionary['views']:
                self.views[view] = CameraViews(self.data_dictionary['views'][view], self.framerate)
        self.rotation_matrix = np.array(self.data_dictionary['Reconstruction'].get('rotation_matrix', np.identity(3)),
                                        dtype=np.float32) #: 3x3 Rotation matrix for aligning reconstructed data.
        assert self.rotation_matrix.shape == (3, 3)
        self.x_len = float(self.data_dictionary['Reconstruction'].get('x_len',-1))
        self.y_len = float(self.data_dictionary['Reconstruction'].get('y_len',-1))
        self.scale = float(self.data_dictionary['Reconstruction'].get('scale', 1.0)) #: Project level scale factor for reconstructed data.
        self.computed_scale = self.data_dictionary['Reconstruction'].get('computed_scale', self.scale) #: Computed scale factor based on pre-known distances to reduce reconstruction noise
        self.translation_vector = np.array(self.data_dictionary['Reconstruction'].get('translation_vector', [0, 0, 0]),
                                           dtype=np.float32) #: Fixed 3-D translational vector for reconstructed data.
        self.reconstruction_algorithm = self.data_dictionary['Reconstruction'].get('reconstruction_algorithm', 'default') #: Reconstruction algorithm. Auto-Subset: Picks at least 2 views based on likelihood values. Regular: Only reconstructs if all views have likelihood higher than the threshold.

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
                    'x_len': self.x_len,
                    'y_len': self.y_len,
                    'scale': self.scale,
                    'framerate': self.framerate,
                    'rotation_matrix': self.rotation_matrix.tolist(),
                    'translation_vector': self.translation_vector.tolist(),
                    'reconstruction_algorithm': self.reconstruction_algorithm,
                    'computed_scale': float(self.computed_scale),
                    'axis_rotation_3D': self.axis_rotation_3D.tolist()
                }
                }

def save_config(path, data_dict):
    """ Saves given data dictionary to Yaml file

    :param path: Output File Path
    :type path: str
    :param data_dict: dictionary containing the project configuration
    :type data_dict: dict
    """
    out_file = ''
    for key in data_dict:
        if key in PoseEstimationConfig.ENABLE_FLOW_STYLE:
            out_file += yml.dump({key: data_dict[key]}) + '\n'
        else:
            out_file += yml.dump({key: data_dict[key]}, default_flow_style=None) + '\n'
    save_file = open(path, 'w')
    save_file.write(out_file)

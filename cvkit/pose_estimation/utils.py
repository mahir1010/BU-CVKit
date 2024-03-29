import json
import math

import numpy as np

from cvkit import MAGIC_NUMBER
from cvkit.pose_estimation import Part
from cvkit.pose_estimation.config import CameraViews
from cvkit.utils import build_intrinsic
import cv2

def rotate(vector, rotation, scale=1.0, is_inv=False, axis_alignment_vector=None):
    """ Rotates a vector with rotation matrix followed by multiplying with axis alignment vector, followed by linear scaling.
        If is_inv is set, the opposite operation is performed. First linear de-scaling, followed by  axis alignment and rotation. Note: Although the function computes scale inverse, it does not compute rotation inverse.

    :param vector: The vector to be rotated
    :param rotation: Rotation Matrix
    :param scale:  Scaling Factor (accepts numpy array defining separate scaling factor for each axis)
    :param is_inv: Flag for deciding flow of operations ('rotate → axis alignment → scale' or 'de-scale → axis alignment → rotate')
    :return: rotated, scaled, and aligned vector
    """
    if axis_alignment_vector is None:
        axis_alignment_vector = np.ones_like(vector)
    else:
        axis_alignment_vector = np.array(axis_alignment_vector)
    if not is_inv:
        op = np.matmul(vector, rotation) * axis_alignment_vector * scale
    else:
        op = np.matmul((vector / scale) * axis_alignment_vector, rotation)
    return op


def magnitude(vector):
    """Computes magnitude of the vector

    :param vector: Input Vector
    :return: Frobenius norm of the vector
    """
    return np.linalg.norm(vector,axis=0)


def compute_distance_matrix(skeleton,threshold=0.6):
    """Generates nxn Euclidean distance matrix for given skeleton where n = number of body parts.

    :param skeleton: Input skeleton
    :param threshold: Threshold for considering a body part as valid
    :return: nxn numpy array containing Euclidean distance among all body parts.
    """
    return np.array(
        [[magnitude(skeleton[p1] - skeleton[p2]) if skeleton[p1] > threshold and skeleton[p2] > threshold else -1 for p2 in
          skeleton.body_parts] for p1 in skeleton.body_parts])


def normalize_vector(vector):
    """Normalized input vector.

    :param vector: input vector
    :return: normalized input vector
    """
    return np.divide(vector, magnitude(vector))


def get_spherical_coordinates(v1, is_degrees=True):
    """
Computes theta and phi spherical coordinates for input 3D vector
    :param v1: Input Vector
    :param is_degrees: Interprets input data as degrees or radians
    :return: [theta,phi] polar coordinates
    """
    multiplier = 57.2958 if is_degrees else 1
    v1 = v1 / magnitude(v1)
    return np.array([math.atan2(v1[1], v1[0]),math.acos(v1[2])]) * multiplier


def spherical_angle_difference(v1, v2, is_abs=True):
    """Calculates shifted difference (v1-v2) between two spherical coordinate vectors.

    :param v1: Target input vector
    :type v1: numpy.ndarray
    :param v2: Source input vector
    :type v2: numpy.ndarray
    :param is_abs: controls whether the difference is absolute
    :type is_abs: bool
    :return: shifted spherical angle difference between two vectors
    :rtype: np.ndarray
    """
    diff = v1 - v2
    absDiff = np.abs(diff)
    output = np.minimum(absDiff, 360.0 - absDiff)
    if not is_abs:
        # Maps True/False to 1/-1
        # [T,F]*2-1 = [1,0]*2-1 = [2,0]*-1 = [1,-1]
        output *= np.sign(diff) * ((output == absDiff) * 2 - 1)
    return output


def convert_to_list(inp):
    if type(inp) != str and type(inp) != list and math.isnan(inp):
        t = np.array([MAGIC_NUMBER, MAGIC_NUMBER, MAGIC_NUMBER], dtype=np.float32)
    else:
        if type(inp) != str:
            t = np.array(inp, dtype=np.float32)
        else:
            if ',' not in inp:
                inp = inp.replace(' ', ',')
            t = np.array(json.loads(inp)).astype(np.float32)
    return t


def convert_to_numpy(input_data, dimensions=3):
    if type(input_data) == np.ndarray and (input_data.dtype == np.float32 or input_data.dtype == np.float32):
        return input_data
    if type(input_data) == list and len(input_data) == dimensions:
        return np.array(input_data)
    elif type(input_data) == str:
        split = ' '
        if ',' in input_data:
            split = ','
        input_data = np.array(input_data[1:-1].strip().split(split)).astype(np.float32)
    else:
        input_data = np.array([MAGIC_NUMBER] * dimensions, dtype=np.float32)
    return input_data


def convert_numpy_to_datastore(pickled_data: np.ndarray, header_names, flavor, output_path):
    from cvkit.pose_estimation.data_readers import initialize_datastore_reader
    assert pickled_data.ndim == 3 and pickled_data.shape[1] == len(header_names)
    datastore = initialize_datastore_reader(header_names, output_path, flavor)
    for index, row in enumerate(pickled_data):
        skeleton = datastore.get_skeleton(index)
        for i, header_name in enumerate(header_names):
            skeleton[header_name] = Part(row[i], header_name, 1.0)
        datastore.set_skeleton(index, skeleton)
    datastore.save_file()


def generate_distance_matrices(num_parts, data_points: list):
    output_distance_matrices = np.zeros((2, num_parts, num_parts))
    distance_matrices = []
    for point in data_points:
        distance_matrices.append(compute_distance_matrix(point))
    output_distance_matrices[0] = np.mean(distance_matrices, axis=0)
    output_distance_matrices[1] = np.std(distance_matrices, axis=0)
    return output_distance_matrices

def undistort_point(point:np.ndarray,camera:CameraViews):
    matrix = build_intrinsic(camera.f_px, camera.principal_point)
    distortion = camera.distortion
    point = point.reshape(-1,1,2)
    return cv2.undistortPoints(point, matrix, distortion, None, matrix).reshape(2,)
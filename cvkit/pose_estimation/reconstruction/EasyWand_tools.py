import csv
import os
import random
from os.path import join

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from cvkit.pose_estimation import Part
from cvkit.pose_estimation.config import PoseEstimationConfig
from cvkit.pose_estimation.processors.util import ClusterAnalysis
from cvkit.pose_estimation.reconstruction.DLT import DLTrecon
from cvkit.pose_estimation.utils import rotate, magnitude, undistort_point
from cvkit.utils import build_intrinsic
import cv2


def generate_EasyWand_data(config: PoseEstimationConfig, csv_maps, common_indices, static_points_map):
    os.makedirs(join(config.output_folder, 'calibration'), exist_ok=True)
    file = open(join(config.output_folder, 'calibration',
                     f'{config.project_name}_calibration_camera_order.txt'), 'w')
    camera_profile = open(
        join(config.output_folder, 'calibration', f'{config.project_name}_calibration_camera_profiles.txt'),
        'w')
    for i, camera in enumerate(config.views):
        file.write(f'{camera} ')
        camera_config = config.views[camera]
        profile = f'{i + 1} {camera_config.f_px} {camera_config.resolution[0]} {camera_config.resolution[1]} {camera_config.principal_point[0]} {camera_config.principal_point[1]} 1 0 0 0 0 0\n'
        camera_profile.write(profile)
    camera_profile.close()
    file.close()
    static_writer = csv.writer(
        open(os.path.join(config.output_folder, 'calibration', f'{config.project_name}_background.csv'), 'w'))
    writer = csv.writer(
        open(os.path.join(config.output_folder, 'calibration', f'{config.project_name}_WandPoints.csv'), 'w'))
    parts = config.body_parts
    for idx in common_indices:
        builder = []
        for part in parts:
            for view_name in csv_maps:
                csv_df = csv_maps[view_name]
                camera = config.views[view_name]
                builder.extend(np.round(undistort_point(csv_df.get_part(idx, part)[:2],camera)).astype(np.int32))
        writer.writerow(builder)

    static_rows = []
    for camera in static_points_map:
        for i, point in enumerate(static_points_map[camera]):
            if len(static_rows) == i:
                static_rows.append([])
            static_rows[i].extend(point)
    static_writer.writerows(static_rows)


def update_alignment_matrices(config: PoseEstimationConfig, source_views: list):
    if len(source_views) < 2:
        return False
    try:
        dlt_coefficients = np.array([config.views[view].dlt_coefficients for view in source_views])
        origin_2D = []
        x_max_2D = []
        y_max_2D = []
        for view in source_views:
            camera = config.views[view]
            matrix = build_intrinsic(camera.f_px, camera.principal_point)
            distortion = camera.distortion
            pts = np.array([config.views[view].axes['origin'], config.views[view].axes['x_max'], config.views[view].axes['y_max']],dtype=float).reshape(-1, 1, 2)
            pts = cv2.undistortPoints(pts, matrix, distortion, None, matrix).reshape(-1,2)
            origin_2D.append(pts[0])
            x_max_2D.append(pts[1])
            y_max_2D.append(pts[2])
        # origin_2D = [config.views[view].axes['origin'] for view in source_views]
        # x_max_2D = [config.views[view].axes['x_max'] for view in source_views]
        # y_max_2D = [config.views[view].axes['y_max'] for view in source_views]
        origin = Part(np.round(DLTrecon(3, len(origin_2D), dlt_coefficients, origin_2D),3), "origin", 1)
        x_max = Part(np.round(DLTrecon(3, len(x_max_2D), dlt_coefficients, x_max_2D),3), "x_max", 1)
        y_max = Part(np.round(DLTrecon(3, len(y_max_2D), dlt_coefficients, y_max_2D),3), "y_max", 1)
        rotation_matrix = Rotation.align_vectors([x_max - origin, y_max - origin], [[1, 0, 0], [0, 1, 0]])[
            0].as_matrix()
        origin = Part(rotate(DLTrecon(3, len(origin_2D), dlt_coefficients, origin_2D), rotation_matrix,scale=config.scale,
                             axis_alignment_vector=config.axis_rotation_3D), "origin", 1)
        x_max = Part(rotate(DLTrecon(3, len(x_max_2D), dlt_coefficients, x_max_2D), rotation_matrix,scale=config.scale,
                            axis_alignment_vector=config.axis_rotation_3D), "x_max", 1)
        y_max = Part(rotate(DLTrecon(3, len(y_max_2D), dlt_coefficients, y_max_2D), rotation_matrix,scale=config.scale,
                            axis_alignment_vector=config.axis_rotation_3D), "y_max", 1)
        if config.x_len>0 and config.y_len >0:
            x_scale = (config.x_len/magnitude(x_max-origin))*config.scale
            y_scale = (config.y_len/magnitude(y_max-origin))*config.scale
            config.computed_scale = np.mean([x_scale,y_scale])
        else:
            config.computed_scale = config.scale
        trans_mat = (-origin)/config.scale
        config.rotation_matrix = rotation_matrix
        config.translation_vector = trans_mat
    except Exception as ex:
        return False
    return True


def pick_calibration_candidates(config: PoseEstimationConfig, data_stores: list, resolution, bin_size,max_frames=20):
    assert len(data_stores) > 1
    cluster_analysis = ClusterAnalysis(config.threshold)
    cluster_analysis.PRINT = True
    for i in range(len(data_stores)):
        if not data_stores[i].verify_stats():
            cluster_analysis.process(data_stores[i])
            data_stores[i] = cluster_analysis.get_output()
    accurate_data_points = data_stores[0].stats.accurate_data_points
    for data_store in data_stores[1:]:
        accurate_data_points = data_store.stats.intersect_accurate_data_points(accurate_data_points)
    num_bins = (resolution[0] // bin_size, resolution[1] // bin_size)
    spatial_bins = np.zeros(num_bins)
    frame_number = np.zeros(num_bins)
    part = data_stores[0].body_parts[0]
    candidates = []
    for accurate_data in accurate_data_points:
        for index in range(accurate_data['begin'], accurate_data['end']):
            position = data_stores[0].get_part(index, part)
            x_bin, y_bin = int(position[0] / bin_size), int(position[1] / bin_size)
            if 0 <= x_bin < num_bins[0] and 0 <= y_bin < num_bins[1] and (
                    spatial_bins[x_bin][y_bin] == 0 and index - frame_number[x_bin][y_bin] > config.framerate * .1):
                spatial_bins[x_bin][y_bin] += 1
                frame_number[x_bin][y_bin] = index
                candidates.append(index)
    if len(candidates)>max_frames:
        candidates = random.sample(candidates,max_frames)
    return candidates


def update_config_dlt_coeffs(config: PoseEstimationConfig, dlt_coefficients_file, order):
    dlt_coefficients = pd.read_csv(dlt_coefficients_file, header=None)
    for index, camera in enumerate(order):
        dlt_coeff = dlt_coefficients[index].tolist()
        dlt_coeff.append(1)
        config.views[camera].dlt_coefficients = np.array(dlt_coeff)
    return config

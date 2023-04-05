import os.path
import pickle
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from cvkit import MAGIC_NUMBER
from cvkit.pose_estimation import Skeleton, Part


class DataStoreInterface(ABC):
    FLAVOR = "Abstract"
    DIMENSIONS = 3
    SEP = ','
    BEHAVIOUR_SEP = '~'
    MAGIC_NUMBER = MAGIC_NUMBER

    def __init__(self, body_parts, path, dimension=3):
        """
        Interface for data reader. This class can be implemented to integrate data files from other toolkits
        Args:
            body_parts: list of column names
            path: path to data file
            dimension: data dimension
        """
        self.body_parts = body_parts
        self.data = None
        self.path = path
        self.stats = DataStoreStats(body_parts)
        self.base_file_path = os.path.splitext(self.path)[0] if self.path is not None else None
        self.DIMENSIONS = dimension
        if os.path.exists(f'{self.base_file_path}_stats.bin'):
            self.stats: DataStoreStats = pickle.load(open(f'{self.base_file_path}_stats.bin', 'rb'))

    def get_skeleton(self, index) -> Skeleton:
        if index in self.data.index:
            return self.build_skeleton(self.data.loc[index])
        else:
            return self.build_empty_skeleton()

    def set_skeleton(self, index, skeleton: Skeleton, force_insert=False) -> None:
        insert = True
        if not force_insert and index not in self.data.index:
            insert = False
            for part in self.body_parts:
                if skeleton[part] > 0:
                    insert = True
                    break
        if insert or force_insert:
            for part in self.body_parts:
                self.set_part(index, skeleton[part])
            self.set_behaviour(index, skeleton.behaviour)

    def get_numpy(self, index):
        s = self.get_skeleton(index)
        arr = [np.array(s[part]) for part in self.body_parts]
        return np.array(arr)

    def delete_skeleton(self, index):
        if index in self.data.index:
            for part in self.body_parts:
                self.delete_part(index, part, True)

    @abstractmethod
    def set_behaviour(self, index, behaviour: str) -> None:
        pass

    @abstractmethod
    def get_behaviour(self, index) -> str:
        pass

    @abstractmethod
    def get_part_slice(self, slice_indices: list, name: str) -> np.ndarray:
        pass

    @abstractmethod
    def set_part_slice(self, slice_indices: list, name: str, data: np.ndarray) -> None:
        pass

    def row_iterator(self):
        for index, row in self.data.iterrows():
            yield index, self.build_skeleton(row)

    def part_iterator(self, part):
        for index, row in self.data[part].items():
            yield index, self.build_part(row, part)

    @abstractmethod
    def get_part(self, index, name) -> Part:
        pass

    @abstractmethod
    def set_part(self, index, part: Part) -> None:
        pass

    @abstractmethod
    def delete_part(self, index, name, force_remove=False):
        pass

    @abstractmethod
    def build_skeleton(self, row) -> Skeleton:
        pass

    @abstractmethod
    def build_part(self, arr, name) -> Part:
        pass

    def save_file(self, path: str = None) -> None:
        if path is None:
            path = self.path
        self.data.sort_index(inplace=True)
        self.data.to_csv(path, sep=self.SEP)

    def set_stats(self, stats):
        if stats.register(self.compute_data_hash()):
            del self.stats
            self.stats = stats
            pickle.dump(self.stats, open(f'{self.base_file_path}_stats.bin', 'wb'))

    def build_empty_skeleton(self):
        part_map = {}
        likelihood_map = {}
        for name in self.body_parts:
            part_map[name] = [MAGIC_NUMBER] * self.DIMENSIONS
            likelihood_map[name] = 0.0
        return Skeleton(self.body_parts, part_map=part_map, likelihood_map=likelihood_map, behaviour='',
                        dims=self.DIMENSIONS)

    def __len__(self):
        return len(self.data)

    def compute_data_hash(self):
        return int(pd.util.hash_pandas_object(self.data).sum())

    def verify_stats(self):
        if not (self.compute_data_hash() == self.stats.data_frame_hash) and (self.stats.body_parts == self.body_parts):
            self.stats.registered = False
            return False
        return True

    @staticmethod
    @abstractmethod
    def convert_to_list(index, skeleton, threshold=0.8):
        pass

    def get_header_rows(self):
        return [self.body_parts]


class DataStoreStats:

    def __init__(self, body_parts):
        self.data_frame_hash = 0
        self.body_parts = body_parts
        self.na_data_points = {}
        self.accurate_data_points = []
        self._na_current_cluster = {}
        self.occupancy_data = []
        for column in body_parts:
            self.na_data_points[column] = []
            self._na_current_cluster[column] = {'begin': -2, 'end': -2}
        self._accurate_cluster = {'begin': -2, 'end': -2}
        self.registered = False

    def add_occupancy_data(self, fraction):
        self.occupancy_data.append(fraction)

    def update_cluster_info(self, index, part, accurate=False):
        cluster = self._na_current_cluster[part] if not accurate else self._accurate_cluster
        data_point = self.na_data_points[part] if not accurate else self.accurate_data_points
        if cluster['end'] + 1 == index:
            cluster['end'] = index
        else:
            if cluster['begin'] != -2:
                data_point.append(cluster.copy())
            cluster['begin'] = cluster['end'] = index

    def register(self, data_frame_hash):
        if not self.registered:
            for col in self._na_current_cluster.keys():
                if self._na_current_cluster[col]['begin'] != -2:
                    self.na_data_points[col].append(self._na_current_cluster[col].copy())
            if self._accurate_cluster['begin'] != -2:
                self.accurate_data_points.append(self._accurate_cluster.copy())
            del self._na_current_cluster, self._accurate_cluster
            self.data_frame_hash = data_frame_hash
            self.registered = True
            return True
        return False

    def iter_na_clusters(self, part):
        for candidate in self.na_data_points[part]:
            yield candidate

    def iter_accurate_clusters(self):
        for accurate in self.accurate_data_points:
            yield accurate

    def get_accurate_cluster_info(self, bin_width=20, max_bin=100):
        histogram = {bucket: 0 for bucket in range(bin_width, max_bin + 1, 20)}
        last_bin = list(histogram.keys())[-1]
        total = 0
        for cluster in self.accurate_data_points:
            width = cluster['end'] - cluster['begin']
            target_bin = last_bin
            total += width
            for key in histogram:
                if width < key:
                    target_bin = key
                    break
            histogram[target_bin] += 1
        return len(self.accurate_data_points), histogram, total

    def get_occupancy_clusters(self, min_occupancy, max_occupancy):
        assert 0 <= min_occupancy <= max_occupancy <= 1.0
        pose_data = []
        cluster = {'begin': -2, 'end': -2}
        for index, occupancy in enumerate(self.occupancy_data):
            if min_occupancy <= occupancy <= max_occupancy:
                if cluster['end'] + 1 == index:
                    cluster['end'] = index
                else:
                    if cluster['begin'] != -2:
                        pose_data.append(cluster.copy())
                    cluster['begin'] = cluster['end'] = index
        return pose_data

    def intersect_accurate_data_points(self, accurate_clusters):
        output_accurate_cluster = []
        source_index = 0
        target_index = 0
        while source_index < len(self.accurate_data_points) and target_index < len(accurate_clusters):
            if self.accurate_data_points[source_index]['end'] < accurate_clusters[target_index]['begin']:
                source_index += 1
                continue
            if self.accurate_data_points[source_index]['begin'] > accurate_clusters[target_index]['end']:
                target_index += 1
                continue
            output_accurate_cluster.append({'begin': max(self.accurate_data_points[source_index]['begin'],
                                                         accurate_clusters[target_index]['begin']),
                                            'end': min(self.accurate_data_points[source_index]['end'],
                                                       accurate_clusters[target_index]['end'])})
            if source_index + 1 < len(self.accurate_data_points) and self.accurate_data_points[source_index + 1][
                'begin'] <= accurate_clusters[target_index]['end']:
                source_index += 1
            else:
                target_index += 1
        return output_accurate_cluster

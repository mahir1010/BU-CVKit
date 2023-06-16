import os

import numpy as np
import pandas as pd

from cvkit.pose_estimation import Skeleton, Part
from cvkit.pose_estimation.data_readers.datastore_interface import DataStoreInterface


class FlattenedDataStore(DataStoreInterface):
    FLAVOR = "flattened"
    DIMENSIONS = 3

    def __init__(self, body_parts, path, dimension=3):
        """
        Plugin to support flattened data file.Expects a csv file where all dimensions are flattened.
        The header should contain 3 consecutive columns per keypoint.
        Ex. Snout_1,Snout_2,Snout_3 for x,y, and z values
        Args:
            body_parts: list of column names
            path: path to file
        """
        super(FlattenedDataStore, self).__init__(body_parts, path, dimension=dimension)
        self.path = path
        if path is not None and os.path.exists(path):
            self.data = pd.read_csv(path, sep=',')
        else:
            columns = []
            for part in body_parts:
                columns.extend([f"{part}_{i}" for i in range(1, self.DIMENSIONS + 1)])
            self.data = pd.DataFrame(columns=columns)
        for part in body_parts:
            for dim in range(1, self.DIMENSIONS + 1):
                if f"{part}_{dim}" not in self.data.columns:
                    self.data[f"{part}_{dim}"] = ""
        if "behaviour" not in self.data.columns:
            self.data['behaviour'] = ""
        if not self.data.index.is_monotonic_increasing:
            self.data.sort_index(inplace=True)

    def save_file(self, path: str = None) -> None:
        if path is None:
            path = self.path
        self.data.sort_index(inplace=True)
        self.data.to_csv(path, index=False, sep=self.SEP)

    def delete_part(self, index, name, force_remove=False):
        if force_remove or index in self.data.index:
            self.data.loc[index, [f"{name}_{i}" for i in range(1, self.DIMENSIONS + 1)]] = pd.NA

    def set_behaviour(self, index, behaviour: list) -> None:
        self.data.loc[index, 'behaviour'] = self.BEHAVIOUR_SEP.join(behaviour)

    def get_behaviour(self, index) -> list:
        if index in self.data.index and not pd.isna(self.data.loc[index, 'behaviour']):
            return self.data.loc[index, 'behaviour'].split(self.BEHAVIOUR_SEP)
        else:
            return []

    def get_part_slice(self, slice_indices: list, name: str) -> np.ndarray:
        return self.data.loc[slice_indices[0]:slice_indices[1] - 1,
               [f"{name}_{i}" for i in range(1, self.DIMENSIONS + 1)]].apply(
            lambda x: self.build_part(x, name), axis=1).to_numpy()

    def set_part_slice(self, slice_indices: list, name: str, data: np.ndarray) -> None:
        for i in range(1, self.DIMENSIONS + 1):
            self.data.loc[slice_indices[0]:slice_indices[1] - 1, f"{name}_{i}"] = [d[i - 1] for d in data]

    def get_part(self, index, name) -> Part:
        if index in self.data.index:
            pt = np.array([self.data.loc[index, f"{name}_{i}"] for i in range(1, self.DIMENSIONS + 1)])
            if any(np.isnan(pt)):
                pt = np.array([self.MAGIC_NUMBER] * self.DIMENSIONS)
            return Part(pt, name, float(not all(pt == self.MAGIC_NUMBER)))
        else:
            return Part([self.MAGIC_NUMBER] * self.DIMENSIONS, name, 0.0)

    def set_part(self, index, part: Part) -> None:
        name = part.name
        for i in range(1, part.shape[0] + 1):
            self.data.loc[index, f"{name}_{i}"] = part[i - 1] if part[i - 1] != self.MAGIC_NUMBER else pd.NA
        if not self.data.index.is_monotonic_increasing:
            self.data.sort_index(inplace=True)

    def build_skeleton(self, row) -> Skeleton:
        part_map = {}
        likelihood_map = {}
        for name in self.body_parts:
            part_map[name] = np.array([row[f"{name}_{i}"] for i in range(1, self.DIMENSIONS + 1)])
            if any(np.isnan(part_map[name])):
                part_map[name] = np.array([self.MAGIC_NUMBER] * self.DIMENSIONS)
            likelihood_map[name] = float(not all(part_map[name] == self.MAGIC_NUMBER))
        behaviour = [] if pd.isna(row['behaviour']) else row['behaviour'].split(self.BEHAVIOUR_SEP)
        return Skeleton(self.body_parts, part_map=part_map, likelihood_map=likelihood_map,
                        behaviour=behaviour,
                        dims=self.DIMENSIONS)

    def build_part(self, row, name):
        pt = row.to_numpy()
        if any(np.isnan(pt)):
            pt = np.array([self.MAGIC_NUMBER] * self.DIMENSIONS)
        return Part(pt, name, float(not all(pt == self.MAGIC_NUMBER)))

    def get_header_rows(self):
        header = []
        for part in self.body_parts:
            header.extend([f'{part}_{i + 1}' for i in range(self.DIMENSIONS)])
        return [header]

    def part_iterator(self, part):
        for index, row in self.data.loc[:, [f'{part}_{i + 1}' for i in range(self.DIMENSIONS)]].iterrows():
            yield index, self.build_part(row, part)

    @staticmethod
    def convert_to_list(index, skeleton, threshold=0.8):
        out = []
        for part in skeleton.body_parts:
            out.extend(skeleton[part].tolist() if skeleton[part] > threshold else ['', '', ''])
        return out

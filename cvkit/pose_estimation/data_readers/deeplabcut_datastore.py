import os

import numpy as np
import pandas as pd

from cvkit.pose_estimation import Skeleton, Part
from cvkit.pose_estimation.data_readers.datastore_interface import DataStoreInterface


class DeeplabcutDataStore(DataStoreInterface):
    FLAVOR = "deeplabcut"
    DIMENSIONS = 2

    def __init__(self, body_parts, path):
        """
        Plugin for reading deeplabcut data files.
        Args:
            body_parts: list of column names
            path: path to file
        """
        super().__init__(body_parts, path)
        if path is not None and os.path.exists(path):
            self.data = pd.read_csv(path, header=[0, 1, 2], index_col=0, dtype='unicode')
        else:
            self.data = None
            for bodypart in body_parts:
                pdindex = pd.MultiIndex.from_product(
                    [["CVKit3D"], [bodypart], ["x", "y", "likelihood"]],
                    names=["scorer", "bodyparts", "coords"],
                )
                frame = pd.DataFrame(columns=pdindex)
                self.data = frame if self.data is None else pd.concat([frame, self.data], axis=1)
        self.scorer = self.data.columns[0][0]
        if (self.scorer, 'behaviour', 'name') not in self.data.columns:
            self.data[self.scorer, 'behaviour', 'name'] = ""
        for bodypart in body_parts:
            if (self.scorer, bodypart, 'x') not in self.data.columns:
                self.data[(self.scorer, bodypart, 'x')] = -1
                self.data[(self.scorer, bodypart, 'y')] = -1
                self.data[(self.scorer, bodypart, 'likelihood')] = "0"
            self.data = self.data.astype({(self.scorer, bodypart, 'x'): float, (self.scorer, bodypart, 'y'): float,
                                          (self.scorer, bodypart, 'likelihood'): float})
        # self.data.sort_index(level=[0,1,2],axis=1,inplace=True)
        if not self.data.index.is_monotonic_increasing:
            self.data.sort_index(inplace=True)

    def get_header_string(self):
        level_0 = ['scorer']
        level_0.extend(self.data.columns.get_level_values(0))
        level_1 = ['bodyparts']
        level_1.extend(self.data.columns.get_level_values(1))
        level_2 = ['coords']
        level_2.extend(self.data.columns.get_level_values(2))
        return [level_0, level_1, level_2]

    @staticmethod
    def convert_to_list(index, skeleton, threshold=0.8):
        out = [index]
        for part in skeleton.body_parts:
            out.extend(skeleton[part].tolist())
            out.append(skeleton[part].likelihood)
        return out

    def delete_part(self, index, name, force_remove=False):
        if force_remove or index in self.data.index:
            self.data.loc[index, (self.scorer, name, 'likelihood')] = 0.0

    def set_behaviour(self, index, behaviour) -> None:
        self.data.loc[index, (self.scorer, 'behaviour', 'name')] = self.BEHAVIOUR_SEP.join(behaviour)

    def get_behaviour(self, index) -> list:
        if index in self.data.index and not pd.isna(self.data.loc[index, (self.scorer, 'behaviour', 'name')]):
            return self.data.loc[index, (self.scorer, 'behaviour', 'name')].split(self.BEHAVIOUR_SEP)
        else:
            return []

    def get_part(self, index, name) -> Part:
        if index in self.data.index:
            return Part(
                [self.data.loc[index, (self.scorer, name, 'x')], self.data.loc[index, (self.scorer, name, 'y')]],
                name, self.data.loc[index, (self.scorer, name, 'likelihood')])
        else:
            return Part([self.MAGIC_NUMBER] * self.DIMENSIONS, name, 0.0)

    def set_part(self, index, part: Part) -> None:
        name = part.name
        self.data.loc[index, (self.scorer, name, 'x')] = part[0]
        self.data.loc[index, (self.scorer, name, 'y')] = part[1]
        self.data.loc[index, (self.scorer, name, 'likelihood')] = part.likelihood
        if not self.data.index.is_monotonic_increasing:
            self.data.sort_index(inplace=True)

    def part_iterator(self, part):
        for index, row in self.data.loc[:, (self.scorer, part)].iterrows():
            yield index, self.build_part(row, part)

    def build_part(self, row, name):
        return Part([row['x'], row['y']], name, row['likelihood'])

    def get_part_slice(self, slice_indices: list, name: str) -> np.ndarray:
        return self.data.loc[slice_indices[0]:slice_indices[1] - 1, (self.scorer, name)].apply(
            lambda x: self.build_part(x, name), axis=1).to_numpy()

    def set_part_slice(self, slice_indices: list, name: str, data: np.ndarray) -> None:
        self.data.loc[slice_indices[0]:slice_indices[1] - 1, (self.scorer, name, 'x')] = data[:, 0]
        self.data.loc[slice_indices[0]:slice_indices[1] - 1, (self.scorer, name, 'y')] = data[:, 1]
        self.data.loc[slice_indices[0]:slice_indices[1] - 1, (self.scorer, name, 'likelihood')] = [d.likelihood for d in
                                                                                                   data]

    def build_skeleton(self, row) -> Skeleton:
        part_map = {}
        likelihood_map = {}
        for name in self.body_parts:
            part_map[name] = [float(row[(self.scorer, name, 'x')]), float(row[(self.scorer, name, 'y')]), 0.0]
            likelihood_map[name] = float(row[(self.scorer, name, 'likelihood')])
        behaviour = [] if pd.isna(row[(self.scorer, 'behaviour', 'name')]) else row[
            (self.scorer, 'behaviour', 'name')].split(self.BEHAVIOUR_SEP)
        return Skeleton(self.body_parts, part_map=part_map, likelihood_map=likelihood_map,
                        behaviour=behaviour)

    def get_valid_marker(self, name, threshold=0.01):
        try:
            # sub_df = self.data[self.data.index.get_level_values('likelihood')>threshold]
            sub_df = self.data[self.data[(self.scorer, name, 'likelihood')] >= threshold]
            # sub_df = sub_df[sub_df['likelihood'] > threshold]
            if len(sub_df) > 0:
                return Part([sub_df[(self.scorer, name, 'x')].iloc[0], sub_df[(self.scorer, name, 'y')].iloc[0]],
                            name, sub_df[(self.scorer, name, 'likelihood')].iloc[0])
        except Exception as e:
            print(name, e)

import os

import numpy as np
import pandas as pd

from cvkit.pose_estimation import Skeleton, Part
from cvkit.pose_estimation.data_readers.datastore_interface import DataStoreInterface
from cvkit.pose_estimation.utils import convert_to_numpy


class CVKitDataStore3D(DataStoreInterface):
    FLAVOR = "CVKit3D"
    SEP = ';'

    def save_file(self, path: str = None) -> None:
        if path is None:
            path = self.path
        self.data.sort_index(inplace=True)
        self.data.to_csv(path, index=False, sep=self.SEP)

    def delete_part(self, index, name, force_remove=False):
        if force_remove or index in self.data.index:
            self.data.loc[index, name] = pd.NA

    def set_behaviour(self, index, behaviour: list) -> None:
        self.data.loc[index, 'behaviour'] = self.BEHAVIOUR_SEP.join(behaviour)

    def get_behaviour(self, index) -> list:
        if index in self.data.index and not pd.isna(self.data.loc[index, 'behaviour']):
            return self.data.loc[index, 'behaviour'].split(self.BEHAVIOUR_SEP)
        else:
            return []

    def get_part_slice(self, slice_indices: list, name: str) -> np.ndarray:
        return self.data.loc[slice_indices[0]:slice_indices[1] - 1, name].map(
            lambda x: self.build_part(x, name)).to_numpy()

    def set_part_slice(self, slice_indices: list, name: str, data: np.ndarray) -> None:
        place_holder = np.empty((data.shape[0],), dtype=np.object)
        place_holder[:] = data.tolist()
        self.data.loc[slice_indices[0]:slice_indices[1] - 1, name] = place_holder

    def get_part(self, index, name) -> Part:
        if index in self.data.index:
            pt = convert_to_numpy(self.data.loc[index, name])
            return Part(pt, name, float(not all(pt == self.MAGIC_NUMBER)))
        else:
            return Part([self.MAGIC_NUMBER] * self.DIMENSIONS, name, 0.0)

    def set_part(self, index, part: Part) -> None:
        name = part.name
        self.data.loc[index, name] = str(part.tolist())
        if not self.data.index.is_monotonic_increasing:
            self.data.sort_index(inplace=True)

    def build_skeleton(self, row) -> Skeleton:
        part_map = {}
        likelihood_map = {}
        for name in self.body_parts:
            part_map[name] = convert_to_numpy(row[name])
            likelihood_map[name] = float(not all(part_map[name] == self.MAGIC_NUMBER))
        behaviour = [] if pd.isna(row['behaviour']) else row['behaviour'].split(self.BEHAVIOUR_SEP)
        return Skeleton(self.body_parts, part_map=part_map, likelihood_map=likelihood_map,
                        behaviour=behaviour)

    def build_part(self, row, name):
        pt = convert_to_numpy(row)
        return Part(pt, name, float(not all(pt == self.MAGIC_NUMBER)))

    def __init__(self, body_parts, path):
        super(CVKitDataStore3D, self).__init__(body_parts, path)
        self.path = path
        if path is not None and os.path.exists(path):
            self.data = pd.read_csv(path, sep=';')
        else:
            self.data = pd.DataFrame(columns=body_parts)
        for part in body_parts:
            if part not in self.data.columns:
                self.data[part] = ""

        if "behaviour" not in self.data.columns:
            self.data['behaviour'] = ""
        if not self.data.index.is_monotonic_increasing:
            self.data.sort_index(inplace=True)

    @staticmethod
    def convert_to_list(index, skeleton, threshold=0.8):
        return [skeleton[part].tolist() if skeleton[part] > threshold else None for part in skeleton.body_parts]

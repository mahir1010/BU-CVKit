import csv
import os

from cvkit.pose_estimation.data_readers.cvkit_datastore import CVKitDataStore3D
from cvkit.pose_estimation.data_readers.datastore_interface import DataStoreInterface, DataStoreStats
from cvkit.pose_estimation.data_readers.deeplabcut_datastore import DeeplabcutDataStore
from cvkit.pose_estimation.data_readers.flattened_datastore import FlattenedDataStore

datastore_readers = {DeeplabcutDataStore.FLAVOR: DeeplabcutDataStore,
                     FlattenedDataStore.FLAVOR: FlattenedDataStore, CVKitDataStore3D.FLAVOR: CVKitDataStore3D}


def initialize_datastore_reader(body_parts, path, reader_type) -> DataStoreInterface:
    try:
        reader = datastore_readers[reader_type]
        return reader(body_parts, path)
    except Exception as e:
        raise Exception(f"Potentially incorrect reader type selected ({reader_type})\n" + str(e))

    return None


def convert_data_flavor(source: DataStoreInterface, target: DataStoreInterface):
    assert not os.path.exists(target.path)
    writer = csv.writer(open(target.path, 'w'), delimiter=target.SEP)
    writer.writerows(target.get_header_rows())
    for index, skeleton in source.row_iterator():
        if index % 200 == 0:
            print(f'\r{index}/{len(source)}', end='')
        writer.writerow(target.convert_to_list(index, skeleton))

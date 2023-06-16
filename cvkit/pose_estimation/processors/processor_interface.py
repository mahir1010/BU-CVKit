from abc import ABC, abstractmethod

from cvkit.pose_estimation.data_readers import DataStoreInterface


class ProcessorMetaData:
    INT = 0x00
    FLOAT = 0x01
    BOOLEAN = 0x02
    TEXT = 0x03
    BODY_PART = 0x04
    GLOBAL_CONFIG = 0x05
    FIXED_RANGE = 0x06
    NUMPY_ARRAY = 0x07
    DATA_STORE = 0x08
    VIEWS = 0x09
    FILE_MAP = 0x10
    FILE_PATH = 0x11
    DIR_PATH = 0x12

    def __init__(self, display_name, param_type, default=None, min_val=None, max_val=None, regex='', tooltip='',
                 serialize=True):
        """
        Metadata class for describing the parameters of the processor. This class will be used to generate a user interface for the processor.
        INT, FLOAT, and Text uses text-field. BOOLEAN generates a checkbox. FIXED_RANGE generates a spin-box. NUMPY_ARRAY and DATA_STORE generates a button for file picker.
        BODY_PART generates a drop-down to select defined body part. GLOBAL_CONFIG does not generate any UI. It is an indication to pass on the global software config to the
        processor.
        :param param_type: One of the predefined constants (INT, FLOAT, BOOLEAN, TEXT, BODY_PART, GLOBAL_CONFIG, FIXED_RANGE, NUMPY_ARRAY, or DATA_STORE)
        :param default: default value
        :param min_val: minimum value for INT/FLOAT and minimum length for TEXT.
        :param max_val: maximum value for INT/FLOAT and maximum length for TEXT.
        :param regex: Regular expression to validate TEXT data.
        """
        self.display_name = display_name
        self.param_type = param_type
        self.default = default
        self.min_val = min_val
        self.max_val = max_val
        self.regex = regex
        self.tooltip = tooltip
        self.serialize = serialize


class Processor(ABC):
    REQUIRES_STATS = False
    PROCESSOR_NAME = "Abstract"
    PROCESSOR_ID = "abstract"
    PROCESSOR_SUMMARY = "Summary"
    PRINT = False
    META_DATA = None
    DISTRIBUTED = False

    def __init__(self):
        self._progress = 0
        self._data_store = None
        self._data_ready = False

    @abstractmethod
    def process(self, data_store: DataStoreInterface):
        pass

    def get_progress(self):
        return self._progress

    @abstractmethod
    def get_output(self):
        pass

    def __eq__(self, other):
        if type(other) == type(self):
            for key in self.META_DATA:
                if self.__getattribute__(key) != other.__getattribute__(key):
                    return False
            return True
        return False

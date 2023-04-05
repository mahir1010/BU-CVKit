import os
import time
from abc import ABC, abstractmethod

import numpy as np


class BaseVideoReaderInterface(ABC):
    FLAVOR = "Abstract"

    def __init__(self, video_path, fps, buffer_size=128):
        self.video_path = video_path
        self.buffer_size = buffer_size
        self.base_file_path = os.path.splitext(self.video_path)[0]
        self.fps = fps
        self.total_frames = -1
        self.current_index = -1
        self.current_frame = None

    def seek_pos(self, index: int) -> None:
        self.stop()
        self.current_index = index - 1
        self.start()
        time.sleep(0.05)

    def get_current_frame(self) -> np.ndarray:
        return self.current_frame

    @abstractmethod
    def next_frame(self) -> np.ndarray:
        pass

    def get_current_index(self) -> int:
        return self.current_index

    @abstractmethod
    def release(self) -> None:
        pass

    @abstractmethod
    def pause(self) -> None:
        pass

    def get_number_of_frames(self) -> int:
        return int(self.total_frames)

    @abstractmethod
    def random_access_image(self, position):
        pass

    def __len__(self):
        return self.get_number_of_frames()

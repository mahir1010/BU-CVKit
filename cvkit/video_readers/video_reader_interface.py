import os
import time
from abc import ABC, abstractmethod

import numpy as np


class BaseVideoReaderInterface(ABC):
    """ An interface to provide intuitive access to video data. Users can implement this interface using various underlying video I/O libraries.
    We provide a few implementations using OpenCV, Deffcode, and Decord.

    :param video_path: Path of the video.
    :type video_path: str
    :param fps: The FPS of the video.
    :type fps: float
    :param buffer_size: The size of the frame pre-fetch buffer.
    :type buffer_size: int
    """

    #: A unique identifier for `BaseVideoReaderInterface` Implementation.
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
        """Resets current buffer and seeks to the frame before the given index.

        :param index: The index of the frame you want to receive next.
        :type index: int
        """
        self.stop()
        self.current_index = index - 1
        self.start()
        time.sleep(0.05)

    def get_current_frame(self) -> np.ndarray:
        """Returns previously fetched frame.

        :return: Numpy array representing the frame previously fetched from the buffer.
        :rtype: Numpy.ndarray
        """
        return self.current_frame

    @abstractmethod
    def next_frame(self) -> np.ndarray:
        """Returns the next frame from the buffer.

        :return: Numpy array representing the newly fetched frame.
        :rtype: Numpy.ndarray
        """
        pass

    def get_current_index(self) -> int:
        """Get the frame number of the newest frame fetched from the buffer.

        :return: Frame number of current frame.
        :rtype: int
        """
        return self.current_index

    @abstractmethod
    def release(self) -> None:
        """Release file resources and clear buffer.

        """
        pass

    @abstractmethod
    def pause(self) -> None:
        """Pause pre-fetching frames to the buffer.

        """
        pass

    def get_number_of_frames(self) -> int:
        """Returns the total number of frames in the video.

        :return: Total number of frames.
        :rtype: int
        """
        return int(self.total_frames)

    @abstractmethod
    def random_access_image(self, position) -> np.ndarray:
        """Fetch frame from the video at random position without affecting the current buffer. It is different from the seek function since it does not reset the internal buffer and does not change the current frame number.

        :param position: index of the desired frame.
        :type position: int
        :return: Numpy array representing the newly fetched frame at given position.
        :rtype: Numpy.ndarray
        """
        pass

    def delete_frame(self,position):
        raise NotImplementedError("Frame Deletion not supported")

    def __len__(self):
        return self.get_number_of_frames()

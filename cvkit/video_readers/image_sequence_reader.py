import os
import tempfile
from glob import glob

import cv2
import numpy as np

from cvkit.video_readers.video_reader_interface import BaseVideoReaderInterface


def generate_image_sequence_reader(video_path, fps, frame_numbers, output_path=None):
    if output_path == None:
        directory = tempfile.TemporaryDirectory()
        directory_path = tempfile.name
    else:
        directory = directory_path = output_path
        os.makedirs(directory_path, exist_ok=True)
    reader = cv2.VideoCapture(video_path)
    for index, frame_number in enumerate(frame_numbers):
        if not os.path.exists(os.path.join(directory_path, f'{frame_number}.png')):
            reader.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = reader.read()
            if ret:
                cv2.imwrite(os.path.join(directory_path, f'{frame_number}.png'),frame)
    return ImageSequenceReader(directory, fps)


class ImageSequenceReader(BaseVideoReaderInterface):
    """This implementation interprets a folder of images as a video stream. This can be useful when reading images from the datasets where the videos are stored as individual frames.

    :param video_path: Path to the folder containing images
    :type video_path: str
    :param fps: FPS of the video stream
    :type fps: float
    :param file_formats: list of valid glob patterns for supported images.
    :type file_formats: list[str]
    """
    def random_access_image(self, position):
        if 0 <= position < self.total_frames:
            return cv2.cvtColor(cv2.imread(self.images[position]), cv2.COLOR_BGR2RGB)

    FLAVOR = "Images"

    def seek_pos(self, index: int) -> None:
        self.frame_number = index - 1

    def next_frame(self) -> np.ndarray:
        self.frame_number += 1
        self.current_frame = cv2.cvtColor(cv2.imread(self.images[self.frame_number]), cv2.COLOR_BGR2RGB)
        return self.current_frame

    def release(self) -> None:
        if type(self.directory) == tempfile.TemporaryDirectory:
            self.directory.cleanup()

    def pause(self) -> None:
        pass

    def __init__(self, video_path, fps, file_formats=['[jJ][pP][gG]', '[pP][nN][gG]', '[bB][mM][pP]']):

        if type(video_path) == tempfile.TemporaryDirectory:
            super(ImageSequenceReader, self).__init__(video_path.name, fps)
        else:
            super(ImageSequenceReader, self).__init__(video_path, fps)
        self.directory = video_path
        self.images = []
        for file_format in file_formats:
            self.images.extend(glob(os.path.join(self.video_path, '*.{}'.format(file_format))))
        self.total_frames = len(self.images)
        self.frame_number = 0

import sys
import time
from queue import Queue, Empty
from threading import Thread

import cv2
import numpy as np

from cvkit.video_readers.video_reader_interface import BaseVideoReaderInterface


class CV2VideoReader(BaseVideoReaderInterface):

    def random_access_image(self, position):
        if 0 <= position < self.total_frames:
            stream = cv2.VideoCapture(self.video_path)
            stream.set(cv2.CAP_PROP_POS_FRAMES, position)
            ret, frame = stream.read()
            stream.release()
            print(sys.getsizeof(frame))
            if ret:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    FLAVOR = "opencv"

    def __init__(self, video_path, fps, buffer_size=64):
        super().__init__(video_path, fps, buffer_size)
        self.state = 0
        self.thread = None
        self.buffer = Queue(maxsize=buffer_size)
        self.stream = cv2.VideoCapture(self.video_path)
        self.total_frames = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))

    def start(self):
        self.stream.set(cv2.CAP_PROP_POS_FRAMES, self.current_index + 1)
        self.thread = Thread(target=self.fill_buffer)
        self.thread.daemon = True
        self.state = 1
        self.thread.start()

    def fill_buffer(self):
        while True:
            if self.state <= 0:
                break
            if not self.buffer.full():
                ret, frame = self.stream.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.buffer.put(frame)
            else:
                time.sleep(0.01)

    def stop(self):
        if self.thread:
            self.state = -1
            self.thread.join()
            with self.buffer.mutex:
                self.buffer.queue.clear()
        self.thread = None

    def pause(self) -> None:
        self.state = 0

    def release(self):
        self.stop()
        self.stream.release()

    def next_frame(self) -> np.ndarray:
        if self.state == -1:
            return None
        elif self.state != 1:
            self.start()
        try:
            self.current_frame = self.buffer.get(timeout=0.5)
            self.current_index += 1
            return self.current_frame
        except Empty:
            self.stop()
            return None

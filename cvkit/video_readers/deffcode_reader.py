import json
import time
from datetime import timedelta
from queue import Queue, Empty
from threading import Thread

import numpy as np
from deffcode import FFdecoder

from cvkit.video_readers.video_reader_interface import BaseVideoReaderInterface


class DeffcodeVideoReader(BaseVideoReaderInterface):
    FLAVOR = "deffcode"

    def random_access_image(self, position):
        if 0 <= position < self.total_frames:
            ts = self.get_timestamp(position)
            stream = FFdecoder(self.video_path, **{'-ss': ts}).formulate()
            frame = next(stream.generateFrame(), None)
            stream.terminate()
            if frame is not None:
                return frame

    def get_timestamp(self, frame_number):
        return str(timedelta(seconds=(frame_number / self.fps)))

    def __init__(self, video_path, fps, buffer_size=128):
        super().__init__(video_path, fps, buffer_size)
        self.state = 0
        self.thread = None
        self.buffer = Queue(maxsize=buffer_size)
        self.stream = FFdecoder(self.video_path).formulate()
        self.total_frames = json.loads(self.stream.metadata)['approx_video_nframes']
        self.stream.terminate()
        self.current_index = -1

    def start(self):
        if self.thread is None:
            with self.buffer.mutex:
                self.buffer.queue.clear()
        ts = self.get_timestamp(self.current_index + 1)
        self.stream = FFdecoder(self.video_path, **{'-ss': ts}).formulate()
        self.thread = Thread(target=self.fill_buffer)
        self.thread.daemon = True
        self.state = 1
        self.thread.start()

    def fill_buffer(self):
        while True:
            if self.state <= 0:
                break
            if not self.buffer.full():
                frame = next(self.stream.generateFrame(), None)
                self.buffer.put(frame)
            else:
                time.sleep(0.01)

    def stop(self):
        if self.thread:
            self.state = -1
            self.thread.join()
        self.stream.terminate()
        self.thread = None

    def pause(self) -> None:
        self.state = 0

    def release(self):
        self.stop()

    def next_frame(self) -> np.ndarray:
        if self.state == -1:
            return None
        elif self.state != 1:
            self.start()
        try:
            self.current_frame = self.buffer.get()
            self.current_index += 1
            return self.current_frame
        except Empty:
            self.stop()
            return None

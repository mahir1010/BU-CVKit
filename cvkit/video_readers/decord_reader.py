import time
from queue import Queue, Empty
from threading import Thread

import decord
import numpy as np

from cvkit.video_readers.video_reader_interface import BaseVideoReaderInterface


class DecordReader(BaseVideoReaderInterface):

    def random_access_image(self, position):
        if 0 <= position < self.total_frames:
            return self.stream[position].asnumpy()

    FLAVOR = "decord"

    def __init__(self, video_path, fps, buffer_size=64, **params):
        super().__init__(video_path, fps, buffer_size)
        self.cpu = not params.get('use_gpu', False)
        if self.cpu:
            self.ctx = decord.cpu(params.get('cpu_id', 0))
        else:
            self.ctx = decord.gpu(params.get('gpu_id', 0))
        self.state = 0
        self.thread = None
        self.buffer = Queue(maxsize=buffer_size)
        self.batch_fetch = params.get('batch_fetch', 1)
        self.fetch_index = 0
        self.stream = decord.VideoReader(self.video_path, self.ctx)
        self.total_frames = self.stream.__len__()

    def start(self):
        self.stream.seek_accurate(self.current_index + 1)
        self.fetch_index = self.current_index + 1
        self.thread = Thread(target=self.fill_buffer)
        self.thread.daemon = True
        self.state = 1
        self.thread.start()

    def fill_buffer(self):
        while True:
            if self.state <= 0:
                break
            if not self.buffer.full():
                if self.batch_fetch > 1:
                    frames = self.stream.get_batch(
                        range(self.fetch_index, self.fetch_index + min(self.batch_fetch, len(self.stream)))).asnumpy()
                    self.fetch_index += min(self.batch_fetch, len(self.stream))
                    for frame in frames:
                        self.buffer.put(frame)
                else:
                    self.buffer.put(self.stream.next().asnumpy())
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

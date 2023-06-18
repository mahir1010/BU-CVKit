from cvkit.video_readers.video_reader_interface import BaseVideoReaderInterface

video_readers = {}
try:
    from cvkit.video_readers.cv2_reader import CV2VideoReader

    video_readers[CV2VideoReader.FLAVOR] = CV2VideoReader
except:
    pass
try:
    from cvkit.video_readers.deffcode_reader import DeffcodeVideoReader

    video_readers[DeffcodeVideoReader.FLAVOR] = DeffcodeVideoReader
except:
    pass
try:
    from cvkit.video_readers.image_sequence_reader import ImageSequenceReader

    video_readers[ImageSequenceReader.FLAVOR] = ImageSequenceReader
except:
    pass


def initialize_video_reader(video_path, fps, reader_type):
    """Detects underlying class based on reader_type. And generates appropriate :py:class:`~cvkit.video_readers.video_reader_interface.BaseVideoReaderInterface` subclass instance.

    :param video_path: Path to the target video file.
    :type video_path: str
    :param fps: The FPS of the video.
    :type fps: float
    :param reader_type: A string to identify underlying video reader type. Refer to the :py:attr:`~cvkit.video_readers.video_reader_interface.BaseVideoReaderInterface.FLAVOR` attributes of each implementation.
    :type reader_type: str
    :return: :py:class:`~cvkit.video_readers.video_reader_interface.BaseVideoReaderInterface` subclass instance.
    :rtype: :py:class:`~cvkit.video_readers.video_reader_interface.BaseVideoReaderInterface`
    """
    try:
        return video_readers[reader_type](video_path, fps)
    except KeyError:
        raise Exception(f"{reader_type} flavor is not installed.")
    except Exception as e:
        raise Exception(f"Error while initializing video reader ({reader_type})" + str(e))
    return None

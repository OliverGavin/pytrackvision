import cv2

from .camera_stream import CameraStream


class OpenCVVideoStream(CameraStream):
    """
    Context manager and iterator for accessing video from a file.

    >>> with OpenCVVideoStream() as stream:
    >>>     for img in stream:
    >>>         ...

    """
    def __init__(self, src, *args, **kwargs):
        CameraStream.__init__(self, *args, **kwargs)
        self._src = src

    def _create_camera(self):
        self._camera = cv2.VideoCapture(self._src)

    def _read_frame(self):
        import time
        time.sleep(0.05)
        _, frame = self._camera.read()
        if frame is None:
            self._camera = cv2.VideoCapture(self._src)
            _, frame = self._camera.read()
        return frame

    def _release_camera(self):
        self._camera.release()

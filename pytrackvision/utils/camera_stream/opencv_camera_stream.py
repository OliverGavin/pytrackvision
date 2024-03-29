import cv2

from .camera_stream import CameraStream


class OpenCVCameraStream(CameraStream):
    """
    Context manager and iterator for accessing webcam.

    >>> with OpenCVCameraStream() as stream:
    >>>     for img in stream:
    >>>         ...

    """

    def _create_camera(self):
        self._camera = cv2.VideoCapture(self.camera_num)
        self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

    def _read_frame(self):
        # import time
        # time.sleep(0.05)
        _, frame = self._camera.read()
        return frame

    def _release_camera(self):
        self._camera.release()

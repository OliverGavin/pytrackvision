from picamera import PiCamera
from picamera.array import PiRGBArray

from .camera_stream import CameraStream


class PiCameraStream(CameraStream):
    """
    Context manager and iterator for accessing webcam.

    >>> with PiCameraStream() as stream:
    >>>     for img in stream:
    >>>         ...

    """

    def _create_camera(self):
        self._camera = PiCamera(camera_num=self.camera_num, resolution=self.resolution, framerate=self.framerate)
        self._raw_capture = PiRGBArray(self._camera, size=self.resolution)
        self._stream = self._camera.capture_continuous(self._raw_capture, format="bgr", use_video_port=True)

    def _read_frame(self):
        frame = next(self._stream)
        self._raw_capture.truncate(0)
        return frame.array

    def _release_camera(self):
        self._stream.close()
        self._raw_capture.close()
        self._camera.close()

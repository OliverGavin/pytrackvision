import numpy
import pygame
import pygame.camera

from .camera_stream import CameraStream


class PyGameCameraStream(CameraStream):
    """
    Context manager and iterator for accessing webcam.

    >>> with PyGameCameraStream() as stream:
    >>>     for img in stream:
    >>>         ...

    """

    def _create_camera(self):
        pygame.init()
        pygame.camera.init()
        self._camera = pygame.camera.Camera(f"/dev/video{self.camera_num}", (self.resolution[0], self.resolution[1]))
        self._camera.start()

    def _read_frame(self):
        frame = self._camera.get_image()
        frame = pygame.surfarray.pixels3d(frame)[:, :, ::-1]
        frame = numpy.rot90(frame, 3).copy()
        return frame

    def _release_camera(self):
        self._camera.stop()
        pygame.quit()

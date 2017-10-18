import os

from .camera_stream import CameraStream


def get_camera_stream(*args, **kwargs):
    """Return an instance of :class:`.camera_stream.CameraStream`.

    An appropriate implementation is chosen based on wether a picamera
    or a webcam is available.

    Parameters
    ----------
    camera_num : int
        index camera to use (for multicamera setups).
    resolution : tuple[int, int]
        frame resolution
    framerate : int
        rate of frame capture
    multi_thread : bool
        enable a worker thread for pulling new frames

    Returns
    -------
    CameraStream
        A consistent interface to either a webcam or picamera

    """
    if 'raspberrypi' in os.uname():
        from .pi_camera_stream import PiCameraStream
        return PiCameraStream(*args, **kwargs)
    else:
        from .web_camera_stream import WebCameraStream
        return WebCameraStream(*args, **kwargs)


__all__ = ['CameraStream', 'get_camera_stream']

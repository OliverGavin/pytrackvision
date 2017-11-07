import os

from .camera_stream import CameraStream


def get_camera_stream(*args, **kwargs):
    """Return an instance of :class:`.camera_stream.CameraStream`.

    An appropriate implementation is chosen based on wether a picamera
    or a webcam and either opencv or pygame is available.

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
        A consistent interface to either a webcam (opencv or pygame) or picamera

    Raises
    ------
    ModuleNotFoundError
        If no suitable camera module is compatible/installed

    """
    camera_funcs = [_get_pi_camera_stream, _get_opencv_camera_stream, _get_pygame_camera_stream]

    for fun in camera_funcs:
        try:
            print(fun.__name__)
            test_kwargs = kwargs
            test_kwargs['multi_thread'] = False
            with fun(*args, **test_kwargs) as stream:
                img = next(stream)
                if img is not None:
                    return fun(*args, **kwargs)

        except ModuleNotFoundError:
            pass

    if 'raspberrypi' in os.uname():
        raise ModuleNotFoundError('Please install the picamera module.')
    else:
        raise ModuleNotFoundError('Please install opencv or pygame.')


def _get_pi_camera_stream(*args, **kwargs):
    from .pi_camera_stream import PiCameraStream
    return PiCameraStream(*args, **kwargs)


def _get_opencv_camera_stream(*args, **kwargs):
    from .opencv_camera_stream import OpenCVCameraStream
    return OpenCVCameraStream(*args, **kwargs)


def _get_pygame_camera_stream(*args, **kwargs):
    from .pygame_camera_stream import PyGameCameraStream
    return PyGameCameraStream(*args, **kwargs)


__all__ = ['CameraStream', 'get_camera_stream']

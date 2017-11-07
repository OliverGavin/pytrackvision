import time
from abc import ABC, abstractmethod
from threading import Thread


class CameraStream(ABC):
    """
    Context manager and iterator for accessing webcam and picam with a consistent interface.

    >>> with ConcreteCameraStream() as stream:
    >>>     for img in stream:
    >>>         ...

    """

    def __init__(self, camera_num=0, resolution=(320, 240), framerate=30, multi_thread=True):
        """Constructor.

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

        """
        self._camera_num = camera_num
        self._resolution = resolution
        self._framerate = framerate
        self._multi_thread = multi_thread

        self._frame = None
        self._frame_waiting = False
        self._release = False
        self._thread = Thread(target=self._update, args=()) if self._multi_thread else None

    def __enter__(self):
        """Acquire resources.

        Creates a thread, if enabled, to pull camera frames asynchronously
        """
        self._create_camera()
        if self._multi_thread:
            self._thread.start()
            while(self._frame is None):
                time.sleep(0.1)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Release resources.

        Notify the thread to stop and wait for it to join.
        Release the camera.
        """
        self._release = True
        if self._multi_thread:
            self._thread.join()
        self._release_camera()

    def __iter__(self):
        """Return an indefinite iterator.
        """
        return self

    def __next__(self):
        """Return the newest frame available.

        If the frame has already been read the thread waits for the next one.

        Returns
        -------
        numpy.ndarray
            Frame from camera in BGR

        """
        if self._multi_thread:
            while(self._frame_waiting is False):
                time.sleep(0.005)
            frame = self._frame
        else:
            frame = self._read_frame()
        self._frame_waiting = False
        return frame

    def next(self):
        """Return the newest frame available.

        Returns
        -------
        numpy.ndarray
            Frame from camera in BGR

        """
        return self.__next__()

    def _update(self):
        while True:
            if self._release:
                return
            self._frame = self._read_frame()
            self._frame_waiting = True

    @property
    def camera_num(self):
        """Get camera_num.
        """
        return self._camera_num

    @property
    def resolution(self):
        """Get resolution.
        """
        return self._resolution

    @property
    def framerate(self):
        """Get framerate.
        """
        return self._framerate

    @abstractmethod
    def _create_camera(self):
        """Create camera resources callback.
        """
        ...

    @abstractmethod
    def _read_frame(self):
        """Get the next frame from the camera resource.

        Returns
        -------
        numpy.ndarray
            Frame from camera in BGR

        """
        ...

    @abstractmethod
    def _release_camera(self):
        """Release camera resources callback.
        """
        ...

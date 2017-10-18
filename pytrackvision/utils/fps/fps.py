from collections import deque
from datetime import datetime


class FPS:
    """Simple class for measuring frame rate."""

    def __init__(self, frame_buffer_size=10):
        """Constructor.

        A double ended queue is used to store the time when a new frame occurs.
        A maximum frame_buffer_size is used, removing times that are no longer
        needed for calculating the fps.

        Parameters
        ----------
        frame_buffer_size : int
            The number of frames to use when calculating the frame rate
        
        """
        self._frame_buffer_size = frame_buffer_size
        self._time_buffer = deque(maxlen=self._frame_buffer_size)
        self._fps = 0

    @property
    def fps(self):
        """Get the current fps."""
        return self._fps

    def next(self):
        """Notify FPS that the next frame has appeared.

        Pushes the current time onto the double ended queue (poping the oldest).
        The frame rate is updated based on the number of frames divided by the
        difference, in seconds, between the newest and oldest frame.
        """
        now = datetime.now()
        self._time_buffer.append(now)
        if len(self._time_buffer) == self._frame_buffer_size:
            then = self._time_buffer[0]
            self._fps = self._frame_buffer_size / (now - then).total_seconds()

    def reset(self):
        """Reset all measurements and sets the fps to zero."""
        self._time_buffer.clear()
        self._fps = 0

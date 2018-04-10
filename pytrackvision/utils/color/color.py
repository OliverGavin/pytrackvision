import cv2
import numpy as np


class Range:

    def __init__(self, range):
        """Store a pair of values defining a range with a min and max.
        """
        self.min = range[0]
        self.max = range[1]


class ColorSpaceRange:

    def __init__(self, colors):
        """Store a range for values in a color space.

        Parameters
        ----------
        colors: Dict[str: Tuple[int]]
            A map of color space labels and a range of values.
        """
        self._colors = {k: Range(v) for k, v in colors.items()}

    def __getitem__(self, key):
        return self._colors[key]

    @property
    def colors(self):
        """
        Returns
        -------
        Dict[str: Range]
        """
        return self._colors


def extract_face_color(roi):
    roi = roi.copy()
    size = len(roi)
    pts = [
                    (1/3, 1/6), (1/2, 1/6), (2/3, 1/6),
       (3/12, 3/6), (1/3, 3/6), (1/2, 3/6), (2/3, 3/6), (9/12, 3/6),
                    (1/3, 4/6), (1/2, 4/6), (2/3, 4/6),
    ]
    w = 5
    h = 5
    pt_samples = []
    for x, y in pts:
        x = int(size * x)
        y = int(size * y)
        pt_roi = roi[y: y + h, x: x + w]
        average_color = [np.median(pt_roi[:, :, i]) for i in range(pt_roi.shape[-1])]
        pt_samples.append(average_color)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 1)

    lower = np.min(pt_samples, axis=0)
    upper = np.max(pt_samples, axis=0)
    # lower = np.min(pt_samples, axis=0)*.75 + np.median(pt_samples, axis=0)*.25
    # upper = np.max(pt_samples, axis=0)*.75 + np.median(pt_samples, axis=0)*.25

    cv2.imshow('pic', roi)
    return lower, upper


def create_color_range_slider(title, color_space_range):
    """Create a slider to adjust the color space range.

    Parameters
    ----------
    title: str
    color_space_range: ColorSpaceRange
    """

    def _update(color_range, attr):
        def _set(v):
            if attr == 'min':
                color_range.min = v
            elif attr == 'max':
                color_range.max = v

        return _set

    cv2.namedWindow(title)
    for label, color_range in color_space_range.colors.items():
        cv2.createTrackbar(f'{label} min', title, color_range.min, 255, _update(color_range, 'min'))
        cv2.createTrackbar(f'{label} max', title, color_range.max, 255, _update(color_range, 'max'))

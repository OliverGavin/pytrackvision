import os
from operator import itemgetter

import cv2


face_cascade = cv2.CascadeClassifier(
    os.path.join(os.path.dirname(__file__),
                 'haarcascades/haarcascade_frontalface_default.xml')
)


class FaceTracker:

    def __init__(self):
        """Create a face tracker.
        """
        self._tracker = None
        self._found = False
        self._i = 0

    def track(self, img):
        """Find and track the primary face in the image.
        """
        self._i = (self._i + 1) % 10  # reset counter every 10 frames
        bbox = None
        jones = False

        # If no face is being tracked or it's time to reset the tracker
        if not self._found or self._i == 0:
            # Convert the image to gray scale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Search for a face using the Viola Jones Cascade Classifier
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            # If a face was found
            if len(faces):
                jones = True
                # Select the largest face
                bbox = tuple(max(faces, key=itemgetter(2)))
                # Update the light flow based tracker
                self._tracker = cv2.TrackerMedianFlow_create()
                if self._tracker.init(img, bbox):
                    self._found = True

        # If a face is being tracked using light flow
        # (and wasn't just found with Viola Jones)
        if self._found and not jones:
            # Update the tracker for the next frame
            self._found, bbox = self._tracker.update(img)
            bbox = tuple([int(d) for d in bbox])

        if self._found:
            return bbox
        else:
            # Don't return stale face trackings
            return None

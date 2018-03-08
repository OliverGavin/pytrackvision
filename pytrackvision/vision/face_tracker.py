from operator import itemgetter
import os

import cv2


face_cascade = cv2.CascadeClassifier(os.path.join(os.path.dirname(__file__),
                                     'haarcascades/haarcascade_frontalface_default.xml'))


class FaceTracker:

    def __init__(self):
        self._tracker = None
        self._found = False
        self._i = 0

    def track(self, img):
        bbox = None
        self._i = (self._i + 1) % 10
        jones = False

        if not self._found or self._i == 0:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            if len(faces):
                jones = True
                # bbox = tuple(faces[0])
                bbox = tuple(max(faces, key=itemgetter(2)))
                # TODO improve finding the main face and tracking it... or multiple??
                self._tracker = cv2.TrackerMedianFlow_create()
                if self._tracker.init(img, bbox):
                    self._found = True

        if self._found and not jones:
            self._found, bbox = self._tracker.update(img)
            bbox = tuple([int(d) for d in bbox])

        if self._found:
            return [bbox]
        else:
            return []

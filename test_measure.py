import unittest

import cv2
import numpy as np

from measure import detect, pixels_per_mm


class MeasurementTests(unittest.TestCase):
    def test_detect_and_calibrate(self):
        image = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.rectangle(image, (150, 100), (450, 300), (255, 255, 255), -1)
        detections = detect(image, min_area=100)
        self.assertTrue(detections)
        self.assertTrue(295 < detections[0].width_px < 310)
        self.assertTrue(195 < detections[0].height_px < 210)
        self.assertTrue(14.7 < pixels_per_mm(detections[0], 20) < 15.5)


if __name__ == "__main__":
    unittest.main()

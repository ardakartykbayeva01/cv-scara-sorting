"""
camera.py  (Student 2)
Reads one image from the CoppeliaSim vision sensor as an OpenCV BGR frame.
"""

import cv2
import numpy as np


def grab_frame(sim, cam):
    """Current image of the vision sensor 'cam' (BGR, rows from the top)."""
    img, res = sim.getVisionSensorImg(cam)
    w, h = res
    frame = np.frombuffer(img, dtype=np.uint8).reshape(h, w, 3)
    frame = cv2.flip(frame, 0)                       # CoppeliaSim images are upside down
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)    # OpenCV uses BGR

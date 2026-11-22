"""
detect.py  (Student 2: Computer Vision)
Finds red, green and blue cubes in a BGR image.

Data contract (agreed with the control module):
  detect_objects(frame) -> list of dicts
      {"color": "red", "center_px": [u, v], "bbox_px": [x, y, w, h], "area_px": 676}
  - u = column (pixels from the left), v = row (pixels from the top)
  - only pixel values; the conversion to metres is done by calibration (pixel_to_robot)
  - an empty list means that no object was found
"""

import cv2
import numpy as np

from config import HSV_RANGES, MIN_AREA_PX, MAX_AREA_PX, KERNEL_SIZE, BLUR_KSIZE


def color_mask(hsv, color):
    """Binary mask (0/255) of one colour, cleaned with morphology."""
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in HSV_RANGES[color]:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)    # remove small noise dots
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)   # fill small holes
    return mask


def detect_objects(frame, blur=None):
    """Detect coloured cubes in a BGR frame (NumPy array).
    blur: Gaussian kernel size (odd number, 0 = off). None -> BLUR_KSIZE from config."""
    if frame is None or frame.size == 0:
        return []
    k = BLUR_KSIZE if blur is None else blur
    if k > 0:
        frame = cv2.GaussianBlur(frame, (k, k), 0)          # reduce pixel noise
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    objects = []
    for color in HSV_RANGES:
        mask = color_mask(hsv, color)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if not (MIN_AREA_PX <= area <= MAX_AREA_PX):
                continue
            m = cv2.moments(c)                              # centre = centroid of the contour
            u = m["m10"] / m["m00"]
            v = m["m01"] / m["m00"]
            x, y, w, h = cv2.boundingRect(c)
            objects.append({"color": color,
                            "center_px": [float(u), float(v)],
                            "bbox_px": [int(x), int(y), int(w), int(h)],
                            "area_px": float(area)})
    return objects


def draw_detections(frame, objects):
    """Copy of the frame with boxes, labels and centres (for debugging and the report)."""
    out = frame.copy()
    colors = {"red": (0, 0, 255), "green": (0, 160, 0), "blue": (255, 0, 0)}
    for o in objects:
        x, y, w, h = o["bbox_px"]
        u, v = o["center_px"]
        c = colors[o["color"]]
        cv2.rectangle(out, (x - 3, y - 3), (x + w + 3, y + h + 3), c, 2)
        cv2.circle(out, (int(round(u)), int(round(v))), 3, (0, 0, 0), -1)
        cv2.putText(out, f'{o["color"]} ({u:.0f},{v:.0f})', (x - 3, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
    return out

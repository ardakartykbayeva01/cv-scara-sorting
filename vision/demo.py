"""
demo.py  (Student 2)
Runs the detector on an image file, prints the result and saves an annotated copy.
Works without CoppeliaSim.

  python3 demo.py camera_frame.png
"""

import sys

import cv2

from detect import detect_objects, draw_detections

path = sys.argv[1] if len(sys.argv) > 1 else "camera_frame.png"
frame = cv2.imread(path)
if frame is None:
    raise SystemExit(f"Cannot read {path}")

objects = detect_objects(frame)
print(f"{len(objects)} object(s) found in {path}:")
for o in objects:
    print("  ", o)

out_path = path.rsplit(".", 1)[0] + "_detected.png"
cv2.imwrite(out_path, draw_detections(frame, objects))
print("Annotated image saved:", out_path)

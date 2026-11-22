"""
config.py
Settings of the vision module: HSV colour ranges and area limits.

OpenCV HSV ranges: H = 0..179, S = 0..255, V = 0..255.
Red sits at both ends of the hue circle, so it needs two ranges.
The white mat, the grey floor and the grey robot have low saturation (S),
so a minimum S removes them.
"""

HSV_RANGES = {
    "red":   [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (179, 255, 255))],
    "green": [((40, 120, 70), (85, 255, 255))],
    "blue":  [((100, 120, 70), (130, 255, 255))],
}

MIN_AREA_PX = 80        # smaller blobs are noise
MAX_AREA_PX = 5000      # bigger blobs are not cubes
KERNEL_SIZE = 3         # size of the morphological opening/closing kernel

# Gaussian blur before HSV (experiment E3). 0 = no blur (used in E1 and E2).
# Blur averages neighbouring pixels, so random noise is reduced before thresholding.
BLUR_KSIZE = 0

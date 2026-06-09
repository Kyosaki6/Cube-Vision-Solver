import cv2
import numpy as np

# We'll define standard reference colors in LAB space for CIEDE2000 comparison.
# First, let's define their typical BGR values (similar to what might be seen on camera)
# and convert them to LAB.
# These values can be calibrated if needed.
REFERENCE_BGR = {
    'white':  (230, 230, 230),
    'red':    (0, 0, 180),
    'orange': (0, 100, 230),
    'yellow': (0, 210, 210),
    'green':  (0, 180, 0),
    'blue':   (180, 0, 0)
}

# Convert reference BGR colors to LAB
REFERENCE_LAB = {}
for color_name, bgr_val in REFERENCE_BGR.items():
    bgr_np = np.uint8([[bgr_val]])
    lab_np = cv2.cvtColor(bgr_np, cv2.COLOR_BGR2LAB)
    REFERENCE_LAB[color_name] = lab_np[0][0]

# Mapping color names to BGR tuples for UI rendering
COLOR_BGR = {
    'white':  (255, 255, 255),
    'red':    (0, 0, 255),
    'orange': (0, 128, 255), # BGR for orange
    'yellow': (0, 255, 255),
    'green':  (0, 255, 0),
    'blue':   (255, 0, 0),
    'unknown':(128, 128, 128)
}

# Fallback patch for colormath in case numpy has removed `asscalar`
import numpy as np
if not hasattr(np, 'asscalar'):
    np.asscalar = lambda a: a.item()

from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000

def ciede2000_distance(lab1, lab2):
    """
    Calculate CIEDE2000 distance between two OpenCV LAB colors.
    OpenCV LAB ranges: L (0-255 mapped to 0-100), a (0-255 mapped to -127 to 127), b (0-255 mapped to -127 to 127).
    """
    # Convert OpenCV LAB to standard LAB ranges
    l1 = (lab1[0] * 100.0) / 255.0
    a1 = lab1[1] - 128.0
    b1 = lab1[2] - 128.0
    
    l2 = (lab2[0] * 100.0) / 255.0
    a2 = lab2[2] - 128.0
    b2 = lab2[2] - 128.0
    
    # We should fix the index mapping
    a2 = lab2[1] - 128.0
    b2 = lab2[2] - 128.0

    color1 = LabColor(lab_l=l1, lab_a=a1, lab_b=b1)
    color2 = LabColor(lab_l=l2, lab_a=a2, lab_b=b2)
    
    return delta_e_cie2000(color1, color2)

def get_dominant_color(roi):
    """
    Given a small BGR image region (ROI), determine the most likely Rubik's cube color
    using CIEDE2000 in the LAB color space.
    """
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(roi, (5, 5), 0)
    lab_roi = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    
    # Calculate the median LAB values in the ROI to be robust against outliers
    median_l = np.median(lab_roi[:, :, 0])
    median_a = np.median(lab_roi[:, :, 1])
    median_b = np.median(lab_roi[:, :, 2])
    
    median_lab = np.array([median_l, median_a, median_b], dtype=np.float32)
    
    best_color = 'unknown'
    min_dist = float('inf')
    
    for color_name, ref_lab in REFERENCE_LAB.items():
        # ref_lab is uint8, median_lab is float32
        dist = ciede2000_distance(median_lab, ref_lab)
        if dist < min_dist:
            min_dist = dist
            best_color = color_name
            
    return best_color

def extract_colors(frame, regions, rect_size):
    """
    Extracts colors from the frame at the specified regions.
    Returns a list of 9 color names.
    """
    colors = []
    for (x, y) in regions:
        # Extract ROI
        roi = frame[y:y+rect_size, x:x+rect_size]
        color = get_dominant_color(roi)
        colors.append(color)
    return colors

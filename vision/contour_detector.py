import cv2
import numpy as np

def preprocess(frame):
    """Preprocess frame for contour detection: grayscale -> blur -> Canny -> dilate."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.blur(gray, (3, 3))
    canny = cv2.Canny(blurred, 30, 60, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    dilated = cv2.dilate(canny, kernel)
    return dilated

def find_sticker_contours(dilated_frame):
    """
    Find and validate exactly 9 sticker contours.
    Returns sorted list of (x, y, w, h) or empty list.
    """
    contours, _ = cv2.findContours(dilated_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.1 * peri, True)
        if len(approx) != 4:
            continue

        x, y, w, h = cv2.boundingRect(approx)
        ratio = w / float(h)
        if not (0.8 <= ratio <= 1.2):
            continue
        if not (30 <= w <= 60):
            continue

        area = cv2.contourArea(c)
        solidity = area / float(w * h)
        if solidity < 0.4:
            continue

        candidates.append((x, y, w, h))

    if len(candidates) < 9:
        return []

    # Build neighbor map: find which contour has 9 neighbors (itself + 8 others)
    # by checking if neighbor positions fall inside other contours.
    contour_neighbors = {}
    for i, (x, y, w, h) in enumerate(candidates):
        contour_neighbors[i] = []
        cx = x + w / 2
        cy = y + h / 2
        radius = 1.5

        neighbor_positions = [
            (cx - w * radius, cy - h * radius),
            (cx,              cy - h * radius),
            (cx + w * radius, cy - h * radius),
            (cx - w * radius, cy),
            (cx,              cy),
            (cx + w * radius, cy),
            (cx - w * radius, cy + h * radius),
            (cx,              cy + h * radius),
            (cx + w * radius, cy + h * radius),
        ]

        for x2, y2, w2, h2 in candidates:
            for x3, y3 in neighbor_positions:
                if (x2 < x3 and y2 < y3) and (x2 + w2 > x3 and y2 + h2 > y3):
                    contour_neighbors[i].append((x2, y2, w2, h2))
                    break

    # Find the contour that has exactly 9 neighbors — this is the center
    final_contours = None
    for neighbors in contour_neighbors.values():
        if len(neighbors) == 9:
            final_contours = neighbors
            break

    if final_contours is None:
        return []

    # Sort by Y then X (top-to-bottom, left-to-right)
    y_sorted = sorted(final_contours, key=lambda item: item[1])
    top_row = sorted(y_sorted[0:3], key=lambda item: item[0])
    middle_row = sorted(y_sorted[3:6], key=lambda item: item[0])
    bottom_row = sorted(y_sorted[6:9], key=lambda item: item[0])
    return top_row + middle_row + bottom_row

def get_dominant_color_kmeans(roi):
    """
    Get dominant color from ROI using K-means clustering (1 cluster).
    Matches qbr's approach exactly.
    """
    pixels = np.float32(roi.reshape(-1, 3))
    n_colors = 1
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    _, counts = np.unique(labels, return_counts=True)
    dominant = palette[np.argmax(counts)]
    return tuple(dominant)

def extract_contour_colors_kmeans(frame, rects):
    """
    Extract dominant BGR colors from each contour's center region using K-means.
    Returns list of 9 BGR tuples.
    """
    colors = []
    for x, y, w, h in rects:
        # Take center region, shrinking by 7px top/bottom, 14px left/right (matching qbr)
        roi = frame[y+7:y+h-7, x+14:x+w-14]
        colors.append(get_dominant_color_kmeans(roi))
    return colors

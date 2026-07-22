import cv2
import numpy as np

def preprocess(frame):
    """Preprocess frame for contour detection: per-channel Canny -> combine -> close -> dilate."""
    edges = np.zeros(frame.shape[:2], dtype=np.uint8)
    for ch in cv2.split(frame):
        blurred = cv2.GaussianBlur(ch, (7, 7), 0)
        canny = cv2.Canny(blurred, 30, 60, 3)
        edges = cv2.bitwise_or(edges, canny)
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_k)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    dilated = cv2.dilate(closed, kernel)
    return dilated

def find_sticker_contours(dilated_frame):
    """
    Find and validate exactly 9 sticker contours.
    Returns sorted list of (x, y, w, h) or empty list.
    """
    fh, fw = dilated_frame.shape[:2]
    # Invert so stickers become white blobs, edges become black separators.
    # Text creates holes inside sticker blobs but the outer boundary stays clean.
    inverted = cv2.bitwise_not(dilated_frame)
    contours, _ = cv2.findContours(inverted, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    min_s = max(15, int(fw * 0.02))
    max_s = min(fw, int(fw * 0.15))

    candidates = []
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.1 * peri, True)
        if len(approx) != 4:
            continue

        x, y, w, h = cv2.boundingRect(approx)
        ratio = w / float(h)
        if not (0.7 <= ratio <= 1.3):
            continue
        if not (min_s <= w <= max_s):
            continue

        area = cv2.contourArea(c)
        solidity = area / float(w * h)
        if solidity < 0.3:
            continue

        candidates.append((x, y, w, h))

    if len(candidates) < 9:
        if len(candidates) < 4:
            return []
        # Infer 3x3 grid from bounding box of found candidates
        xs = [c[0] + c[2] / 2 for c in candidates]
        ys = [c[1] + c[3] / 2 for c in candidates]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        cell_w = (max_x - min_x) / 2
        cell_h = (max_y - min_y) / 2
        grid = []
        for row in range(3):
            for col in range(3):
                gx = min_x + col * cell_w
                gy = min_y + row * cell_h
                best_d = float('inf')
                best = None
                for x, y, w, h in candidates:
                    px, py = x + w / 2, y + h / 2
                    d = (px - gx) ** 2 + (py - gy) ** 2
                    if d < best_d:
                        best_d = d
                        best = (x, y, w, h)
                if best and best_d < (cell_w * 0.35) ** 2:
                    grid.append(best)
                else:
                    gw, gh = cell_w * 0.7, cell_h * 0.7
                    grid.append((int(gx - gw / 2), int(gy - gh / 2), int(gw), int(gh)))
        if len(grid) == 9:
            y_sorted = sorted(grid, key=lambda item: item[1])
            top_row = sorted(y_sorted[0:3], key=lambda item: item[0])
            middle_row = sorted(y_sorted[3:6], key=lambda item: item[0])
            bottom_row = sorted(y_sorted[6:9], key=lambda item: item[0])
            return top_row + middle_row + bottom_row
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
        y1 = y + 7
        y2 = y + h - 7
        x1 = x + 14
        x2 = x + w - 14
        if y2 - y1 < 5 or x2 - x1 < 5:
            y1, y2 = y, y + h
            x1, x2 = x + 7, x + w - 7
        if y2 - y1 < 5 or x2 - x1 < 5:
            y1, y2 = y, y + h
            x1, x2 = x, x + w
        roi = frame[y1:y2, x1:x2]
        if roi.size < 9:
            colors.append((0, 0, 0))
        else:
            colors.append(get_dominant_color_kmeans(roi))
    return colors

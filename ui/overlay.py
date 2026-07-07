import cv2
import numpy as np

def get_grid_regions(frame_width, frame_height, rect_size=40, spacing=60):
    """
    Returns a list of 9 (x, y) tuples representing the top-left corners of the 9 regions,
    plus the rect_size.
    The regions are ordered left-to-right, top-to-bottom.
    """
    cx, cy = frame_width // 2, frame_height // 2
    
    # Offsets for a 3x3 grid
    offsets = [-1, 0, 1]
    
    regions = []
    for dy in offsets:
        for dx in offsets:
            # Center of this specific rectangle
            rx = cx + dx * spacing
            ry = cy + dy * spacing
            
            # Top-left corner
            tl_x = rx - rect_size // 2
            tl_y = ry - rect_size // 2
            
            regions.append((tl_x, tl_y))
            
    return regions, rect_size

def draw_grid(frame, regions, rect_size, color=(255, 255, 255), thickness=2):
    """
    Draws the 3x3 grid on the frame.
    """
    for (x, y) in regions:
        cv2.rectangle(frame, (x, y), (x + rect_size, y + rect_size), color, thickness)
    return frame

def draw_text_overlay(frame, text, position=(30, 30), color=(255, 255, 255), font_scale=1.0, thickness=2):
    """
    Draws text with a dark background for better visibility.
    """
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2)
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    return frame

def draw_current_colors(frame, regions, rect_size, colors):
    """
    Fills the rectangles with the detected colors for feedback.
    colors is a list of 9 BGR tuples.
    """
    for i, (x, y) in enumerate(regions):
        if i < len(colors) and colors[i] is not None:
            # colors[i] should be BGR
            # Fill rectangle
            cv2.rectangle(frame, (x + 2, y + 2), (x + rect_size - 2, y + rect_size - 2), colors[i], -1)
    return frame

def draw_calibration_legend(frame, calibration_status, position=(30, 120), reference_bgr=None):
    """
    Draws the calibration color legend showing number keys and current reference colors.
    calibration_status: dict mapping color_name -> bool (calibrated this session)
    reference_bgr: dict mapping color_name -> BGR tuple (for swatch display)
    """
    y_start = position[1]
    color_names = ['white', 'red', 'orange', 'yellow', 'green', 'blue']
    display_labels = ['1:White', '2:Red', '3:Orange', '4:Yellow', '5:Green', '6:Blue']
    text_colors = {
        'white': (255, 255, 255),
        'red': (0, 0, 255),
        'orange': (0, 128, 255),
        'yellow': (0, 255, 255),
        'green': (0, 255, 0),
        'blue': (255, 0, 0),
    }

    for i, (cname, label) in enumerate(zip(color_names, display_labels)):
        y_pos = y_start + i * 25
        status = calibration_status.get(cname, False)
        prefix = "[SAVED]" if status else "       "
        swatch_bgr = reference_bgr.get(cname, (128, 128, 128)) if reference_bgr else (128, 128, 128)
        swatch_bgr = tuple(int(v) for v in swatch_bgr)
        cv2.rectangle(frame, (position[0], y_pos), (position[0] + 20, y_pos + 18), swatch_bgr, -1)
        cv2.rectangle(frame, (position[0], y_pos), (position[0] + 20, y_pos + 18), (200, 200, 200), 1)
        display_text = f"{label}  {prefix}"
        cv2.putText(frame, display_text, (position[0] + 25, y_pos + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_colors[cname], 1)

def draw_cube_net(frame, scanned_faces, color_bgr_map, offset_x, offset_y, block_size=10):
    """
    Draws a 2D unfolded cube net based on the current scanned faces.
    Layout:
         [ U ]
    [ L ][ F ][ R ][ B ]
         [ D ]
         
    Expected scanned_faces order (based on SCAN_ORDER): F, L, B, R, U, D
    """
    # Mapping the scan index to grid positions (row, col) in the net
    # Net dimensions: 3 rows, 4 cols
    # Row 0: empty, empty, U, empty -> wait, standard is U above F
    # So:
    # row 0: col 1 -> U (index 4)
    # row 1: col 0 -> L (index 1), col 1 -> F (index 0), col 2 -> R (index 3), col 3 -> B (index 2)
    # row 2: col 1 -> D (index 5)
    
    face_positions = {
        0: (1, 1), # F
        1: (1, 0), # L
        2: (1, 3), # B
        3: (1, 2), # R
        4: (0, 1), # U
        5: (2, 1)  # D
    }
    
    # Draw backgrounds/outlines for all faces first
    for scan_idx, (grid_r, grid_c) in face_positions.items():
        start_x = offset_x + grid_c * (block_size * 3 + 5)
        start_y = offset_y + grid_r * (block_size * 3 + 5)
        
        # Draw the 3x3 face
        for r in range(3):
            for c in range(3):
                pt1 = (start_x + c * block_size, start_y + r * block_size)
                pt2 = (pt1[0] + block_size, pt1[1] + block_size)
                
                # Default empty color
                bgr_color = (50, 50, 50)
                
                # If this face has been scanned, fill with scanned colors
                if scan_idx < len(scanned_faces):
                    color_name = scanned_faces[scan_idx][r * 3 + c]
                    bgr_color = color_bgr_map.get(color_name, (128, 128, 128))
                
                # Fill
                cv2.rectangle(frame, pt1, pt2, bgr_color, -1)
                # Border
                cv2.rectangle(frame, pt1, pt2, (200, 200, 200), 1)

    return frame

import cv2
import numpy as np
import sys
import argparse
import time
from ui.overlay import (
    get_grid_regions, draw_grid, draw_text_overlay, draw_current_colors,
    draw_cube_net, draw_calibration_legend, draw_detected_contours, draw_contour_colors
)
from vision.color_detector import (
    extract_colors, get_dominant_color, COLOR_BGR, REFERENCE_BGR, REFERENCE_LAB,
    CALIBRATION_COLORS, DEFAULT_REFERENCE_BGR, update_reference_color,
    save_calibration, load_calibration, sample_calibration_color, ciede2000_distance
)
from vision.state_extractor import extract_state_string
from vision.contour_detector import preprocess, find_sticker_contours, extract_contour_colors_kmeans
from vision.stream import MJPEGStream
from solver.cube_solver import solve_cube

CENTER_TO_FACE = {
    'green': 0, 'orange': 1, 'blue': 2,
    'red': 3, 'white': 4, 'yellow': 5
}
FACE_NAMES = ['Front (green)', 'Left (orange)', 'Back (blue)', 'Right (red)', 'Up (white)', 'Down (yellow)']
FACE_HINTS = [
    "Keep white on top",
    "Keep white on top, front face on left",
    "Keep white on top",
    "Keep white on top, front face on right",
    "Keep green toward camera, white up",
    "Keep green toward camera, yellow up"
]

CONTOUR_AVERAGE_ROUNDS = 8

def rotate_face_cw(face_colors):
    return [
        face_colors[6], face_colors[3], face_colors[0],
        face_colors[7], face_colors[4], face_colors[1],
        face_colors[8], face_colors[5], face_colors[2]
    ]

def rotate_face_ccw(face_colors):
    return [
        face_colors[2], face_colors[5], face_colors[8],
        face_colors[1], face_colors[4], face_colors[7],
        face_colors[0], face_colors[3], face_colors[6]
    ]

def is_solved_state(state_str):
    for i in range(6):
        face = state_str[i*9:(i+1)*9]
        if len(set(face)) != 1:
            return False
    return True

def map_display_to_physical_order(display_colors):
    """Map display-order colors (mirrored screen left-to-right) to physical left-to-right order."""
    return [
        display_colors[2], display_colors[1], display_colors[0],
        display_colors[5], display_colors[4], display_colors[3],
        display_colors[8], display_colors[7], display_colors[6]
    ]

def compute_display_bgrs(physical_colors):
    """Mirror physical colors back to display order for on-screen drawing."""
    return [
        COLOR_BGR.get(physical_colors[2], COLOR_BGR['unknown']),
        COLOR_BGR.get(physical_colors[1], COLOR_BGR['unknown']),
        COLOR_BGR.get(physical_colors[0], COLOR_BGR['unknown']),
        COLOR_BGR.get(physical_colors[5], COLOR_BGR['unknown']),
        COLOR_BGR.get(physical_colors[4], COLOR_BGR['unknown']),
        COLOR_BGR.get(physical_colors[3], COLOR_BGR['unknown']),
        COLOR_BGR.get(physical_colors[8], COLOR_BGR['unknown']),
        COLOR_BGR.get(physical_colors[7], COLOR_BGR['unknown']),
        COLOR_BGR.get(physical_colors[6], COLOR_BGR['unknown']),
    ]

def main():
    parser = argparse.ArgumentParser(description='Cube Vision Solver')
    parser.add_argument('--source', type=str, default='0',
                        help='Camera source: "0" for webcam, or URL for iPhone stream (e.g. http://192.168.1.X:8080/video)')
    parser.add_argument('--max-width', type=int, default=960,
                        help='Maximum display width in pixels (auto-scales down if larger)')
    args = parser.parse_args()

    print("Cube Vision Solver starting...")

    source = int(args.source) if args.source.isdigit() else args.source
    if isinstance(source, str) and source.startswith('http'):
        print(f"Connecting to iPhone stream: {source}")
        cap = MJPEGStream(source)
        for _ in range(20):
            if cap.isOpened():
                break
            time.sleep(0.25)
        if not cap.isOpened():
            if '@' not in source:
                print(f"Error: Could not connect. If the app requires login, use:")
                print(f"  python3 main.py --source \"http://user:pass@{source.split('://')[1]}\"")
            sys.exit(1)
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Error: Could not open camera source: {source}")
            sys.exit(1)

    load_calibration()

    state = "SCANNING"
    scanned_faces = [None] * 6
    last_captured_index = -1
    solution_moves = []
    cube_already_solved = False
    error_msg = ""
    scanning_feedback = ""

    calibration_status = {c: False for c in CALIBRATION_COLORS}
    calibration_feedback = ""
    calibration_feedback_timer = 0

    contour_average_colors = {}
    contour_preview_state = [None] * 9
    contour_rects = []
    contour_last_auto_face = -1
    contour_auto_cooldown = 0

    while True:
        ret, raw_frame = cap.read()
        if not ret:
            break

        h_raw, w_raw = raw_frame.shape[:2]
        scale = min(1.0, args.max_width / w_raw)
        if scale < 1.0:
            new_w = int(w_raw * scale)
            new_h = int(h_raw * scale)
            raw_frame = cv2.resize(raw_frame, (new_w, new_h))

        display_frame = cv2.flip(raw_frame, 1)
        h, w, _ = display_frame.shape
        display_regions, rect_size = get_grid_regions(w, h, rect_size=40, spacing=60)

        if state == "SCANNING":
            raw_grid_regions, _ = get_grid_regions(w, h, rect_size=40, spacing=60)
            raw_colors = extract_colors(raw_frame, raw_grid_regions, rect_size)
            current_colors = list(raw_colors)

            display_bgrs = compute_display_bgrs(current_colors)
            display_frame = draw_grid(display_frame, display_regions, rect_size)
            display_frame = draw_current_colors(display_frame, display_regions, rect_size, display_bgrs)

            center_color = current_colors[4]
            if center_color != 'unknown' and center_color in CENTER_TO_FACE:
                face_idx = CENTER_TO_FACE[center_color]
                face_name = FACE_NAMES[face_idx]
                hint = FACE_HINTS[face_idx]
                already = " [ALREADY CAPTURED]" if scanned_faces[face_idx] is not None else ""
                display_frame = draw_text_overlay(display_frame, f"Detected: {face_name}{already}",
                                                  position=(30, 30), font_scale=0.7)
                display_frame = draw_text_overlay(display_frame, hint,
                                                  position=(30, 60), font_scale=0.5, color=(200, 200, 200))
            else:
                display_frame = draw_text_overlay(display_frame, "Unknown center — press C to calibrate",
                                                  position=(30, 30), color=(0, 0, 255), font_scale=0.7)

            captured_count = sum(1 for f in scanned_faces if f is not None)
            display_frame = draw_text_overlay(display_frame, f"Captured: {captured_count}/6 — SPACE to capture, S to solve",
                                              position=(30, 90), font_scale=0.5, color=(200, 200, 200))
            if scanning_feedback:
                display_frame = draw_text_overlay(display_frame, scanning_feedback,
                                                  position=(30, 120), font_scale=0.5, color=(0, 255, 0))

            controls_text = "SPACE: Capture | A: Contour | U: Undo | R: Reset | C: Calibrate | Q: Quit"
            if captured_count > 0:
                controls_text += " | [ / ]: Rotate"
            display_frame = draw_text_overlay(display_frame, controls_text,
                                              position=(30, 145), font_scale=0.5, color=(200, 200, 200))

        elif state == "CONTOUR_SCANNING":
            dilated = preprocess(display_frame)
            rects = find_sticker_contours(dilated)

            if rects:
                contour_rects = rects
                display_frame = draw_detected_contours(display_frame, rects, color=(0, 255, 0), thickness=2)

                kmeans_bgrs = extract_contour_colors_kmeans(display_frame, rects)

                for idx, bgr in enumerate(kmeans_bgrs):
                    if idx in contour_average_colors:
                        contour_average_colors[idx].append(tuple(bgr))
                    else:
                        contour_average_colors[idx] = [tuple(bgr)]

                    samples = contour_average_colors[idx]
                    if len(samples) >= CONTOUR_AVERAGE_ROUNDS:
                        from collections import Counter
                        bgr_counts = Counter(samples)
                        most_common_bgr = bgr_counts.most_common(1)[0][0]
                        contour_average_colors[idx] = []
                        contour_preview_state[idx] = most_common_bgr

                stable_bgrs = [c for c in contour_preview_state if c is not None]
                if len(stable_bgrs) == 9:
                    disp_bgrs_original = list(contour_preview_state)
                    disp_bgrs_display = []
                    current_colors = []
                    for bgr_tuple in contour_preview_state:
                        bgr_np = np.uint8([[bgr_tuple]])
                        lab = cv2.cvtColor(bgr_np, cv2.COLOR_BGR2LAB)[0][0]
                        best = 'unknown'
                        best_dist = float('inf')
                        for name, ref_lab in REFERENCE_LAB.items():
                            dist = ciede2000_distance(lab, ref_lab)
                            if dist < best_dist:
                                best_dist = dist
                                best = name
                        current_colors.append(best)
                        disp_bgrs_display.append(COLOR_BGR.get(best, COLOR_BGR['unknown']))

                    display_frame = draw_contour_colors(display_frame, rects, disp_bgrs_display)
                    current_colors = map_display_to_physical_order(current_colors)

                    center_color = current_colors[4]
                    if center_color != 'unknown' and center_color in CENTER_TO_FACE:
                        face_idx = CENTER_TO_FACE[center_color]
                        face_name = FACE_NAMES[face_idx]
                        hint = FACE_HINTS[face_idx]
                        already = " [ALREADY CAPTURED]" if scanned_faces[face_idx] is not None else ""
                        display_frame = draw_text_overlay(display_frame, f"Detected: {face_name}{already}",
                                                          position=(30, 30), font_scale=0.7)
                        display_frame = draw_text_overlay(display_frame, hint,
                                                          position=(30, 60), font_scale=0.5, color=(200, 200, 200))

                        # Auto-capture if all colors known, face not already captured, and different from last auto-capture
                        if (contour_auto_cooldown <= 0
                                and scanned_faces[face_idx] is None
                                and all(c != 'unknown' for c in current_colors)
                                and face_idx != contour_last_auto_face):
                            scanned_faces[face_idx] = current_colors
                            last_captured_index = face_idx
                            contour_last_auto_face = face_idx
                            contour_auto_cooldown = 30
                            captured_count = sum(1 for f in scanned_faces if f is not None)
                            scanning_feedback = f"Auto-captured {face_name} ({captured_count}/6)"
                            print(scanning_feedback)
                            if captured_count == 6:
                                scanning_feedback += " — Press S to solve!"
                    else:
                        display_frame = draw_text_overlay(display_frame, "Unknown center color",
                                                          position=(30, 30), color=(0, 0, 255), font_scale=0.7)
                else:
                    current_colors = None
                    display_frame = draw_text_overlay(display_frame, f"Stabilizing colors: {len(stable_bgrs)}/9",
                                                      position=(30, 30), color=(0, 255, 255), font_scale=0.7)
                    display_frame = draw_text_overlay(display_frame, "Hold the cube steady...",
                                                      position=(30, 60), font_scale=0.5, color=(200, 200, 200))
            else:
                contour_rects = []
                current_colors = None
                display_frame = draw_text_overlay(display_frame, "Detecting stickers: 0/9 found",
                                                  position=(30, 30), color=(0, 0, 255), font_scale=0.7)
                display_frame = draw_text_overlay(display_frame, "Position the face so all 9 stickers are outlined",
                                                  position=(30, 60), font_scale=0.5, color=(200, 200, 200))

            contour_auto_cooldown = max(0, contour_auto_cooldown - 1)
            captured_count = sum(1 for f in scanned_faces if f is not None)
            display_frame = draw_text_overlay(display_frame, f"Captured: {captured_count}/6 — Auto-scan active, S to solve",
                                              position=(30, 90), font_scale=0.5, color=(200, 200, 200))
            if scanning_feedback:
                display_frame = draw_text_overlay(display_frame, scanning_feedback,
                                                  position=(30, 120), font_scale=0.5, color=(0, 255, 0))
            controls_text = "S: Solve | A: Grid | U: Undo | R: Reset | C: Calibrate | Q: Quit"
            if captured_count > 0:
                controls_text += " | [ / ]: Rotate"
            display_frame = draw_text_overlay(display_frame, controls_text,
                                              position=(30, 145), font_scale=0.5, color=(200, 200, 200))

        elif state == "ERROR":
            display_frame = draw_text_overlay(display_frame, "Scan Error!", position=(30, 40), color=(0, 0, 255))
            if error_msg:
                display_err = error_msg if len(error_msg) < 50 else error_msg[:47] + "..."
                display_frame = draw_text_overlay(display_frame, display_err, position=(30, 70), color=(0, 0, 255), font_scale=0.5)
            display_frame = draw_text_overlay(display_frame, "Press 'R' reset | 'U' undo | 'Q' quit.", position=(30, 100), font_scale=0.6)

        elif state == "SOLVING":
            if cube_already_solved:
                display_frame = draw_text_overlay(display_frame, "The cube is already solved!", position=(30, 40), color=(0, 255, 0), font_scale=0.7)
                display_frame = draw_text_overlay(display_frame, "Press 'R' restart | 'Q' quit.", position=(30, 80), font_scale=0.6)
            else:
                solution_str = " ".join(solution_moves)
                if len(solution_str) > 40:
                    mid = len(solution_moves) // 2
                    part1 = " ".join(solution_moves[:mid])
                    part2 = " ".join(solution_moves[mid:])
                    display_frame = draw_text_overlay(display_frame, f"Solution: {part1}", position=(30, 40), color=(0, 255, 0), font_scale=0.7)
                    display_frame = draw_text_overlay(display_frame, part2, position=(30, 70), color=(0, 255, 0), font_scale=0.7)
                    display_frame = draw_text_overlay(display_frame, "Press 'R' restart | 'Q' quit.", position=(30, 110), font_scale=0.6)
                else:
                    display_frame = draw_text_overlay(display_frame, f"Solution: {solution_str}", position=(30, 40), color=(0, 255, 0), font_scale=0.7)
                    display_frame = draw_text_overlay(display_frame, "Press 'R' restart | 'Q' quit.", position=(30, 80), font_scale=0.6)

        elif state == "CALIBRATING":
            raw_grid_regions, _ = get_grid_regions(w, h, rect_size=40, spacing=60)
            raw_colors = extract_colors(raw_frame, raw_grid_regions, rect_size)
            current_colors = list(raw_colors)

            display_bgrs = compute_display_bgrs(current_colors)
            display_frame = draw_grid(display_frame, display_regions, rect_size)
            display_frame = draw_current_colors(display_frame, display_regions, rect_size, display_bgrs)
            draw_calibration_legend(display_frame, calibration_status, position=(30, 130), reference_bgr=REFERENCE_BGR)

            if calibration_feedback and calibration_feedback_timer > 0:
                display_frame = draw_text_overlay(display_frame, calibration_feedback, position=(30, 30), color=(0, 255, 255), font_scale=0.7)
                calibration_feedback_timer -= 1
            else:
                display_frame = draw_text_overlay(display_frame, "CALIBRATION MODE", position=(30, 30), color=(0, 255, 255), font_scale=0.7)
            display_frame = draw_text_overlay(display_frame, "Show face filling grid, then press 1-6 to sample that color", position=(30, 60), font_scale=0.5, color=(200, 200, 200))
            display_frame = draw_text_overlay(display_frame, "C: Exit & Save | R: Reset defaults | Q: Quit", position=(30, 90), font_scale=0.5, color=(200, 200, 200))

        display_frame = draw_cube_net(display_frame, scanned_faces, COLOR_BGR, offset_x=w - 160, offset_y=20, block_size=10)
        cv2.imshow("Cube Vision Solver", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            if state in ("SCANNING", "CONTOUR_SCANNING"):
                if current_colors is None:
                    scanning_feedback = "Cannot capture — make sure all 9 stickers are detected"
                else:
                    center_color = current_colors[4]
                    if center_color == 'unknown':
                        scanning_feedback = "Cannot capture — center color is unknown"
                    elif center_color not in CENTER_TO_FACE:
                        scanning_feedback = f"Cannot capture — unexpected center: {center_color}"
                    elif scanned_faces[CENTER_TO_FACE[center_color]] is not None:
                        face_name = FACE_NAMES[CENTER_TO_FACE[center_color]]
                        scanning_feedback = f"{face_name} already captured — show a different face"
                    else:
                        face_idx = CENTER_TO_FACE[center_color]
                        scanned_faces[face_idx] = current_colors
                        last_captured_index = face_idx
                        face_name = FACE_NAMES[face_idx]
                        captured_count = sum(1 for f in scanned_faces if f is not None)
                        scanning_feedback = f"Captured {face_name} ({captured_count}/6)"
                        print(scanning_feedback)
                        if captured_count == 6:
                            scanning_feedback += " — Press S to solve!"
        elif key == ord('a'):
            if state == "SCANNING":
                state = "CONTOUR_SCANNING"
                contour_average_colors = {}
                contour_preview_state = [None] * 9
                contour_rects = []
                scanning_feedback = "Switched to Contour Scan mode"
                print(scanning_feedback)
            elif state == "CONTOUR_SCANNING":
                state = "SCANNING"
                scanning_feedback = "Switched to Grid Scan mode"
                print(scanning_feedback)
        elif key == ord('s'):
            if state in ("SCANNING", "CONTOUR_SCANNING"):
                captured_count = sum(1 for f in scanned_faces if f is not None)
                if captured_count < 6:
                    scanning_feedback = f"Not all faces scanned yet ({captured_count}/6)"
                    print(scanning_feedback)
                else:
                    print("All faces scanned. Extracting state...")
                    try:
                        state_str = extract_state_string(scanned_faces)
                        print(f"Cube state string: {state_str}")
                        try:
                            with open("cube_state.txt", "w") as f:
                                f.write(state_str)
                            print("Saved state string to cube_state.txt")
                        except Exception as e:
                            print(f"Could not save state string to file: {e}")
                        if is_solved_state(state_str):
                            print("Cube is already solved!")
                            solution_moves = []
                            cube_already_solved = True
                            state = "SOLVING"
                        else:
                            solution_moves = solve_cube(state_str)
                            cube_already_solved = False
                            print(f"Solution: {' '.join(solution_moves)}")
                            state = "SOLVING"
                    except Exception as e:
                        error_msg = str(e)
                        print(f"Error extracting state: {error_msg}")
                        state = "ERROR"
        elif key == ord('c'):
            if state == "CALIBRATING":
                save_calibration()
                print("Calibration saved.")
                calibration_feedback = ""
                state = "SCANNING"
            elif state in ("SCANNING", "CONTOUR_SCANNING", "ERROR"):
                calibration_feedback = ""
                calibration_feedback_timer = 0
                state = "CALIBRATING"
                print("Entered Calibration mode.")
        elif key == ord('r'):
            if state == "CALIBRATING":
                for cname in CALIBRATION_COLORS:
                    update_reference_color(cname, DEFAULT_REFERENCE_BGR[cname])
                    calibration_status[cname] = False
                calibration_feedback = "Calibration reset to defaults"
                calibration_feedback_timer = 60
                print("Calibration reset to defaults.")
            else:
                scanned_faces = [None] * 6
                last_captured_index = -1
                cube_already_solved = False
                scanning_feedback = ""
                if state in ("ERROR", "SOLVING"):
                    state = "SCANNING"
                print("Reset all scanned faces.")
        elif key == ord('u'):
            if state in ("SCANNING", "CONTOUR_SCANNING") and last_captured_index >= 0:
                face_name = FACE_NAMES[last_captured_index]
                scanned_faces[last_captured_index] = None
                captured_count = sum(1 for f in scanned_faces if f is not None)
                last_captured_index = -1
                for i in range(5, -1, -1):
                    if scanned_faces[i] is not None:
                        last_captured_index = i
                        break
                scanning_feedback = f"Undid {face_name} ({captured_count}/6)"
                print(scanning_feedback)
            elif state == "ERROR" and any(f is not None for f in scanned_faces):
                for i in range(5, -1, -1):
                    if scanned_faces[i] is not None:
                        scanned_faces[i] = None
                        last_captured_index = -1
                        state = "SCANNING"
                        print("Undid last scan and recovered from error.")
                        break
        elif key == ord('['):
            if state in ("SCANNING", "CONTOUR_SCANNING") and last_captured_index >= 0:
                scanned_faces[last_captured_index] = rotate_face_ccw(scanned_faces[last_captured_index])
                print(f"Rotated {FACE_NAMES[last_captured_index]} counter-clockwise.")
        elif key == ord(']'):
            if state in ("SCANNING", "CONTOUR_SCANNING") and last_captured_index >= 0:
                scanned_faces[last_captured_index] = rotate_face_cw(scanned_faces[last_captured_index])
                print(f"Rotated {FACE_NAMES[last_captured_index]} clockwise.")
        elif ord('1') <= key <= ord('6') and state == "CALIBRATING":
            idx = key - ord('1')
            color_name = CALIBRATION_COLORS[idx]
            raw_grid_regions, _ = get_grid_regions(w, h, rect_size=40, spacing=60)
            sampled_bgr = sample_calibration_color(raw_frame, raw_grid_regions, rect_size)
            update_reference_color(color_name, sampled_bgr)
            calibration_status[color_name] = True
            calibration_feedback = f"Sampled {color_name}: BGR({int(sampled_bgr[0])},{int(sampled_bgr[1])},{int(sampled_bgr[2])})"
            calibration_feedback_timer = 60
            print(calibration_feedback)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

import cv2
import sys
from ui.overlay import get_grid_regions, draw_grid, draw_text_overlay, draw_current_colors, draw_cube_net
from vision.color_detector import extract_colors, COLOR_BGR
from vision.state_extractor import extract_state_string, SCAN_ORDER
from solver.cube_solver import solve_cube
from solver.simulator import apply_move, faces_to_kociemba_list, kociemba_list_to_faces

# Specific instructions on how to hold the cube for each scan
SCAN_INSTRUCTIONS = [
    "Scan F (Front): Show the front face. Keep white on top if possible.",
    "Scan L (Left): From Front, rotate cube RIGHT.",
    "Scan B (Back): From Left, rotate cube RIGHT.",
    "Scan R (Right): From Back, rotate cube RIGHT.",
    "Scan U (Up): Go back to Front. Tilt cube DOWN.",
    "Scan D (Down): Go back to Front. Tilt cube UP."
]

# Auto-scan mode configuration
EXPECTED_CENTER_COLORS = ['green', 'orange', 'blue', 'red', 'white', 'yellow']
AUTO_SCAN_INSTRUCTIONS = [
    "Auto: Show FRONT face (center: GREEN)",
    "Auto: Show LEFT face (center: ORANGE)",
    "Auto: Show BACK face (center: BLUE)",
    "Auto: Show RIGHT face (center: RED)",
    "Auto: Show TOP face (center: WHITE)",
    "Auto: Show BOTTOM face (center: YELLOW)"
]
STABLE_THRESHOLD = 30

def rotate_face_cw(face_colors):
    """Rotates a 3x3 face array 90 degrees clockwise."""
    # 0 1 2      6 3 0
    # 3 4 5  ->  7 4 1
    # 6 7 8      8 5 2
    return [
        face_colors[6], face_colors[3], face_colors[0],
        face_colors[7], face_colors[4], face_colors[1],
        face_colors[8], face_colors[5], face_colors[2]
    ]

def rotate_face_ccw(face_colors):
    """Rotates a 3x3 face array 90 degrees counter-clockwise."""
    # 0 1 2      2 5 8
    # 3 4 5  ->  1 4 7
    # 6 7 8      0 3 6
    return [
        face_colors[2], face_colors[5], face_colors[8],
        face_colors[1], face_colors[4], face_colors[7],
        face_colors[0], face_colors[3], face_colors[6]
    ]

def is_solved_state(state_str):
    """Check if all 6 faces in a 54-char Kociemba string are uniform (solved)."""
    for i in range(6):
        face = state_str[i*9:(i+1)*9]
        if len(set(face)) != 1:
            return False
    return True

def main():
    print("Cube Vision Solver starting...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)
        
    state = "SCANNING"  # SCANNING, AUTO_SCANNING, SOLVING, ERROR
    scanned_faces = [] # Store colors for each face
    solution_moves = []
    cube_already_solved = False
    error_msg = ""

    # Auto-scan state
    auto_face_index = 0
    auto_stable_counter = 0
    auto_stable_center = None
    auto_feedback_msg = ""
    auto_feedback_color = (255, 255, 255)
    
    while True:
        ret, raw_frame = cap.read()
        if not ret:
            break
            
        # Display frame is mirrored for intuitive interaction
        display_frame = cv2.flip(raw_frame, 1)
        h, w, _ = display_frame.shape
        
        # We need to calculate grid regions for the display frame so we can draw UI
        display_regions, rect_size = get_grid_regions(w, h, rect_size=40, spacing=60)
        
        if state == "SCANNING":
            # Extract colors from the RAW (unflipped) frame.
            # To do this, we need to map the display regions back to the raw frame coordinates.
            # Since we flip horizontally, x_raw = width - x_display - rect_size
            raw_regions = []
            for (x, y) in display_regions:
                # The region logic uses centers. In the display frame, left-to-right is 0, 1, 2.
                # In the raw frame, this corresponds to right-to-left. 
                # But our extraction logic reads top-left to bottom-right.
                # Let's map the top-left coordinate (x,y) of the display frame back to the raw frame.
                raw_x = w - x - rect_size
                raw_regions.append((raw_x, y))
                
            # Note: Because the display regions are ordered left-to-right (0, 1, 2), 
            # mapping them individually to raw_x means raw_regions will be ordered right-to-left physically.
            # E.g. display_regions[0] (left) -> raw_regions[0] (right side of physical face).
            # This is exactly what we want! We want the array index 0 to correspond to the physical right 
            # side ONLY IF we are trying to mirror it. But wait, we want to extract EXACTLY what the user sees
            # on the screen, just using the unflipped image to avoid artifacts or coordinate confusion.
            # Actually, the user looks at the mirrored screen and aligns the cube. 
            # Top-left on the mirrored screen = Top-right physically. 
            # If the user expects standard reading order (Top-Left = Index 0), we *must* read the Top-Left of the physical face.
            # Physical Top-Left = Top-Right on the mirrored screen.
            # Let's just use `extract_colors` on `raw_frame` using the standard `get_grid_regions` without mirroring the regions themselves!
            # Since `raw_frame` is what the camera physically sees, `get_grid_regions` on `raw_frame` will extract left-to-right 
            # *physically*.
            
            # Let's verify: 
            # physical left -> appears on right of raw_frame -> appears on left of display_frame.
            # So `get_grid_regions` on `raw_frame` ordered left-to-right will extract physical right-to-left.
            # To extract physical left-to-right, we just need to mirror the result array if we use standard regions,
            # OR we can just extract directly from the display_frame.
            # The prompt asks: "Don't mirror the processing frame: Pass the raw, un-flipped frame to your contour-finding and color-detection algorithms."
            
            # Let's get regions for the raw frame directly.
            raw_grid_regions, _ = get_grid_regions(w, h, rect_size=40, spacing=60)
            
            raw_colors = extract_colors(raw_frame, raw_grid_regions, rect_size)
            
            # Use colors exactly as extracted
            current_colors = list(raw_colors)
            
            # The solver requires current_colors. However, the display_frame is mirrored horizontally.
            # To make the UI overlay visually match the mirrored screen, we must reverse the rows of the colors just for drawing.
            display_bgrs = [
                COLOR_BGR.get(current_colors[2], COLOR_BGR['unknown']), COLOR_BGR.get(current_colors[1], COLOR_BGR['unknown']), COLOR_BGR.get(current_colors[0], COLOR_BGR['unknown']),
                COLOR_BGR.get(current_colors[5], COLOR_BGR['unknown']), COLOR_BGR.get(current_colors[4], COLOR_BGR['unknown']), COLOR_BGR.get(current_colors[3], COLOR_BGR['unknown']),
                COLOR_BGR.get(current_colors[8], COLOR_BGR['unknown']), COLOR_BGR.get(current_colors[7], COLOR_BGR['unknown']), COLOR_BGR.get(current_colors[6], COLOR_BGR['unknown'])
            ]
            
            # Draw UI on the mirrored display_frame
            display_frame = draw_grid(display_frame, display_regions, rect_size)
            
            display_frame = draw_current_colors(display_frame, display_regions, rect_size, display_bgrs)
            
            faces_scanned = len(scanned_faces)
            instruction = SCAN_INSTRUCTIONS[faces_scanned] if faces_scanned < 6 else ""
            display_frame = draw_text_overlay(display_frame, instruction, position=(30, 30), font_scale=0.7)
            
            # Additional controls info
            controls_text = "SPACE: Scan | U: Undo | R: Reset | Q: Quit"
            if faces_scanned > 0:
                controls_text += " | [ / ]: Rotate"
            display_frame = draw_text_overlay(display_frame, controls_text, position=(30, 60), font_scale=0.5, color=(200, 200, 200))
            
        elif state == "ERROR":
            display_frame = draw_text_overlay(display_frame, "Scan Error!", position=(30, 40), color=(0, 0, 255))
            if 'error_msg' in locals() and error_msg:
                # Truncate if too long to fit
                display_err = error_msg if len(error_msg) < 50 else error_msg[:47] + "..."
                display_frame = draw_text_overlay(display_frame, display_err, position=(30, 70), color=(0, 0, 255), font_scale=0.5)
            display_frame = draw_text_overlay(display_frame, "Press 'R' reset | 'U' undo | 'Q' quit.", position=(30, 100), font_scale=0.6)
        elif state == "SOLVING":
            if cube_already_solved:
                display_frame = draw_text_overlay(display_frame, "The cube is already solved!", position=(30, 40), color=(0, 255, 0), font_scale=0.7)
                display_frame = draw_text_overlay(display_frame, "Press 'R' restart | 'Q' quit.", position=(30, 80), font_scale=0.6)
            else:
                solution_str = " ".join(solution_moves)
                # Break into two lines if it's long
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
        elif state == "AUTO_SCANNING":
            raw_grid_regions, _ = get_grid_regions(w, h, rect_size=40, spacing=60)
            raw_colors = extract_colors(raw_frame, raw_grid_regions, rect_size)
            current_colors = list(raw_colors)

            center_color = current_colors[4]

            # Stability check
            if center_color == auto_stable_center:
                auto_stable_counter += 1
            else:
                auto_stable_center = center_color
                auto_stable_counter = 0

            if auto_stable_counter >= STABLE_THRESHOLD and auto_face_index < 6 and center_color != 'unknown':
                expected = EXPECTED_CENTER_COLORS[auto_face_index]
                if center_color == expected:
                    scanned_faces.append(current_colors)
                    print(f"Auto-captured {SCAN_ORDER[auto_face_index]} (center: {center_color})")
                    auto_face_index += 1
                    auto_feedback_msg = f"Captured {SCAN_ORDER[auto_face_index - 1]}!"
                    auto_feedback_color = (0, 255, 0)
                    auto_stable_counter = 0
                    auto_stable_center = None

                    if len(scanned_faces) == 6:
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
                else:
                    auto_feedback_msg = f"Wrong face! Center is {center_color}, expected {expected}"
                    auto_feedback_color = (0, 0, 255)
                    auto_stable_counter = 0
                    auto_stable_center = None

            # Mirror colors for display overlay
            display_bgrs = [
                COLOR_BGR.get(current_colors[2], COLOR_BGR['unknown']), COLOR_BGR.get(current_colors[1], COLOR_BGR['unknown']), COLOR_BGR.get(current_colors[0], COLOR_BGR['unknown']),
                COLOR_BGR.get(current_colors[5], COLOR_BGR['unknown']), COLOR_BGR.get(current_colors[4], COLOR_BGR['unknown']), COLOR_BGR.get(current_colors[3], COLOR_BGR['unknown']),
                COLOR_BGR.get(current_colors[8], COLOR_BGR['unknown']), COLOR_BGR.get(current_colors[7], COLOR_BGR['unknown']), COLOR_BGR.get(current_colors[6], COLOR_BGR['unknown'])
            ]

            display_frame = draw_grid(display_frame, display_regions, rect_size)
            display_frame = draw_current_colors(display_frame, display_regions, rect_size, display_bgrs)

            if auto_face_index < 6:
                instruction = AUTO_SCAN_INSTRUCTIONS[auto_face_index]
            else:
                instruction = "All faces captured! Solving..."
            display_frame = draw_text_overlay(display_frame, instruction, position=(30, 30), font_scale=0.7)

            if auto_feedback_msg:
                display_frame = draw_text_overlay(display_frame, auto_feedback_msg, position=(30, 90), font_scale=0.6, color=auto_feedback_color)

            controls_text = "M: Back to Manual | R: Reset | Q: Quit"
            display_frame = draw_text_overlay(display_frame, controls_text, position=(30, 60), font_scale=0.5, color=(200, 200, 200))
        
        # Draw the 2D Cube Net in the top-right corner
        display_frame = draw_cube_net(display_frame, scanned_faces, COLOR_BGR, offset_x=w - 160, offset_y=20, block_size=10)
        
        cv2.imshow("Cube Vision Solver", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            if state == "SCANNING" and len(scanned_faces) < 6:
                scanned_faces.append(current_colors)
                print(f"Captured face {len(scanned_faces)}")
                if len(scanned_faces) == 6:
                    print("All faces scanned. Extracting state...")
                    try:
                        state_str = extract_state_string(scanned_faces)
                        print(f"Cube state string: {state_str}")
                        
                        # Save the state string to a file
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
                        print("Please reset ('R') and scan again.")
                        # Could reset automatically or just stay in SCANNING state (wait for reset)
                        # We'll pop the last face so the user can re-scan it, or just let them press 'R'
                        state = "ERROR"
        elif key == ord('m'):
            if state == "AUTO_SCANNING":
                state = "SCANNING"
                print("Switched to Manual Scan mode")
            elif state == "SCANNING":
                state = "AUTO_SCANNING"
                auto_face_index = len(scanned_faces)
                auto_stable_counter = 0
                auto_stable_center = None
                auto_feedback_msg = ""
                print("Switched to Auto-Scan mode")
        elif key == ord('r'):
            scanned_faces = []
            cube_already_solved = False
            state = "SCANNING"
            auto_face_index = 0
            auto_stable_counter = 0
            auto_stable_center = None
            auto_feedback_msg = ""
            print("Resetting scan state.")
        elif key == ord('u'):
            if state in ("SCANNING", "AUTO_SCANNING") and len(scanned_faces) > 0:
                scanned_faces.pop()
                if state == "AUTO_SCANNING":
                    auto_face_index = len(scanned_faces)
                    auto_stable_counter = 0
                    auto_stable_center = None
                    auto_feedback_msg = ""
                print(f"Undid last scan. Now at face {len(scanned_faces)}/6")
            elif state == "ERROR" and len(scanned_faces) > 0:
                scanned_faces.pop()
                state = "SCANNING"
                print("Undid last scan and recovered from error.")
        elif key == ord('['):
            # Rotate last scanned face counter-clockwise
            if state in ("SCANNING", "AUTO_SCANNING") and len(scanned_faces) > 0:
                scanned_faces[-1] = rotate_face_ccw(scanned_faces[-1])
                print("Rotated last face counter-clockwise.")
                
        elif key == ord(']'):
            # Rotate last scanned face clockwise
            if state in ("SCANNING", "AUTO_SCANNING") and len(scanned_faces) > 0:
                scanned_faces[-1] = rotate_face_cw(scanned_faces[-1])
                print("Rotated last face clockwise.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

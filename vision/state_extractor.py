SCAN_ORDER = ['F (Front)', 'L (Left)', 'B (Back)', 'R (Right)', 'U (Up)', 'D (Down)']

def extract_state_string(scanned_faces):
    if any(f is None for f in scanned_faces):
        raise ValueError("Not all 6 faces have been scanned yet")

    scan_face_letters = ['F', 'L', 'B', 'R', 'U', 'D']

    color_to_face = {}
    face_colors_map = {}
    for i, face_colors in enumerate(scanned_faces):
        center_color = face_colors[4]
        face_letter = scan_face_letters[i]

        if center_color in color_to_face:
            raise ValueError(f"Duplicate center color detected: {center_color}")

        color_to_face[center_color] = face_letter
        face_colors_map[face_letter] = face_colors

    kociemba_order = ['U', 'R', 'F', 'D', 'L', 'B']
    state_str = ""
    for face_letter in kociemba_order:
        for color in face_colors_map[face_letter]:
            if color not in color_to_face:
                 raise ValueError(f"Unknown color {color} not found in centers")
            state_str += color_to_face[color]

    return state_str

# Map standard cube face names
# Typical Kociemba string order: U1..U9, R1..R9, F1..F9, D1..D9, L1..L9, B1..B9
# 54 characters.
# The centers are U5, R5, F5, D5, L5, B5.
# We map scanned colors to these faces based on the center color of each scanned face.

# Define the scanning order for the user.
# The user starts facing the Front face.
# L: from F, rotate cube right
# B: from L, rotate cube right
# R: from B, rotate cube right (now facing Right face, original Front is left)
# Then return to F.
# U: from F, tilt cube down
# D: from F, tilt cube up
# Actually, the instructions will be managed in main.py, but we keep the logical order here.
SCAN_ORDER = ['F (Front)', 'L (Left)', 'B (Back)', 'R (Right)', 'U (Up)', 'D (Down)']

def extract_state_string(scanned_faces):
    """
    scanned_faces is a list of 6 lists, each containing 9 color names.
    We assume the scanning order was Front, Left, Back, Right, Up, Down.
    
    1. Identify the center color of each face (index 4).
    2. Create a mapping from color name to face letter (F, L, B, R, U, D).
    3. Construct the 54-char string in Kociemba order (U, R, F, D, L, B).
    """
    if any(f is None for f in scanned_faces):
        raise ValueError("Not all 6 faces have been scanned yet")
        
    # Letters corresponding to the SCAN_ORDER
    scan_face_letters = ['F', 'L', 'B', 'R', 'U', 'D']
    
    # Map color to face letter based on the center of each scanned face
    color_to_face = {}
    face_colors_map = {}
    for i, face_colors in enumerate(scanned_faces):
        center_color = face_colors[4]
        face_letter = scan_face_letters[i]
        
        # In a valid scan, centers must be unique
        if center_color in color_to_face:
            raise ValueError(f"Duplicate center color detected: {center_color}")
            
        color_to_face[center_color] = face_letter
        face_colors_map[face_letter] = face_colors
        
    # Construct the string in the order expected by Kociemba
    kociemba_order = ['U', 'R', 'F', 'D', 'L', 'B']
    state_str = ""
    for face_letter in kociemba_order:
        for color in face_colors_map[face_letter]:
            if color not in color_to_face:
                 raise ValueError(f"Unknown color {color} not found in centers")
            state_str += color_to_face[color]
            
    return state_str

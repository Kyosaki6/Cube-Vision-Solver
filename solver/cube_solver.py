import kociemba

def solve_cube(state_string):
    """
    Given a 54-character state string (composed of U, R, F, D, L, B),
    returns a list of moves to solve the cube using Kociemba's algorithm.
    
    The moves are returned as a list of strings, e.g. ["R", "U'", "F2", ...].
    """
    if len(state_string) != 54:
        raise ValueError(f"State string must be exactly 54 characters, got {len(state_string)}")
        
    try:
        # kociemba.solve returns a string like "R U' F2 D B R' ..."
        solution_str = kociemba.solve(state_string)
        moves = solution_str.split(" ")
        return moves
    except Exception as e:
        raise ValueError(f"Kociemba solver failed: {e}")

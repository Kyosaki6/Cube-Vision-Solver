import kociemba

# cubestring = input("Enter the cube string: ")
with open("cube_state.txt", "r", encoding = "utf-8") as file:
    cubestring = file.read()
try:
    print(kociemba.solve(cubestring))
except Exception as e:
    print(e)

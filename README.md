# Cube Vision Solver

**Real-time Rubik's Cube solving assistant using Computer Vision**

![Demo placeholder](https://via.placeholder.com/800x400?text=Cube+Vision+Solver+Demo)

## Overview

Cube Vision Solver is a desktop application that turns your webcam into an interactive Rubik's Cube guide.  
Scan the scrambled cube, get an optimal solution, and follow on‑screen arrow overlays – step by step – until the cube is solved.

No buttons, no manual notation entry. Just show the cube to the camera and turn as directed.

## How It Works

1. **Position the cube** inside the fixed 3×3 grid on your screen.
2. The system scans all six faces and recognises the current colour state.
3. An optimal solution (typically ≤20 moves) is calculated automatically.
4. **Dynamic arrows** appear around the grid, showing exactly which face to turn and in which direction.
5. Turn the cube – the application detects the movement and waits for you to finish.
6. Once stable, the next arrow appears. Repeat until solved.

> The auto‑advance mode uses **frame differencing**: no keystrokes, no buttons – just turn the cube and the system follows.

## Features

- 🧩 **Static alignment guidance** – place the cube inside a fixed on‑screen frame; no complex 3D tracking required.
- 🎨 **Robust colour detection** – HSV colour space + Gaussian blur for consistent results under varying indoor lighting.
- ⚡ **Fast optimal solver** – integrates Kociemba's Two‑Phase Algorithm (via `pykociemba`). Solves any cube in milliseconds.
- 🖱️ **Auto‑advance mechanism** – detects when the cube is being turned and when it becomes still, automatically moving to the next instruction.
- 📐 **Visual arrow overlays** – uses OpenCV drawing primitives (`cv2.arrowedLine`, `cv2.rectangle`) to render intuitive turn instructions.

## Tech Stack

| Component       | Technology / Library               |
|----------------|------------------------------------|
| Language       | Python 3                           |
| Computer Vision| OpenCV (cv2)                       |
| Colour handling| HSV conversion, Gaussian blur      |
| Solving engine | `pykociemba` (Kociemba's algorithm)|
| UI overlay     | OpenCV + NumPy                     |

## Project Structure (MVP Modules)

```
cube_vision_solver/
├── vision/          # Module 1: Colour detection & state extraction
├── solver/          # Module 2: Kociemba integration (move sequence)
├── ui/              # Module 3: Arrow rendering & frame differencing
├── main.py          # Application entry point
└── requirements.txt
```

## Development Milestones

| Milestone | Goal |
|-----------|------|
| **M1 – Static recognition** | Live grid overlay, scan 6 faces, store colour arrays (HSV sampling). |
| **M2 – Solver integration** | Feed colour data to Kociemba; verify that manual execution of the output string solves the cube. |
| **M3 – Auto‑advance filter** | Implement `absdiff` frame differencing to detect “still” state without user input. |
| **M4 – Arrow graphics & packaging** | Map all Singmaster notations (`R`, `U'`, `F2`, …) to 2D arrows; finalise standalone application. |

## Getting Started

### Prerequisites
- Python 3.8+
- Webcam

### Installation
```bash
git clone https://github.com/yourusername/cube-vision-solver.git
cd cube-vision-solver
pip install -r requirements.txt
```

### Usage
```bash
python main.py
```
Follow the on‑screen instructions.  
Hold the Rubik’s cube so that each face fits the 3×3 grid when prompted.  
Once all faces are scanned, the solving mode begins – simply follow the arrows.

## Limitations (Current MVP)

- Requires **static alignment** (the user moves the cube into the fixed grid, not the other way around).
- Works best with **good uniform lighting** (standard room light is fine; avoid strong backlight or shadows).
- Does **not** support arbitrary cube rotations during scanning – each face must be presented deliberately.

## Future Plans

- Full 3D cube tracking (no fixed grid)
- Support for more cube sizes (2×2, 4×4, etc.)
- Voice guidance as an alternative to arrows

## License

This project is for educational and research purposes.  
Kociemba's algorithm and `pykociemba` retain their respective licenses.

---

**Made with computer vision, combinatorial optimisation, and a lot of Rubik’s turns.**

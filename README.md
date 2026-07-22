# Cube Vision Solver

Real-time Rubik's Cube scanner and solver using computer vision.

## Features

- **Two scan modes** — fixed 3×3 grid (center) or automatic contour detection (finds stickers anywhere in frame)
- **CIEDE2000 color matching** — perceptual color distance in LAB space, far more accurate than HSV
- **On-the-fly calibration** — press `C`, hold a face, press `1`–`6` to sample reference colors for your specific cube and lighting
- **Kociemba optimal solver** — computes a solution (≤20 moves) from the scanned state
- **Real-time preview** — detected colors fill the grid/contours live on camera feed
- **phone / wireless camera** — use `--source` with an IP camera URL instead of the built-in webcam
- **Auto-scale** — high-resolution streams are automatically scaled to fit your screen

## Tech Stack

| Component | Library |
|-----------|---------|
| Language | Python 3 |
| Vision | OpenCV (`cv2`) |
| Color matching | CIEDE2000 (`colormath`) + LAB space |
| Solver | `kociemba` (two-phase algorithm) |
| Clustering | OpenCV `kmeans` for contour color extraction |

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python3 main.py
```

With an phone / IP camera:
```bash
python3 main.py --source "http://admin:pass@192.168.1.X:8081/video"
```

### Controls

| Key | Action |
|-----|--------|
| `A` | Toggle Grid Scan ↔ Contour Scan |
| `Space` | Capture current face |
| `S` | Solve (after all 6 faces captured) |
| `C` | Enter / Exit calibration mode |
| `R` | Reset scanned faces / reset calibration |
| `U` | Undo last capture |
| `[` / `]` | Rotate last captured face CCW / CW |
| `Q` | Quit |

### Calibration

For accurate color detection under your specific lighting and camera:

1. Press `C` to enter calibration mode
2. Hold a face so it fills the 3×3 grid
3. Press `1` (white), `2` (red), `3` (orange), `4` (yellow), `5` (green), `6` (blue)
4. Repeat for all 6 colors
5. Press `C` again to save (`calibration.json`)

### Scan Modes

**Grid Scan** — a fixed 3×3 grid at screen center. Align the cube face to the grid. Most reliable for consistent lighting.

**Contour Scan** — detects sticker boundaries automatically via per-channel Canny edge detection. Works at any position in the frame. Red/blue/white faces handled by per-channel edge detection and edge-map inversion (handles black text on white stickers).

## Project Structure

```
Cube-Vision-Solver/
├── main.py                     # Entry point
├── calibration.json            # Saved reference colors
├── vision/
│   ├── color_detector.py       # CIEDE2000 matching, calibration save/load
│   ├── contour_detector.py     # Contour finding, K-means color extraction
│   ├── state_extractor.py      # 6 faces → 54-char Kociemba string
│   ├── motion_detector.py      # Frame-differencing (infrastructure)
│   └── stream.py               # MJPEG stream reader (phone cameras)
├── ui/
│   └── overlay.py              # Grid, contours, cube net, calibration legend
├── solver/
│   ├── cube_solver.py          # Kociemba wrapper
│   └── simulator.py           # Move permutation tables
└── requirements.txt
```

## Requirements

- Python 3.8+
- Webcam or IP camera (phone with an IP camera app)
- `opencv-python`, `numpy`, `kociemba`, `colormath`

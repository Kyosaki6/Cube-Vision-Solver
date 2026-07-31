# 🧩 Cube Vision Solver

> Real-time Rubik's Cube scanner and solver using computer vision.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg?logo=opencv&logoColor=white)](https://opencv.org/)
[![Solver](https://img.shields.io/badge/Solver-Kociemba%20Two--Phase-orange.svg)](#features)
[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)

Cube Vision Solver scans a Rubik's Cube face-by-face from a webcam or IP camera, detects sticker colors reliably under varied lighting using perceptual color matching (CIEDE2000 in LAB space), and computes an optimal Kociemba solution (typically ≤ 20 moves).

---

## ✨ Key Features

- Two scan modes: fixed grid (3×3 center) and automatic contour detection (finds stickers anywhere in the frame).
- Perceptual color matching using CIEDE2000 in LAB space for robust color separation (better than HSV), with temporal averaging and median filtering to reduce noise and glare.
- On-the-fly calibration: press `C` and sample reference colors (`1`–`6`) to save a `calibration.json` tuned to your cube and lighting.
- Kociemba two-phase solver for fast optimal solutions.
- Real-time overlay: detected colors fill the grid or contours live on the camera feed.
- Support for phone / wireless IP cameras via MJPEG stream (`--source`).
- Auto-scaling for high-resolution streams.

---

## Tech Stack

| Component | Library / Tool |
|---|---|
| Language | Python 3.8+ |
| Vision | OpenCV (`cv2`) |
| Numeric | NumPy |
| Color Matching | CIEDE2000 (`colormath`), LAB color space |
| Solver | `kociemba` (two-phase algorithm) |
| Clustering | OpenCV `kmeans` (contour color extraction) |

---

## Installation

Requirements: Python 3.8 or newer, a webcam or an IP camera that streams MJPEG.

Clone and install:

```bash
git clone https://github.com/Kyosaki6/Cube-Vision-Solver.git
cd Cube-Vision-Solver
python -m venv venv
source venv/bin/activate    # Linux / macOS
# venv\Scripts\activate   # Windows (PowerShell)

pip install -r requirements.txt
```

---

## Usage

Run the app with the default webcam:

```bash
python main.py
```

Run using an IP camera / phone stream (MJPEG):

```bash
python main.py --source "http://admin:pass@192.168.1.X:8081/video"
```

---

## Controls

| Key | Action |
|---:|:---|
| A | Toggle Grid Scan ↔ Contour Scan |
| Space | Capture current face |
| S | Solve (after all 6 faces captured) |
| C | Enter / Exit calibration mode |
| R | Reset captured faces / reset calibration |
| U | Undo last capture |
| [ / ] | Rotate last captured face CCW / CW |
| Q | Quit |

---

## Calibration (recommended)

For best results under your lighting and camera:
1. Press `C` to enter calibration mode.
2. Fill the 3×3 grid with a single cube face.
3. Press the number keys to sample and save each face color:
   - `1` = White
   - `2` = Red
   - `3` = Orange
   - `4` = Yellow
   - `5` = Green
   - `6` = Blue
4. Repeat for all six colors then press `C` to save `calibration.json`.

Calibration greatly improves accuracy in environments with warm/cool lighting or directional glare.

---

## Scan Modes

Grid Scan — a fixed 3×3 grid shown at the center of the screen. Align the cube face to the grid and press Space to capture. This mode is most consistent when lighting is uniform and the cube face is roughly frontal to the camera.

Contour Scan — automatic sticker detection using a per-channel Canny edge detector followed by contour filtering and morphological operations (closing/dilation). For each plausible sticker contour we extract the dominant color using k-means clustering (k=1) and map detected patches to the cube net using geometric heuristics. This mode is more flexible: it can find faces placed anywhere in the frame and handle non-frontal orientations, but it may require stronger lighting or a clearer background for best results.

---

## Project Structure

```
Cube-Vision-Solver/
├── main.py                     # Entry point
├── calibration.json            # Saved reference colors (created after calibration)
├── vision/
│   ├── color_detector.py       # LAB conversion, CIEDE2000 matching, calibration I/O
│   ├── contour_detector.py     # Contour finding, morphological filtering, K-means color extraction
│   ├── state_extractor.py      # Convert 6 captured faces -> 54-char Kociemba string
│   └── stream.py               # MJPEG stream reader (phone/IP cameras)
├── ui/
│   └── overlay.py              # Grid, contours, cube net overlay and HUD
├── solver/
│   └── cube_solver.py          # Wrapper around kociemba solver
└── requirements.txt
```

---

## Requirements

- Python 3.8+
- Webcam or IP camera (phone with an IP camera app)
- Python packages: `opencv-python`, `numpy`, `kociemba`, `colormath` (see requirements.txt)

---

## Authors & Affiliation

- Nguyễn Trường Phước
- Huỳnh Quang Thịnh
- Nguyễn Đức Hoàng Nam

Supervisor: Dr. Nguyễn Văn Tới

Affiliation: Phenikaa School of Information and Communication Technology — Phenikaa University

---

## License

This repository assumes an MIT license. If you prefer a different license, add or replace the `LICENSE` file accordingly.

# OpenVC — Weekly OpenCV Exercises

This repository contains three weeks of computer vision exercises written in Python.  
Each week builds on the last, progressing from basic image I/O, to channel operations, and finally to landmark-based face swapping.  
The repository is structured so that each week can be run independently with its own `requirements.txt` file.

---

## Environment Setup

All weeks are designed to run inside a Python virtual environment.  
From the project root (`openvc/`):

```bash
# Create and activate a virtual environment
python3 -m venv env
source env/bin/activate
```

Then, before running code for a given week, install that week's requirements:

```bash
cd week1   # or week2 / week3
pip install -r requirements.txt
```

---

## Week 1 — Basic Image I/O

**Script:** `week1/week1MS.py`  
**Input file:** `shutterstock130285649--250.jpg`  

### Setup
```bash
cd week1
pip install -r requirements.txt
```

### Behavior
- Loads a local JPEG image.
- Displays the image in an OpenCV window (if a GUI is available).
- Writes a copy of the image to the Desktop as `numbers_copy.jpg`.
- Includes error handling for missing or corrupted images.

### Expected result
- OpenCV window titled "Numbers Image (Press any key to close)" will appear.
- Closing the window saves the file to `~/Desktop/numbers_copy.jpg`.
- Console logs confirm loading, display, and save operations.

---

## Week 2 — Color Channels & Manipulation

**Script:** `week2/week2CT.py`  
**Input file:** `kitten_photo.jpg`  

### Setup
```bash
cd week2
pip install -r requirements.txt
```

### Behavior
- Loads the input kitten image (supports bundled assets if compiled with PyInstaller).
- Extracts B, G, R channels individually and prints their shapes.
- Reconstructs the image using three methods (`cv2.merge`, NumPy stack, manual assignment) and compares results.
- Creates a GRB version by swapping red and green channels.
- Generates visualizations of individual channels (blue-only, green-only, red-only).
- Saves all outputs (`original_image.png`, `merged_image.png`, `grb_swapped_image.png`, and individual channel images).

### Expected result
- Console shows shapes of channels and True/False checks for reconstruction accuracy.
- Several `.png` output images appear in the working directory.
- If running in a headless VM, images will be saved but not displayed.

---

## Week 3 — Face Swapping with Dlib Landmarks

**Script:** `week3/face_swapper.py`  
**Input files:**
- `destination.jpg` (target face)
- `my_picture.jpg` (source face)
- `shape_predictor_68_face_landmarks.dat` (68-point landmark model, not tracked in Git)

### Setup

⚠️ **Important**: Download the dlib predictor file and place it into `week3/`.

Then install dependencies:
```bash
cd week3
pip install -r requirements.txt
```

### Behavior
- Detects faces in both source and destination images.
- Computes 68-point facial landmarks for both images.
- Generates a convex hull of the destination face and applies Delaunay triangulation.
- Warps triangles from the source face to match the destination geometry.
- Blends the warped face into the destination image using OpenCV's `seamlessClone`.
- Saves the final composite as `deepfake_output.jpg`.
- Attempts to display result in a window; on headless setups, only the file is saved.

### Expected result
- Console output logs steps: detection, triangulation, warping, blending.
- Output file `deepfake_output.jpg` shows the source face mapped onto the destination.
- If no faces are detected, the script exits without creating an output.

---

## System Requirements

- **OS**: Linux (tested on Ubuntu)
- **Python**: 3.10–3.12
- **GUI**: OpenCV windows with headless-safe fallbacks
- **Dependencies**: Listed in each week's `requirements.txt`

---

## Summary

- **Week 1**: Learn fundamental I/O — reading, displaying, and saving images with OpenCV.
- **Week 2**: Understand color channels, image decomposition, and recomposition.
- **Week 3**: Apply facial landmarks and geometric transforms to perform a realistic face swap.

This sequence moves from basics to advanced image processing, providing practical demonstrations of how OpenCV and dlib can be applied to real tasks.

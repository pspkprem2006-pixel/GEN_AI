# Project 3 - OpenCV Image Processing

The vision fundamentals every system uses before deep learning (Module 5,
section 14, built on the operations of section 8.4).

Generates a 600x400 sample image containing 5 filled shapes (square,
circle, rectangle, diamond, dark circle), then walks the whole pipeline:
grayscale -> Gaussian blur -> Canny edges -> threshold -> findContours,
and finally draws a red box around every detected object.

## Files
- `opencv_processing.py` - the full program
- `processing_steps.png` - generated when run: a 2x3 montage showing all
  six steps (original / grayscale / blur / edges / threshold / contours)
- `README.md` - this file

## How to run
```bash
pip install opencv-python matplotlib numpy   # if not installed
python opencv_processing.py
```

Runs headless (matplotlib 'Agg') - no window pops up; open the saved PNG
afterwards. Prints the generated image shape, the contour count, and a
confirmation line.

## Requirements
- Python 3.x, opencv-python, matplotlib, numpy (all installed)

## Expected output (sample run)
```
Sample image created: 600x400 pixels, 3 color channels (array shape (400, 600, 3)).

Contour detection found 5 object(s) in the image.
[OK] Montage saved to 'processing_steps.png'.
```

Identical in shape to the notes' sample output: 5 shapes in, 5 objects
detected, montage saved.

## The pipeline (notes section 8.4)
| Step | Function | Purpose |
|---|---|---|
| Grayscale | `cv2.cvtColor(..., BGR2GRAY)` | drop color |
| Blur | `cv2.GaussianBlur(gray, (9, 9), 0)` | smooth / reduce noise |
| Edges | `cv2.Canny(blurred, 50, 150)` | find outlines |
| Threshold | `cv2.threshold(gray, 240, 255, THRESH_BINARY_INV)` | pure black/white |
| Contours | `cv2.findContours(thresh, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)` | detect objects |
| Draw | `cv2.rectangle` + `cv2.putText` | box each object |

## Challenges
1. Add more shapes (triangles, ellipses) - the contour count should grow
   accordingly.
2. Load one of your own photos with `cv2.imread` and run the same pipeline
   on it.
3. Change the Canny thresholds (50, 150) and see the edges change - low
   thresholds find more (noisier) edges.
4. Try the Haar cascade face detector (section 8.5) on a group photo.

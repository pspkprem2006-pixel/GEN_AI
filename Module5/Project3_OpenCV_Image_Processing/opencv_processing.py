"""
Project 3 - OpenCV Image Processing
Module 5: Deep Learning & Computer Vision

The vision fundamentals every system uses before deep learning (notes
section 8.4). Generates a 600x400 sample image containing ~5 simple
shapes, then applies the full processing pipeline:

  1. grayscale       - cv2.cvtColor
  2. blur            - cv2.GaussianBlur
  3. edges           - cv2.Canny
  4. threshold       - cv2.threshold
  5. findContours    - count the objects
  6. draw            - box each detected object

Prints the contour count and saves a 2x3 montage PNG ('processing_steps.png')
showing original / grayscale / blur / edges / threshold / contours.
"""

import matplotlib
matplotlib.use("Agg")   # headless: save PNGs, no window
import matplotlib.pyplot as plt
import cv2
import numpy as np


def create_sample_image():
    """Build a 600x400 RGB image with 5 filled shapes."""
    img = np.full((400, 600, 3), 255, dtype=np.uint8)   # white background
    cv2.rectangle(img, (40, 40), (160, 160), (40, 120, 220), -1)    # blue square
    cv2.circle(img, (280, 100), 60, (40, 200, 60), -1)             # green circle
    cv2.rectangle(img, (420, 40), (540, 160), (240, 140, 40), -1)  # teal rect
    pts = np.array([[100, 340], [200, 260], [300, 340],
                    [200, 400]], dtype=np.int32)
    cv2.fillPoly(img, [pts], (70, 70, 220))                        # purple diamond
    cv2.circle(img, (480, 320), 70, (40, 40, 40), -1)              # dark circle
    print(f"Sample image created: 600x400 pixels, 3 color channels "
          f"(array shape {img.shape}).")
    return img


def process(img):
    """The 5-step pipeline: grayscale, blur, edges, threshold, contours."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    edges = cv2.Canny(blurred, 50, 150)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    print(f"\nContour detection found {len(contours)} object(s) in the image.")
    return gray, blurred, edges, thresh, contours


def draw_contours(img, contours):
    """Copy of the original with every detected object boxed in red."""
    boxed = img.copy()
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(boxed, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv2.putText(boxed, "object", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)
    return boxed


def save_montage(img, gray, blurred, edges, thresh, boxed):
    """Save a 2x3 montage of all six steps to processing_steps.png."""
    titles = ["1. Original", "2. Grayscale", "3. Blur",
              "4. Edges (Canny)", "5. Threshold", "6. Contours found"]
    images = [img, gray, blurred, edges, thresh, boxed]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, image, title in zip(axes.flatten(), images, titles):
        ax.imshow(image, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig("processing_steps.png", dpi=110)
    plt.close(fig)
    print(f"[OK] Montage saved to 'processing_steps.png'.")


def main():
    print("=" * 60)
    print("PROJECT 3 - OPENCV IMAGE PROCESSING (Module 5)")
    print("=" * 60)

    img = create_sample_image()
    gray, blurred, edges, thresh, contours = process(img)
    boxed = draw_contours(img, contours)
    save_montage(img, gray, blurred, edges, thresh, boxed)


if __name__ == "__main__":
    main()

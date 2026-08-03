"""
Project 2 - Object Detection with YOLO
Module 5: Deep Learning & Computer Vision

Runs the pre-trained YOLOv8 nano model (yolov8n.pt, ~6 MB, downloaded on
first run) on a sample photo. Transfer learning in action - no training
needed, the model already knows 80 object classes.

Prints every detection with its confidence, plus a per-class summary
exactly like the notes' sample output, and saves the annotated image
('detection_result.png') with boxes and labels drawn by YOLO.

The sample photo is 'sample_photo.jpg' (the classic YOLO street/bus photo,
downloaded from the Ultralytics GitHub assets).
"""

from ultralytics import YOLO
import cv2

IMAGE_PATH = "sample_photo.jpg"
OUTPUT_PATH = "detection_result.png"


def run_detection(model, image_path):
    """Run YOLO on the image; print detections and per-class summary."""
    results = model(image_path)
    boxes = results[0].boxes
    names = model.names

    print("\n----- OBJECTS DETECTED -----")
    for box in boxes:
        label = names[int(box.cls)]
        conf = float(box.conf)
        print(f"   {label:<12} (confidence {conf * 100:.0f}%)")

    counts = {}
    for box in boxes:
        label = names[int(box.cls)]
        counts[label] = counts.get(label, 0) + 1
    summary = ", ".join(f"{n} x {label}" for label, n in counts.items())
    print(f"\nSummary (counts):  {summary}")
    return results


def save_annotated(results):
    """Save the annotated image (results[0].plot() -> BGR numpy array)."""
    annotated = results[0].plot()
    cv2.imwrite(OUTPUT_PATH, annotated)
    print(f"[OK] Annotated image saved to '{OUTPUT_PATH}'.")


def main():
    print("=" * 60)
    print("PROJECT 2 - OBJECT DETECTION WITH YOLO (Module 5)")
    print("=" * 60)
    print("Loading YOLOv8n model (yolov8n.pt - downloaded on first run)...")

    model = YOLO("yolov8n.pt")
    print(f"Model loaded. Classifying image: {IMAGE_PATH}")

    results = run_detection(model, IMAGE_PATH)
    save_annotated(results)


if __name__ == "__main__":
    main()

# Project 2 - Object Detection with YOLO

Detect and label objects in a real photo with a state-of-the-art model -
in about 15 lines (Module 5, section 13).

Runs the pre-trained YOLOv8 nano model (`yolov8n.pt`, ~6 MB) on a real
street photo. This is transfer learning in action (section 10): the model
was already trained on 80 object classes; you just *use* it - no training
required.

## Files
- `object_detection.py` - the full program
- `sample_photo.jpg` - the test image: the classic YOLO street/bus photo
  (bus + people + stop sign), downloaded from the official Ultralytics
  website, so the detections match the notes' sample output
- `detection_result.png` - generated when run: the photo with YOLO's
  bounding boxes and labels drawn on it
- `README.md` - this file

## How to run
```bash
pip install ultralytics   # if not installed (installs torch + opencv)
python object_detection.py
```

First run downloads `yolov8n.pt` (~6 MB) from GitHub - allow internet and
be patient. Afterwards runs are instant and offline.

## Requirements
- Python 3.x, ultralytics (v8.4.115 installed), torch, opencv-python

## Expected output (sample run)
```
----- OBJECTS DETECTED -----
   bus          (confidence 87%)
   person       (confidence 87%)
   person       (confidence 85%)
   person       (confidence 83%)
   person       (confidence 26%)
   stop sign    (confidence 26%)

Summary (counts):  1 x bus, 4 x person, 1 x stop sign
[OK] Annotated image saved to 'detection_result.png'.
```

This matches the notes' sample output (bus 87%, persons 87/85/83%, stop
sign 26%) - a fourth person is also detected at 26%.

## Challenges
1. Point it at your own photos - change `IMAGE_PATH` and re-run; note what
   it detects and what it misses (small, distant, or overlapping objects
   are common misses).
2. Filter the detections to count only people (`person` class).
3. Try a larger model (`yolov8s.pt`) - it usually detects more and more
   accurately, at the cost of speed.
4. Loop over a whole folder of photos with `model(folder)` - YOLO accepts
   directories and batches them.

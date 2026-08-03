# Module 5 - Deep Learning & Computer Vision

Completed hands-on tasks for Module 5 of the AI Powered Engineering
Upskilling Program.

This module covers the jump from classic machine learning to deep
learning: neural networks (neurons, activation functions, backpropagation,
gradient descent), CNNs (convolution, pooling), images as data, OpenCV
processing, and YOLO object detection with a pre-trained model (transfer
learning).

## Task Files (one file per task)

| Task | Folder / File | What it does |
|---|---|---|
| Project 1 | `Project1_Digit_Recognition/` | Digit Recognition - trains a 2-hidden-layer neural network (64 -> 32, scikit-learn MLPClassifier) on the 8x8 digits dataset, prints accuracy + misclassified count, and saves a colour-coded prediction grid (`digit_grid.png`). |
| Project 2 | `Project2_Object_Detection_YOLO/` | Object Detection - runs pre-trained YOLOv8n on a real street photo, prints every detection with confidence plus a per-class summary, and saves the annotated image (`detection_result.png`). Downloads yolov8n.pt (~6 MB) on first run. |
| Project 3 | `Project3_OpenCV_Image_Processing/` | OpenCV Image Processing - generates a 600x400 image of 5 shapes and applies the full vision pipeline (grayscale, blur, edges, threshold, contours), printing the contour count and saving a 2x3 montage (`processing_steps.png`). |
| Practice | `Practice_Exercises/answers.md` | Question + answer for every practice exercise in section 17 of the notes (concept checks, neural network coding, OpenCV, YOLO, integrative, quiz) with runnable code. |

## How to run

Python 3.13 with installed libraries: numpy, pandas, scikit-learn,
matplotlib, opencv-python, ultralytics (YOLOv8), torch. No TensorFlow
needed - Project 1 uses scikit-learn's MLPClassifier, the exact
architecture of the notes' Keras model.

```bash
cd Project1_Digit_Recognition && python digit_recognition.py
cd Project2_Object_Detection_YOLO && python object_detection.py
cd Project3_OpenCV_Image_Processing && python opencv_processing.py
```

Each script runs headless (matplotlib 'Agg') and saves a PNG you open
afterwards. Project 2 downloads the ~6 MB YOLOv8n weights on its first
run - allow internet the first time.

## Key concepts demonstrated

- A neural network is scikit-learn's `MLPClassifier` - two Dense layers
  (64 -> 32), ReLU activation, softmax output, trained by gradient descent
- Images are NumPy arrays; OpenCV loads them in BGR
- The vision pipeline: grayscale -> blur -> edges -> threshold -> contours
- YOLO = transfer learning: use a pre-trained model, no training required

## Results

- Project 1: ~97% test accuracy on unseen handwriting (98% in the notes'
  sample run - the split seed shifts the exact number)
- Project 2: 4 persons, 1 bus, 1 stop sign detected on the street photo
- Project 3: all 5 generated shapes detected and boxed

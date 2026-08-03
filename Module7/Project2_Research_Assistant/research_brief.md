# Research Brief: How CNNs work
## 1. Overview
A Convolutional Neural Network (CNN) is a deep learning model built for image-like data. Instead of connecting every pixel to every neuron, a CNN slides small filters over the image to detect local patterns such as edges, textures, and shapes. These patterns are stacked into layers, so later layers detect bigger structures like eyes, wheels, or faces.
## 2. Key Concepts
- Convolution: sliding a small filter over the image to find patterns.
- Kernel / filter: the small matrix of weights that does the detecting.
- Stride and padding: how fast the filter moves and how edges are handled.
- Pooling: downsampling to keep the important info and shrink size.
- Feature map: the output of a convolution layer.
- Fully connected layer: the final classifier on top of the features.
## 3. Important Questions
- Why are filters shared across the image instead of per-pixel weights?
- What does each layer of a CNN actually learn?
- Why does pooling reduce overfitting?
- How do CNNs work on non-image data like audio or text?
- What is a receptive field, and why does it grow with depth?
## 4. Subtopics to Study Next
- Popular architectures: LeNet, AlexNet, ResNet.
- Data augmentation and transfer learning.
- Training tricks: batch norm, dropout, learning-rate schedules.
- Applications: object detection (YOLO), segmentation (U-Net).
- Visualizing what CNNs learn (feature-map visualization).
## 5. How to Learn More
- Work through an online course on deep learning for computer vision.
- Read the original papers describing each architecture.
- Build and train a small CNN from scratch on a toy dataset.
- Follow reputable tutorials that show working code, and verify sources.

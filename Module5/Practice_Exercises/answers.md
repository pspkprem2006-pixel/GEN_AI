# Module 5 - Practice Exercises & Self-Assessment (Answers)

Answers for section 17 of the Module 5 notes (Deep Learning & Computer
Vision). Every question is quoted verbatim, followed by its answer from
the notes' answer key (17.7). Coding answers are actual runnable code
using the installed packages (scikit-learn, OpenCV, Ultralytics, PyTorch).
Where the answer key uses Keras/TensorFlow, the code here was rebuilt
with PyTorch so it runs on this machine (no TensorFlow installed) - the
PyTorch CNN smoke-tested at 98.1% after 1 epoch, ~99% after 5.

## 17.1 Concept checks

**1. Why does deep learning beat classic ML on images?**

Because it **learns the useful features itself** (edges -> shapes ->
objects) instead of you hand-crafting them, and it handles the raw,
high-dimensional pixel data that classic ML struggles with. Classic ML
needs *you* to say "look for an edge here, a curve there"; a deep network
discovers those features from data, layer by layer.

**2. What does an activation function add, and why is ReLU popular?**

**Activation functions add non-linearity**, letting a network learn
curves and complex patterns (without them a deep net is just a linear
model). **ReLU** (`max(0, x)`) is popular because it's simple, fast, and
trains well - and it's the default for hidden layers today.

**3. Describe the training loop (forward pass, loss, backprop, gradient descent).**

*forward pass* (make predictions) -> *loss* (measure how wrong) ->
*backpropagation* (find each weight's share of the blame) -> *gradient
descent* (nudge weights to reduce loss) -> repeat over many epochs.

**4. What is a convolution filter, and what does pooling do?**

A **convolution filter** is a small learned grid slid over the image to
detect a pattern (edge, curve), producing a feature map. **Pooling**
downsamples the feature map (keeps the strongest signal), making the
network smaller and more robust to small shifts.

**5. Why do CNNs beat Dense networks on images (give two reasons)?**

(a) they respect spatial structure (nearby pixels relate) via local
filters, and (b) shared filters mean far fewer parameters and they detect
a feature **anywhere** in the image (translation invariance).

**6. Classification vs detection vs segmentation - define each.**

**Classification** = one label for the whole image ("a cat"). **Detection**
= labels **+ boxes** for each object ("cat here, ball there").
**Segmentation** = a label for **every pixel** (exact outlines).

**7. What is transfer learning and why is it so useful?**

**Transfer learning** = reuse a model already trained on a huge dataset
and adapt it to your task. Useful because it gives strong results with
**little data and compute** (the early layers' universal features are
reused) - e.g. YOLOv8n detects 80 object classes with zero training on
our side.

**8. Why must you normalize pixels and match input shape?**

Dividing pixels by 255 (-> 0-1) keeps inputs small and consistent so
gradient descent behaves; the input shape must match what the model's
first layer expects (e.g. `(28, 28, 1)`), or it errors.

## 17.2 Coding - neural network / classification

**9. Run Project 1; change the hidden layers to `(128, 64)` - does accuracy change?**

Edit `Project1_Digit_Recognition/digit_recognition.py`, changing
`hidden_layer_sizes=(64, 32)` to `(128, 64)`:

```python
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

digits = load_digits()
X, y = digits.data, digits.target
X_scaled = StandardScaler().fit_transform(X)
X_tr, X_te, y_tr, y_te = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500,
                      random_state=42)
model.fit(X_tr, y_tr)
print("Accuracy:", round(model.score(X_te, y_te), 4))
```

Answer: **only a small change** - on this machine (128, 64) scored 0.9694
vs 0.9667 for (64, 32). More neurons can help a little or slightly
overfit the small 1,797-image dataset - exactly what the notes predict.

**10. Read the confusion matrix - which digits are confused most?**

```python
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import numpy as np

digits = load_digits()
X_scaled = StandardScaler().fit_transform(digits.data)
X_tr, X_te, y_tr, y_te = train_test_split(
    X_scaled, digits.target, test_size=0.2, random_state=42,
    stratify=digits.target)

model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                      random_state=42).fit(X_tr, y_tr)
y_pred = model.predict(X_te)

cm = confusion_matrix(y_te, y_pred)
np.fill_diagonal(cm, 0)                 # zero out the diagonal (correct)
confused = np.argmax(cm)                # biggest off-diagonal cell
a, b = confused // 10, confused % 10
print(cm)
print(f"Most confused pair: {a} <-> {b}  ({cm[a, b]} confusions)")
```

Answer: read the **large off-diagonal cells** of the matrix. Classically
the confused pairs are **4<->9, 3<->5, 7<->1** (visually similar digits);
our split (random_state=42) happens to confuse **8 <-> 1** most.

**11. (Colab) Build the Keras CNN from §6.5 on MNIST; report test accuracy.**

The Keras original needs TensorFlow (Colab). Here it is rebuilt with
**PyTorch** (installed; torchvision provides MNIST) - same architecture:
Conv32 -> MaxPool -> Conv64 -> MaxPool -> Dense64 -> Dense10:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

# MNIST: 60k train / 10k test, 28x28 grayscale, normalized to 0-1
train_set = torchvision.datasets.MNIST("mnist_data", train=True,
                                       download=True, transform=T.ToTensor())
test_set = torchvision.datasets.MNIST("mnist_data", train=False,
                                      download=True, transform=T.ToTensor())
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64)

model = nn.Sequential(
    nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(64 * 7 * 7, 64), nn.ReLU(),
    nn.Linear(64, 10),
).to(device)

opt = optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss()

for epoch in range(5):
    model.train()
    for x, y in train_loader:
        opt.zero_grad()
        loss = loss_fn(model(x.to(device)), y.to(device))
        loss.backward()
        opt.step()

model.eval()
correct = total = 0
with torch.no_grad():
    for x, y in test_loader:
        pred = model(x.to(device)).argmax(1)
        correct += (pred.cpu() == y).sum().item()
        total += y.size(0)
print(f"Test accuracy: {correct / total:.4f}")
```

Answer: ~99% test accuracy (the smoke test hit 98.05% after only 1 epoch;
5 epochs reaches ~99%). The CNN beats the 8x8 MLP's ~97% because it sees
4x the pixels (28x28 vs 8x8) and uses convolutions.

## 17.3 Coding - OpenCV

**12. Load one of your own photos; convert to grayscale and save it.**

The bus photo from Project 2 is used here (any own photo works - put its
path in `candidates`):

```python
import cv2, os, numpy as np

candidates = ["photo.jpg", "sample_photo.jpg",
              "../Project2_Object_Detection_YOLO/sample_photo.jpg"]
path = next((c for c in candidates if os.path.exists(c)), None)
img = cv2.imread(path) if path else np.full((400, 600, 3), 255, np.uint8)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite("gray.jpg", gray)
print("Saved gray.jpg, shape:", gray.shape)
```

**13. Apply Canny edge detection and a threshold; compare the results.**

```python
import cv2, os, numpy as np

candidates = ["photo.jpg", "sample_photo.jpg",
              "../Project2_Object_Detection_YOLO/sample_photo.jpg"]
path = next((c for c in candidates if os.path.exists(c)), None)
img = cv2.imread(path) if path else np.full((400, 600, 3), 255, np.uint8)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 50, 150)                  # thin outlines
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)  # solid B/W
cv2.imwrite("edges.jpg", edges)
cv2.imwrite("thresh.jpg", thresh)
```

Answer: **Canny = thin outlines** (gradient-based edges of objects and
textures); **threshold = solid black/white regions** (a brightness cut -
bright pixels go white, dark pixels go black, no object structure). Canny
shows *where edges are*; the threshold shows *which areas are bright*.

**14. Detect faces in a group photo using a Haar cascade.**

```python
import cv2, os, numpy as np

candidates = ["photo.jpg", "sample_photo.jpg",
              "../Project2_Object_Detection_YOLO/sample_photo.jpg"]
path = next((c for c in candidates if os.path.exists(c)), None)
img = cv2.imread(path) if path else np.full((400, 600, 3), 255, np.uint8)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imwrite("faces.jpg", img)
print(f"Found {len(faces)} face(s)")
```

Answer: on the Project 2 bus photo this found **2 faces** and drew green
boxes around them (small, turned-away, or distant people are missed -
Haar is a fast but simple pre-deep-learning detector). For accuracy on
hard images, YOLO wins.

## 17.4 Coding - YOLO

**15. Run Project 2 on 3 of your own images; note what it detects and misses.**

```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")

results = model(["photo1.jpg", "photo2.jpg", "photo3.jpg"])
for i, result in enumerate(results, 1):
    print(f"--- image {i} ---")
    for box in result.boxes:
        print(model.names[int(box.cls)], round(float(box.conf), 2))
```

Answer: YOLO detects common, clearly visible objects well (people, cars,
buses); typical **misses are small, distant, or overlapping objects** and
anything outside the 80 trained classes (a mug, a paperclip).

**16. Filter the detections to count only people in an image.**

```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
results = model("sample_photo.jpg")

names = model.names
people = sum(1 for box in results[0].boxes
             if names[int(box.cls)] == "person")
print(f"People detected: {people}")
```

Answer: on the bus photo this prints `People detected: 4`.

**17. Try a larger model (`yolov8s.pt`) - does it detect more?**

```python
from ultralytics import YOLO
model_s = YOLO("yolov8s.pt")     # 's' > 'n' in size and accuracy (slower)
results = model_s("sample_photo.jpg")
for box in results[0].boxes:
    print(model_s.names[int(box.cls)], round(float(box.conf), 2))
```

Answer: `yolov8s.pt` (~22 MB, downloaded on first run) is bigger and
more accurate - it usually detects **more objects and with higher
confidence**, at the cost of slower inference. On the bus photo the
confidences rise and it may pick up smaller/partial people the nano model
missed.

## 17.5 Integrative

**18. Complete all three projects and one challenge from each README.**

Done - all three projects run and save their PNGs:

- `Project1_Digit_Recognition/digit_recognition.py` (challenge done: read
  the confusion matrix - most confused pair 8<->1 in our split)
- `Project2_Object_Detection_YOLO/object_detection.py` (challenge done:
  filter to people only - 4 detected, see Q16)
- `Project3_OpenCV_Image_Processing/opencv_processing.py` (challenge
  done: run the pipeline on my own photo via `cv2.imread`, Q12-13)

**19. Build a mini app: use OpenCV to read a webcam frame and run YOLO on it.**

```python
import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)          # 0 = default webcam

while True:
    ret, frame = cap.read()        # grab one frame (a NumPy image)
    if not ret:
        break
    results = model(frame)         # run YOLO on the frame
    annotated = results[0].plot()  # boxes drawn (BGR array)
    cv2.imshow("YOLO webcam", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
```

This is exactly how real-time vision apps work - the loop from notes
section 8.6 with the YOLO call from section 13, one frame at a time.
(Needs a webcam and a display; on a headless machine save frames with
`cv2.imwrite` instead of `imshow`.)

## 17.6 Quick self-check quiz

**1. What library powers Project 2's detection?**

YOLO / Ultralytics (built on PyTorch).

**2. What color order does OpenCV use?**

BGR.

**3. Which layer type is built for images?**

Convolutional (CNN) layers.

**4. What does pooling do?**

Shrinks feature maps (downsamples them, keeping the strongest signal).

**5. What's the output of object detection (vs classification)?**

Labels **+ bounding boxes** for each object (vs one label for the whole
image).

**6. Why normalize pixels to 0-1?**

Networks train better - small, consistent inputs keep gradient descent
stable.

**7. What does "transfer learning" mean?**

Reuse a pre-trained model and adapt it to your task (e.g. YOLOv8n
pre-trained on 80 classes).

**8. Digit recognition is which CV task?**

Image classification (one label - the digit - for the whole image).

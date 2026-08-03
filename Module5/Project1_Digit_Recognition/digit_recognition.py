"""
Project 1 - Digit Recognition
Module 5: Deep Learning & Computer Vision

Trains a neural network (scikit-learn MLPClassifier, 2 hidden layers
64 -> 32, max_iter=500) on the classic 8x8 load_digits dataset. This is
the exact architecture of the Keras model in notes section 4.2, rebuilt
with scikit-learn so it runs anywhere without TensorFlow.

Prints the test accuracy and the number of misclassified digits, then
saves a PNG grid ('digit_grid.png') of test images, each labelled with
the true label and the prediction, coloured GREEN = correct,
RED = wrong.
"""

import matplotlib
matplotlib.use("Agg")   # headless: save PNGs, no window
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import numpy as np


def load_and_scale():
    """Load the digits dataset and scale pixels to zero mean / unit variance."""
    digits = load_digits()
    X, y = digits.data, digits.target
    print(f"Loaded {len(X)} images, each 8x8 pixels, labelled 0-9.")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, digits


def train_model(X_train_scaled, y_train):
    """Train the 2-hidden-layer neural network (64 -> 32 neurons)."""
    print("Training a neural network (2 hidden layers: 64 -> 32)...")
    model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                          random_state=42)
    model.fit(X_train_scaled, y_train)
    return model


def count_misclassified(y_true, y_pred):
    """Number of labels the model got wrong."""
    return int((y_true != y_pred).sum())


def save_prediction_grid(digits, X_test, y_test, y_pred):
    """Save a PNG grid of test images coloured by correct/wrong prediction."""
    n = len(y_test)
    cols = 6
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.6))
    axes = axes.flatten()
    correct = 0
    for ax, img, true, pred in zip(axes, X_test, y_test, y_pred):
        ax.imshow(img.reshape(8, 8), cmap="gray_r")
        color = "green" if pred == true else "red"
        correct += pred == true
        ax.set_title(f"T:{true} P:{pred}", color=color, fontsize=9)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Digit grid - green = correct, red = wrong",
                 fontsize=13, y=0.995)
    fig.tight_layout()
    fig.savefig("digit_grid.png", dpi=110)
    plt.close(fig)
    print(f"[OK] Grid of {n} test images saved to 'digit_grid.png' "
          f"({correct} correct, {n - correct} wrong).")


def main():
    print("=" * 60)
    print("PROJECT 1 - DIGIT RECOGNITION (Module 5)")
    print("=" * 60)

    X_scaled, y, digits = load_and_scale()

    X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    model = train_model(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    accuracy = model.score(X_test_scaled, y_test)
    mis = count_misclassified(y_test, y_pred)

    print(f"\nTest accuracy: {accuracy:.3f}  ({accuracy * 100:.1f}% of "
          f"digits correct)")
    print(f"Total misclassified: {mis} out of {len(y_test)} test images.")

    cm = confusion_matrix(y_test, y_pred)
    np.fill_diagonal(cm, 0)
    confused = np.argmax(cm)
    a, b = confused // 10, confused % 10
    print(f"Confusion matrix (off-diagonal): most confused pair is "
          f"{a} <-> {b}.")

    save_prediction_grid(digits, X_test_scaled, y_test, y_pred)


if __name__ == "__main__":
    main()

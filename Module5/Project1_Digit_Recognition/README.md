# Project 1 - Digit Recognition

Train a neural network to read handwritten digits - the "hello world" of
computer vision (Module 5, section 12).

Uses the 8x8 `load_digits` dataset (1,797 images, labels 0-9) with a
scikit-learn `MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500)` -
two hidden layers of 64 and 32 neurons, the exact architecture of the
Keras model in the notes' section 4.2, rebuilt with scikit-learn so it
runs anywhere without TensorFlow.

## Files
- `digit_recognition.py` - the full program
- `digit_grid.png` - generated when run: grid of all test images, labelled
  "T:<true> P:<prediction>", green = correct, red = wrong
- `README.md` - this file

## How to run
```bash
pip install scikit-learn matplotlib numpy   # if not installed
python digit_recognition.py
```

The script prints a header, dataset size, the training message, the test
accuracy, the misclassified count, the most-confused digit pair (from the
confusion matrix), and a confirmation that `digit_grid.png` was saved.

## Requirements
- Python 3.x, numpy, scikit-learn, matplotlib (all installed)

## Expected output (sample run)
```
Loaded 1797 images, each 8x8 pixels, labelled 0-9.
Training a neural network (2 hidden layers: 64 -> 32)...

Test accuracy: 0.967  (96.7% of digits correct)
Total misclassified: 12 out of 360 test images.
Confusion matrix (off-diagonal): most confused pair is 8 <-> 1.
[OK] Grid of 360 test images saved to 'digit_grid.png' (348 correct, 12 wrong).
```

The notes' sample run reached 98.1% with 7 errors - the exact numbers move
with the random train/test split; the pipeline is identical.

## Challenges
1. Change the hidden layers to `(128, 64)` - does the accuracy change?
   (Often a small change; more neurons can help or slightly overfit a
   small dataset.)
2. Read the confusion matrix properly - which digits are confused most?
   (Classically 4<->9, 3<->5, 7<->1; our split happens to confuse 8<->1.)
3. Normalize pixels to 0-1 instead of StandardScaler and compare.
4. Rebuild as a Keras CNN on the full 28x28 MNIST (notes section 6.5) to
   beat 98% - needs TensorFlow/Colab.

from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix


# ============================================================
# NIRMAAN AI - TACO CROPPED IMAGE EVALUATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "ai" / "models" / "efficientnetb0_oversampled.keras"
CROP_DIR = PROJECT_ROOT / "data" / "processed" / "messy_crops"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
]


print("=" * 60)
print("NIRMAAN AI - TACO CROPPED IMAGE EVALUATION")
print("=" * 60)

print(f"Model : {MODEL_PATH}")
print(f"Data  : {CROP_DIR}")
print()


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

model = load_model(MODEL_PATH)

print("Model loaded successfully.")
print()


# ------------------------------------------------------------
# Load cropped TACO images
# ------------------------------------------------------------

test_ds = tf.keras.utils.image_dataset_from_directory(
    CROP_DIR,
    labels="inferred",
    label_mode="categorical",
    class_names=CLASS_NAMES,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# ------------------------------------------------------------
# Evaluate
# ------------------------------------------------------------

loss, accuracy = model.evaluate(test_ds, verbose=1)

print()
print("=" * 60)
print("CROPPED TACO RESULTS")
print("=" * 60)

print(f"Images   : {len(test_ds.file_paths)}")
print(f"Loss     : {loss:.4f}")
print(f"Accuracy : {accuracy * 100:.2f}%")
print()


# ------------------------------------------------------------
# Predictions
# ------------------------------------------------------------

y_true = np.concatenate(
    [y.numpy() for _, y in test_ds],
    axis=0
)

y_true = np.argmax(y_true, axis=1)

predictions = model.predict(test_ds, verbose=1)

y_pred = np.argmax(predictions, axis=1)


# ------------------------------------------------------------
# Classification report
# ------------------------------------------------------------

print()
print("Classification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
)


# ------------------------------------------------------------
# Confusion matrix
# ------------------------------------------------------------

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))


# ------------------------------------------------------------
# Comparison with original held-out messy result
# ------------------------------------------------------------

original_accuracy = 36.11
crop_accuracy = accuracy * 100

improvement = crop_accuracy - original_accuracy

print()
print("=" * 60)
print("ORIGINAL vs CROPPED COMPARISON")
print("=" * 60)

print(f"Original messy accuracy : {original_accuracy:.2f}%")
print(f"Cropped accuracy        : {crop_accuracy:.2f}%")
print(f"Change                  : {improvement:+.2f} percentage points")
print("=" * 60)
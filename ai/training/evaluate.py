from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

# ============================================================
# NIRMAAN AI - CLEAN VS MESSY EVALUATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "ai" / "models" / "efficientnetb0_oversampled.keras"

CLEAN_TEST_DIR = ROOT / "data" / "processed" / "clean" / "test"
MESSY_DIR = ROOT / "data" / "processed" / "messy"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

print("=" * 60)
print("NIRMAAN AI - CLEAN VS MESSY EVALUATION")
print("=" * 60)

print(f"\nModel: {MODEL_PATH}")

# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

print("\nLoading trained model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")

# ------------------------------------------------------------
# Load datasets
# ------------------------------------------------------------

print("\nLoading clean test dataset...")

clean_ds = tf.keras.utils.image_dataset_from_directory(
    CLEAN_TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    label_mode="categorical"
)

print("\nLoading messy TACO dataset...")

messy_ds = tf.keras.utils.image_dataset_from_directory(
    MESSY_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    label_mode="categorical"
)

class_names = clean_ds.class_names

print("\nClasses:")
for i, name in enumerate(class_names):
    print(f"  {i}: {name}")

# ------------------------------------------------------------
# Evaluation helper
# ------------------------------------------------------------

def evaluate_dataset(dataset, name):
    print("\n" + "=" * 60)
    print(f"{name.upper()} EVALUATION")
    print("=" * 60)

    loss, accuracy = model.evaluate(dataset, verbose=1)

    print(f"\n{name} Loss     : {loss:.4f}")
    print(f"{name} Accuracy : {accuracy * 100:.2f}%")

    y_true = []
    y_pred = []

    for images, labels in dataset:
        predictions = model.predict(images, verbose=0)

        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(predictions, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print(f"\n{name} Classification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            digits=4,
            zero_division=0
        )
    )

    print(f"{name} Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    return accuracy, y_true, y_pred


# ------------------------------------------------------------
# Evaluate clean test set
# ------------------------------------------------------------

clean_accuracy, clean_true, clean_pred = evaluate_dataset(
    clean_ds,
    "Clean Test"
)

# ------------------------------------------------------------
# Evaluate messy TACO set
# ------------------------------------------------------------

messy_accuracy, messy_true, messy_pred = evaluate_dataset(
    messy_ds,
    "Messy TACO"
)

# ------------------------------------------------------------
# Robustness calculation
# ------------------------------------------------------------

degradation = clean_accuracy - messy_accuracy

print("\n" + "=" * 60)
print("ROBUSTNESS RESULT")
print("=" * 60)

print(f"\nClean Accuracy : {clean_accuracy * 100:.2f}%")
print(f"Messy Accuracy : {messy_accuracy * 100:.2f}%")
print(f"Accuracy Drop  : {degradation * 100:.2f} percentage points")

if degradation >= 0:
    print("\nThe model shows an accuracy degradation under messy conditions.")
else:
    print("\nMessy accuracy is higher than clean accuracy for this sample.")

print("\nEvaluation complete.")
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "ai" / "models" / "efficientnetb0_oversampled.keras"
TEST_DIR = PROJECT_ROOT / "data" / "processed" / "messy_split" / "test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Loading messy held-out test set...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_names=CLASS_NAMES,
    label_mode="categorical",
    shuffle=False
)

loss, accuracy = model.evaluate(test_ds, verbose=1)

print("\n========================================")
print("HELD-OUT MESSY TEST RESULTS")
print("========================================")
print(f"Images: {36}")
print(f"Loss: {loss:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")

y_true = np.concatenate([y.numpy() for _, y in test_ds])
y_true = np.argmax(y_true, axis=1)

predictions = model.predict(test_ds, verbose=0)
y_pred = np.argmax(predictions, axis=1)

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4
    )
)

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
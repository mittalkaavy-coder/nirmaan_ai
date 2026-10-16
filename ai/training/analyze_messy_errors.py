from pathlib import Path
import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "ai"
    / "models"
    / "efficientnetb0_oversampled.keras"
)

TEST_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "messy_split"
    / "test"
)

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

print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_names=CLASS_NAMES,
    label_mode="categorical",
    shuffle=False,
)

predictions = model.predict(test_ds, verbose=0)

y_true = np.concatenate(
    [y.numpy() for _, y in test_ds]
)

y_true = np.argmax(y_true, axis=1)
y_pred = np.argmax(predictions, axis=1)

file_paths = test_ds.file_paths

print("\n" + "=" * 80)
print("MISCLASSIFICATION ANALYSIS")
print("=" * 80)

correct = 0
wrong = 0

for path, actual, predicted, probs in zip(
    file_paths,
    y_true,
    y_pred,
    predictions
):
    confidence = float(np.max(probs))

    if actual == predicted:
        correct += 1
    else:
        wrong += 1

        print("\nWRONG")
        print(f"File       : {Path(path).name}")
        print(f"Actual     : {CLASS_NAMES[actual]}")
        print(f"Predicted  : {CLASS_NAMES[predicted]}")
        print(f"Confidence : {confidence:.4f}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"Correct: {correct}/36")
print(f"Wrong  : {wrong}/36")
print(f"Accuracy: {(correct / 36) * 100:.2f}%")
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "messy_error_grid.png"

IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
]

model = tf.keras.models.load_model(MODEL_PATH)

dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=36,
    class_names=CLASS_NAMES,
    label_mode="int",
    shuffle=False,
)

images, labels = next(iter(dataset))

predictions = model.predict(images, verbose=0)
predicted_labels = np.argmax(predictions, axis=1)

fig = plt.figure(figsize=(18, 20))

for i in range(36):
    ax = plt.subplot(6, 6, i + 1)

    ax.imshow(images[i].numpy().astype("uint8"))
    ax.axis("off")

    actual = CLASS_NAMES[labels[i].numpy()]
    predicted = CLASS_NAMES[predicted_labels[i]]
    confidence = np.max(predictions[i]) * 100

    if actual == predicted:
        title = f"{actual}\n✓ {confidence:.0f}%"
    else:
        title = f"{actual} → {predicted}\n{confidence:.0f}%"

    ax.set_title(title, fontsize=8)

plt.tight_layout()

plt.savefig(
    OUTPUT_FILE,
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print(f"Saved error grid to:")
print(OUTPUT_FILE)
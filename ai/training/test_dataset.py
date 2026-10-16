from pathlib import Path
import tensorflow as tf

keras = tf.keras

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_DIR = PROJECT_ROOT / "data" / "processed" / "clean" / "train"
VAL_DIR = PROJECT_ROOT / "data" / "processed" / "clean" / "val"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

print("=" * 50)
print("NIRMAAN AI - DATASET LOADING TEST")
print("=" * 50)

print("\nTensorFlow:", tf.__version__)

print("\nLoading training dataset...")

train_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42,
    label_mode="categorical"
)

print("\nLoading validation dataset...")

val_ds = keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    label_mode="categorical"
)

print("\nClasses:", train_ds.class_names)

print(
    "Training batches:",
    tf.data.experimental.cardinality(train_ds).numpy()
)

print(
    "Validation batches:",
    tf.data.experimental.cardinality(val_ds).numpy()
)

print("\nDataset loading: SUCCESS")
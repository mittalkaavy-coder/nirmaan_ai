from pathlib import Path

import tensorflow as tf

keras = tf.keras
layers = keras.layers
EfficientNetB0 = keras.applications.EfficientNetB0


# ============================================================
# NIRMAAN AI - EFFICIENTNETB0 FINE-TUNING
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_DIR = PROJECT_ROOT / "data" / "processed" / "clean" / "train"
VAL_DIR = PROJECT_ROOT / "data" / "processed" / "clean" / "val"

MODEL_DIR = PROJECT_ROOT / "ai" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "efficientnetb0_finetuned.keras"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
NUM_CLASSES = 6
SEED = 42
EPOCHS = 10


print("=" * 60)
print("NIRMAAN AI - EFFICIENTNETB0 FINE-TUNING")
print("=" * 60)

print("\nTensorFlow version:", tf.__version__)
print("Training directory:", TRAIN_DIR)
print("Validation directory:", VAL_DIR)


# ------------------------------------------------------------
# 1. Load datasets
# ------------------------------------------------------------

print("\nLoading training dataset...")

train_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
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

class_names = train_ds.class_names

print("\nClasses:")

for i, name in enumerate(class_names):
    print(f"  {i}: {name}")


# ------------------------------------------------------------
# 2. Data augmentation
# ------------------------------------------------------------

print("\nBuilding augmentation pipeline...")

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(
            height_factor=0.15,
            width_factor=0.15
        ),
        layers.RandomTranslation(
            height_factor=0.10,
            width_factor=0.10
        ),
        layers.RandomContrast(0.15),
    ],
    name="data_augmentation"
)


# ------------------------------------------------------------
# 3. Performance optimization
# ------------------------------------------------------------

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)


# ------------------------------------------------------------
# 4. Build EfficientNetB0
# ------------------------------------------------------------

print("\nBuilding EfficientNetB0 model...")

base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)


# ------------------------------------------------------------
# 5. Freeze first layers
# ------------------------------------------------------------

base_model.trainable = True

# Freeze most of the backbone.
# Only the last 30 layers will be fine-tuned.

for layer in base_model.layers[:-30]:
    layer.trainable = False

# Keep BatchNormalization layers frozen.
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False


trainable_layers = sum(
    1 for layer in base_model.layers
    if layer.trainable
)

print("\nFine-tuning configuration:")
print("Total backbone layers :", len(base_model.layers))
print("Trainable backbone layers:", trainable_layers)


# ------------------------------------------------------------
# 6. Build complete model
# ------------------------------------------------------------

inputs = keras.Input(
    shape=(224, 224, 3)
)

x = data_augmentation(inputs)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.2)(x)

outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)

model = keras.Model(
    inputs,
    outputs
)


# ------------------------------------------------------------
# 7. Compile with LOW learning rate
# ------------------------------------------------------------

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


model.summary()


# ------------------------------------------------------------
# 8. Callbacks
# ------------------------------------------------------------

callbacks = [

    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    ),

    keras.callbacks.ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )
]


# ------------------------------------------------------------
# 9. Fine-tune
# ------------------------------------------------------------

print("\nStarting fine-tuning...")

print(f"Epochs: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print("Learning rate: 1e-5")
print("Fine-tuned layers: last 30")


history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)


# ------------------------------------------------------------
# 10. Final validation result
# ------------------------------------------------------------

val_loss, val_accuracy = model.evaluate(
    val_ds,
    verbose=0
)


print("\n" + "=" * 60)
print("FINE-TUNING COMPLETE")
print("=" * 60)

print(f"Validation Loss     : {val_loss:.4f}")
print(f"Validation Accuracy : {val_accuracy * 100:.2f}%")

print("\nModel saved at:")
print(MODEL_PATH)
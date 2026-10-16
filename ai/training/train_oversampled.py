from pathlib import Path

import tensorflow as tf

keras = tf.keras
layers = keras.layers
EfficientNetB0 = keras.applications.EfficientNetB0


# ============================================================
# NIRMAAN AI - MESSY OVERSAMPLED TRAINING
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLEAN_TRAIN_DIR = (
    PROJECT_ROOT / "data" / "processed" / "clean" / "train"
)

MESSY_TRAIN_DIR = (
    PROJECT_ROOT / "data" / "processed" / "messy_split" / "train"
)

VAL_DIR = (
    PROJECT_ROOT / "data" / "processed" / "clean" / "val"
)

MODEL_DIR = PROJECT_ROOT / "ai" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "efficientnetb0_oversampled.keras"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
NUM_CLASSES = 6
SEED = 42
EPOCHS = 10

# Messy data exposure
MESSY_REPEAT = 5


print("=" * 60)
print("NIRMAAN AI - MESSY OVERSAMPLED TRAINING")
print("=" * 60)

print("\nTensorFlow version:", tf.__version__)

print("\nClean training directory:")
print(CLEAN_TRAIN_DIR)

print("\nMessy training directory:")
print(MESSY_TRAIN_DIR)

print("\nValidation directory:")
print(VAL_DIR)


# ------------------------------------------------------------
# 1. Load datasets
# ------------------------------------------------------------

print("\nLoading clean training dataset...")

clean_ds = keras.utils.image_dataset_from_directory(
    CLEAN_TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
    label_mode="categorical"
)

print("\nLoading messy training dataset...")

messy_ds = keras.utils.image_dataset_from_directory(
    MESSY_TRAIN_DIR,
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


class_names = clean_ds.class_names

print("\nClasses:")

for i, name in enumerate(class_names):
    print(f"  {i}: {name}")


# ------------------------------------------------------------
# 2. Repeat messy dataset
# ------------------------------------------------------------

print("\nPreparing oversampled training data...")

messy_ds = messy_ds.repeat(MESSY_REPEAT)

# Combine clean + repeated messy data
mixed_ds = clean_ds.concatenate(messy_ds)

# Shuffle combined dataset
mixed_ds = mixed_ds.shuffle(
    buffer_size=2000,
    seed=SEED,
    reshuffle_each_iteration=True
)


print("\nTraining data configuration:")

print("Clean images         : 1766")
print("Original messy images: 72")
print("Messy repeat factor  :", MESSY_REPEAT)
print("Effective messy exposure:", 72 * MESSY_REPEAT)

print(
    "Effective training exposure:",
    1766 + (72 * MESSY_REPEAT)
)


# ------------------------------------------------------------
# 3. Data augmentation
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
# 4. Performance optimization
# ------------------------------------------------------------

AUTOTUNE = tf.data.AUTOTUNE

mixed_ds = mixed_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)


# ------------------------------------------------------------
# 5. Build EfficientNetB0
# ------------------------------------------------------------

print("\nBuilding EfficientNetB0 model...")

base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)

base_model.trainable = False


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
# 6. Compile
# ------------------------------------------------------------

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


model.summary()


# ------------------------------------------------------------
# 7. Callbacks
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
# 8. Training
# ------------------------------------------------------------

print("\nStarting oversampled training...")

print("Epochs:", EPOCHS)
print("Batch size:", BATCH_SIZE)
print("Messy repeat factor:", MESSY_REPEAT)


# Important:
# Since messy_ds is repeated, explicitly set steps_per_epoch.
# One epoch = clean dataset + 5x messy exposure.

clean_batches = tf.data.experimental.cardinality(
    clean_ds
).numpy()

messy_batches = tf.data.experimental.cardinality(
    keras.utils.image_dataset_from_directory(
        MESSY_TRAIN_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
        label_mode="categorical"
    )
).numpy()

steps_per_epoch = clean_batches + (
    messy_batches * MESSY_REPEAT
)

print("\nSteps per epoch:", steps_per_epoch)

history = model.fit(
    mixed_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    callbacks=callbacks
)


# ------------------------------------------------------------
# 9. Final validation result
# ------------------------------------------------------------

val_loss, val_accuracy = model.evaluate(
    val_ds,
    verbose=0
)


print("\n" + "=" * 60)
print("OVERSAMPLED TRAINING COMPLETE")
print("=" * 60)

print(f"Validation Loss     : {val_loss:.4f}")
print(f"Validation Accuracy : {val_accuracy * 100:.2f}%")

print("\nModel saved at:")
print(MODEL_PATH)
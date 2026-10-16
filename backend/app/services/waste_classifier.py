from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = (
    PROJECT_ROOT
    / "ai"
    / "models"
    / "efficientnetb0_oversampled.keras"
)

CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
]

IMAGE_SIZE = (224, 224)


# --------------------------------------------------
# Load model once
# --------------------------------------------------

model = tf.keras.models.load_model(MODEL_PATH)


# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Convert an uploaded PIL image into the model's
    expected input shape: (1, 224, 224, 3).
    """

    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.asarray(image, dtype=np.float32)

    image_array = np.expand_dims(image_array, axis=0)

    return image_array


# --------------------------------------------------
# Prediction
# --------------------------------------------------

def predict_waste(image: Image.Image) -> dict:
    """
    Run waste classification on a PIL image.
    """

    image_array = preprocess_image(image)

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = int(np.argmax(predictions))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(predictions[predicted_index])

    return {
        "class_name": predicted_class,
        "confidence": confidence,
        "probabilities": {
            CLASS_NAMES[i]: float(predictions[i])
            for i in range(len(CLASS_NAMES))
        },
    }
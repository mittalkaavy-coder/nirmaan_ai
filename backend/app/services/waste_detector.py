
from pathlib import Path
from functools import lru_cache

from PIL import Image, ImageOps
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "detect"
    / "runs"
    / "nirmaan"
    / "taco_yolo_finetune"
    / "weights"
    / "best.pt"
)

# Bounded fallback thresholds: strict first, then more permissive.
CONFIDENCE_LEVELS = (0.25, 0.15, 0.10)


@lru_cache(maxsize=1)
def load_detector():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"YOLO model checkpoint not found: {MODEL_PATH}"
        )
    return YOLO(str(MODEL_PATH))


def detect_waste(image: Image.Image):
    image = ImageOps.exif_transpose(image).convert("RGB")
    model = load_detector()

    selected_result = None
    threshold_used = CONFIDENCE_LEVELS[-1]

    # Retry only when the stricter pass finds no objects.
    for threshold in CONFIDENCE_LEVELS:
        results = model.predict(
            source=image,
            imgsz=416,
            conf=threshold,
            iou=0.45,
            device="cpu",
            max_det=100,
            verbose=False,
        )

        result = results[0]
        selected_result = result
        threshold_used = threshold

        if result.boxes is not None and len(result.boxes) > 0:
            break

    detections = []
    result = selected_result

    if result is not None and result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().tolist()
        confidences = result.boxes.conf.cpu().tolist()
        class_ids = result.boxes.cls.cpu().tolist()

        for box, confidence, class_id in zip(
            boxes, confidences, class_ids
        ):
            class_id = int(class_id)
            detections.append({
                "class_name": result.names[class_id],
                "confidence": round(float(confidence), 4),
                "bbox": [round(float(value), 2) for value in box],
            })

    counts = {}
    for detection in detections:
        name = detection["class_name"]
        counts[name] = counts.get(name, 0) + 1

    return {
        "model": "Nirmaan AI YOLO11n fine-tuned on TACO",
        "image_width": image.width,
        "image_height": image.height,
        "total_detections": len(detections),
        "counts_by_class": counts,
        "detections": detections,
        "confidence_threshold_used": threshold_used,
        "adaptive_threshold": True,
        "experimental": True,
    }

from pathlib import Path
import json
from PIL import Image


# ============================================================
# NIRMAAN AI - TACO PADDED CROP CREATION
# Uses ONLY the 36 held-out messy test images
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ANNOTATIONS_FILE = (
    PROJECT_ROOT / "data" / "raw" / "taco" / "data" / "annotations.json"
)

INPUT_DIR = PROJECT_ROOT / "data" / "processed" / "messy_split" / "test"

OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "processed" / "messy_padded_crops"
)

PADDING_RATIO = 0.20

CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
]


print("=" * 60)
print("NIRMAAN AI - TACO PADDED CROP CREATION")
print("=" * 60)

print(f"Input : {INPUT_DIR}")
print(f"Output: {OUTPUT_DIR}")
print()


# ------------------------------------------------------------
# Load TACO annotations
# ------------------------------------------------------------

with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
    taco = json.load(f)

images = taco["images"]
annotations = taco["annotations"]
categories = taco["categories"]

image_lookup = {
    image["id"]: image
    for image in images
}

category_lookup = {
    category["id"]: category["name"]
    for category in categories
}


# ------------------------------------------------------------
# Same mapping used in previous experiments
# ------------------------------------------------------------

TACO_MAPPING = {
    "cardboard": {
        "Other carton",
        "Drink carton",
        "Corrugated carton",
        "Meal carton",
    },
    "glass": {
        "Glass bottle",
        "Broken glass",
    },
    "metal": {
        "Aluminium foil",
        "Food Can",
        "Drink can",
        "Metal bottle cap",
        "Pop tab",
        "Scrap metal",
    },
    "paper": {
        "Paper cup",
        "Tissues",
        "Normal paper",
        "Paper bag",
    },
    "plastic": {
        "Other plastic bottle",
        "Clear plastic bottle",
        "Plastic bottle cap",
        "Disposable plastic cup",
        "Plastic lid",
        "Other plastic",
        "Plastic film",
        "Garbage bag",
        "Other plastic wrapper",
        "Single-use carrier bag",
        "Crisp packet",
        "Disposable food container",
        "Plastic utensils",
        "Plastic straw",
        "Styrofoam piece",
    },
    "trash": {
        "Unlabeled litter",
        "Cigarette",
    },
}


# ------------------------------------------------------------
# Category -> Nirmaan class
# ------------------------------------------------------------

category_to_class = {}

for class_name, category_names in TACO_MAPPING.items():
    for category_name in category_names:
        category_to_class[category_name] = class_name


# ------------------------------------------------------------
# Build annotation lookup
# ------------------------------------------------------------

image_annotations = {}

for annotation in annotations:

    image_id = annotation["image_id"]

    category_name = category_lookup.get(
        annotation["category_id"]
    )

    if category_name in category_to_class:

        image_annotations.setdefault(
            image_id,
            []
        ).append(annotation)


# ------------------------------------------------------------
# Process ONLY messy_split/test
# ------------------------------------------------------------

processed = 0
failed = 0

counts = {
    class_name: 0
    for class_name in CLASS_NAMES
}


for class_name in CLASS_NAMES:

    input_class_dir = INPUT_DIR / class_name
    output_class_dir = OUTPUT_DIR / class_name

    output_class_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    image_files = sorted(
        [
            p for p in input_class_dir.iterdir()
            if p.suffix.lower() in {
                ".jpg",
                ".jpeg",
                ".png",
                ".webp",
            }
        ]
    )

    print(
        f"{class_name:<10}: {len(image_files)} input images"
    )

    for image_path in image_files:

        try:

            # ------------------------------------------------
            # Extract TACO image ID
            # taco_<id>_<filename>
            # ------------------------------------------------

            parts = image_path.name.split("_", 2)

            image_id = int(parts[1])

            if image_id not in image_lookup:
                raise ValueError(
                    "Image ID not found in TACO annotations"
                )

            if image_id not in image_annotations:
                raise ValueError(
                    "No mapped annotation found"
                )

            image = Image.open(
                image_path
            ).convert("RGB")

            width, height = image.size

            # ------------------------------------------------
            # Original TACO dimensions
            # ------------------------------------------------

            original_width = image_lookup[
                image_id
            ]["width"]

            original_height = image_lookup[
                image_id
            ]["height"]

            scale_x = width / original_width
            scale_y = height / original_height

            # ------------------------------------------------
            # Get bounding boxes belonging to this class
            # ------------------------------------------------

            boxes = []

            for annotation in image_annotations[image_id]:

                category_name = category_lookup[
                    annotation["category_id"]
                ]

                mapped_class = category_to_class.get(
                    category_name
                )

                if mapped_class == class_name:

                    boxes.append(
                        annotation["bbox"]
                    )

            if not boxes:
                raise ValueError(
                    "No matching bounding box"
                )

            # ------------------------------------------------
            # Combine bounding boxes
            # ------------------------------------------------

            x1 = min(
                box[0]
                for box in boxes
            )

            y1 = min(
                box[1]
                for box in boxes
            )

            x2 = max(
                box[0] + box[2]
                for box in boxes
            )

            y2 = max(
                box[1] + box[3]
                for box in boxes
            )

            # ------------------------------------------------
            # Scale coordinates
            # ------------------------------------------------

            x1 *= scale_x
            y1 *= scale_y
            x2 *= scale_x
            y2 *= scale_y

            # ------------------------------------------------
            # Add 20% padding
            # ------------------------------------------------

            box_width = x2 - x1
            box_height = y2 - y1

            pad_x = box_width * PADDING_RATIO
            pad_y = box_height * PADDING_RATIO

            x1 -= pad_x
            y1 -= pad_y

            x2 += pad_x
            y2 += pad_y

            # ------------------------------------------------
            # Clip to image boundaries
            # ------------------------------------------------

            x1 = max(
                0,
                int(x1)
            )

            y1 = max(
                0,
                int(y1)
            )

            x2 = min(
                width,
                int(x2)
            )

            y2 = min(
                height,
                int(y2)
            )

            if x2 <= x1 or y2 <= y1:
                raise ValueError(
                    "Invalid padded crop"
                )

            # ------------------------------------------------
            # Crop
            # ------------------------------------------------

            cropped = image.crop(
                (x1, y1, x2, y2)
            )

            output_path = (
                output_class_dir
                / image_path.name
            )

            cropped.save(
                output_path,
                quality=95
            )

            processed += 1
            counts[class_name] += 1

        except Exception as e:

            failed += 1

            print(
                f"FAILED: "
                f"{image_path.name} -> {e}"
            )


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print()
print("=" * 60)
print("PADDED CROP SUMMARY")
print("=" * 60)

print(
    f"Input images  : {processed + failed}"
)

print(
    f"Crops created : {processed}"
)

print(
    f"Failed        : {failed}"
)

print()
print("Crop counts:")

for class_name in CLASS_NAMES:

    print(
        f"  {class_name:<10}: "
        f"{counts[class_name]}"
    )

print()
print("Expected total: 36")
print()
print("Output directory:")
print(OUTPUT_DIR)
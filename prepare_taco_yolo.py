
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

# Project root: run this script from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent
TACO_ROOT = PROJECT_ROOT / "data" / "raw" / "taco" / "data"
ANNOTATIONS_FILE = TACO_ROOT / "annotations.json"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "taco_yolo"

SEED = 42
CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

# Practical first-pass mapping from TACO's 60 categories
# to Nirmaan AI's six waste classes.
MAPPING = {
    "cardboard": {
        "Other carton", "Egg carton", "Drink carton",
        "Corrugated carton", "Meal carton", "Pizza box",
    },
    "glass": {
        "Glass bottle", "Broken glass", "Glass cup", "Glass jar",
    },
    "metal": {
        "Aluminium foil", "Aluminium blister pack",
        "Metal bottle cap", "Food Can", "Aerosol",
        "Drink can", "Metal lid", "Pop tab", "Scrap metal",
    },
    "paper": {
        "Toilet tube", "Paper cup", "Magazine paper", "Tissues",
        "Wrapping paper", "Normal paper", "Paper bag",
        "Plastified paper bag", "Paper straw",
    },
    "plastic": {
        "Other plastic bottle", "Clear plastic bottle",
        "Plastic bottle cap", "Disposable plastic cup",
        "Foam cup", "Other plastic cup", "Plastic lid",
        "Other plastic", "Plastic film", "Six pack rings",
        "Garbage bag", "Other plastic wrapper",
        "Single-use carrier bag", "Polypropylene bag",
        "Crisp packet", "Spread tub", "Tupperware",
        "Disposable food container", "Foam food container",
        "Other plastic container", "Plastic glooves",
        "Plastic utensils", "Squeezable tube",
        "Plastic straw", "Styrofoam piece",
    },
    "trash": {"Unlabeled litter", "Cigarette"},
}

CATEGORY_TO_CLASS = {
    category: CLASS_TO_ID[class_name]
    for class_name, categories in MAPPING.items()
    for category in categories
}


def main():
    if not ANNOTATIONS_FILE.is_file():
        raise FileNotFoundError(
            f"Annotations not found: {ANNOTATIONS_FILE}"
        )

    # Avoid accidentally mixing a new dataset with an old conversion.
    if OUTPUT_ROOT.exists():
        raise FileExistsError(
            f"\nOutput folder already exists:\n{OUTPUT_ROOT}\n"
            "Rename or remove that generated folder before rerunning."
        )

    with ANNOTATIONS_FILE.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    category_names = {
        item["id"]: item["name"]
        for item in coco["categories"]
    }

    annotations_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        annotations_by_image[ann["image_id"]].append(ann)

    records = []
    missing_images = []
    invalid_boxes = 0
    unmapped_annotations = 0
    object_counts = Counter()

    for image_info in coco["images"]:
        source = TACO_ROOT / image_info["file_name"]

        # Never substitute a same-named image from another batch.
        if not source.is_file():
            missing_images.append(image_info["file_name"])
            continue

        try:
            with Image.open(source) as im:
                width, height = im.size
                im.verify()
        except (OSError, ValueError):
            missing_images.append(image_info["file_name"])
            continue

        if width <= 0 or height <= 0:
            continue

        labels = []
        image_counts = Counter()

        for ann in annotations_by_image[image_info["id"]]:
            category_name = category_names.get(ann["category_id"])
            class_id = CATEGORY_TO_CLASS.get(category_name)

            if class_id is None:
                unmapped_annotations += 1
                continue

            bbox = ann.get("bbox")
            if not bbox or len(bbox) != 4:
                invalid_boxes += 1
                continue

            x, y, w, h = map(float, bbox)

            # Convert COCO xywh coordinates into clipped corners.
            x1 = max(0.0, min(float(width), x))
            y1 = max(0.0, min(float(height), y))
            x2 = max(0.0, min(float(width), x + w))
            y2 = max(0.0, min(float(height), y + h))

            if x2 <= x1 or y2 <= y1:
                invalid_boxes += 1
                continue

            # YOLO format: class x_center y_center box_width box_height
            xc = ((x1 + x2) / 2.0) / width
            yc = ((y1 + y2) / 2.0) / height
            bw = (x2 - x1) / width
            bh = (y2 - y1) / height

            labels.append(
                f"{class_id} {xc:.6f} {yc:.6f} "
                f"{bw:.6f} {bh:.6f}"
            )
            image_counts[CLASS_NAMES[class_id]] += 1

        # Include only images that contain at least one mapped object.
        if labels:
            records.append({
                "image_id": image_info["id"],
                "source": source,
                "labels": labels,
                "counts": image_counts,
            })
            object_counts.update(image_counts)

    if len(records) < 10:
        raise RuntimeError(
            f"Only {len(records)} usable images found. "
            "Check the annotation paths and category mapping."
        )

    # Split by image, not by individual bounding box.
    random.Random(SEED).shuffle(records)
    n = len(records)
    train_end = int(n * 0.80)
    val_end = int(n * 0.90)

    splits = {
        "train": records[:train_end],
        "val": records[train_end:val_end],
        "test": records[val_end:],
    }

    for split, split_records in splits.items():
        image_dir = OUTPUT_ROOT / "images" / split
        label_dir = OUTPUT_ROOT / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        for record in split_records:
            source = record["source"]
            # Unique name avoids collisions between TACO batches.
            stem = (
                f"img_{record['image_id']:06d}_{source.stem}"
            )
            destination = image_dir / f"{stem}{source.suffix.lower()}"

            shutil.copy2(source, destination)
            (label_dir / f"{stem}.txt").write_text(
                "\n".join(record["labels"]) + "\n",
                encoding="utf-8",
            )

    dataset_path = OUTPUT_ROOT.as_posix()
    yaml_lines = [
        f'path: "{dataset_path}"',
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        f"nc: {len(CLASS_NAMES)}",
        "names:",
    ]
    for class_id, name in enumerate(CLASS_NAMES):
        yaml_lines.append(f"  {class_id}: {name}")

    (OUTPUT_ROOT / "data.yaml").write_text(
        "\n".join(yaml_lines) + "\n",
        encoding="utf-8",
    )

    print("\nTACO YOLO dataset preparation complete")
    print(f"Output: {OUTPUT_ROOT}")
    print(f"Usable images: {n}")
    print(f"Missing/unreadable images: {len(missing_images)}")
    print(f"Unmapped annotations skipped: {unmapped_annotations}")
    print(f"Invalid bounding boxes skipped: {invalid_boxes}")

    print("\nImage split:")
    for split, split_records in splits.items():
        print(f"  {split}: {len(split_records)} images")

    print("\nMapped object counts:")
    for name in CLASS_NAMES:
        print(f"  {name}: {object_counts[name]}")

    if missing_images:
        missing_file = OUTPUT_ROOT / "missing_images.txt"
        missing_file.write_text(
            "\n".join(missing_images) + "\n",
            encoding="utf-8",
        )
        print(f"\nMissing image list: {missing_file}")

    print("\nDataset YAML:")
    print((OUTPUT_ROOT / "data.yaml").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
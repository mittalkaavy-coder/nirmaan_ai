import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ANNOTATIONS_FILE = (
    PROJECT_ROOT / "data" / "raw" / "taco" / "data" / "annotations.json"
)

TEST_DIR = (
    PROJECT_ROOT / "data" / "processed" / "messy_split" / "test"
)

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


def get_nirmaan_class(category_names):
    mapped_classes = set()

    for class_name, categories in TACO_MAPPING.items():
        if any(category in categories for category in category_names):
            mapped_classes.add(class_name)

    return mapped_classes


with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

categories = {
    category["id"]: category["name"]
    for category in data["categories"]
}

images = {
    image["id"]: image
    for image in data["images"]
}

annotations_by_image = {}

for annotation in data["annotations"]:
    image_id = annotation["image_id"]
    category_id = annotation["category_id"]

    annotations_by_image.setdefault(image_id, set()).add(
        categories[category_id]
    )


print("=" * 80)
print("HELD-OUT TACO IMAGE ANNOTATION CHECK")
print("=" * 80)

total = 0
valid_single_class = 0
mismatch = 0

for class_dir in sorted(TEST_DIR.iterdir()):
    if not class_dir.is_dir():
        continue

    assigned_class = class_dir.name

    for image_file in sorted(class_dir.iterdir()):
        if not image_file.is_file():
            continue

        match = re.search(r"taco_(\d+)_", image_file.name)

        if not match:
            print(f"\nCould not extract ID: {image_file.name}")
            continue

        image_id = int(match.group(1))

        category_names = sorted(
            annotations_by_image.get(image_id, set())
        )

        mapped_classes = get_nirmaan_class(category_names)

        is_single = len(mapped_classes) == 1
        correct = mapped_classes == {assigned_class}

        total += 1

        if is_single:
            valid_single_class += 1

        if not correct:
            mismatch += 1

        print(f"\nImage: {image_file.name}")
        print(f"TACO ID: {image_id}")
        print(f"Our label: {assigned_class}")
        print(f"TACO categories: {category_names}")
        print(f"Mapped classes: {sorted(mapped_classes)}")
        print(f"Single mapped class: {is_single}")
        print(f"Label matches mapping: {correct}")


print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"Total images checked: {total}")
print(f"Single mapped class: {valid_single_class}")
print(f"Mapping mismatches: {mismatch}")
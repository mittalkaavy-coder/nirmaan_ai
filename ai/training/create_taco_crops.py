import csv
import json
from pathlib import Path
from PIL import Image


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ANNOTATIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "taco"
    / "data"
    / "annotations.json"
)

SELECTION_FILE = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "taco_selection.csv"
)

SOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "messy"
)

HELDOUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "messy_split"
    / "test"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "messy_crops"
)


# =========================================================
# TACO -> NIRMAAN CLASS MAPPING
# =========================================================

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


# =========================================================
# LOAD TACO ANNOTATIONS
# =========================================================

print("=" * 60)
print("NIRMAAN AI - TACO OBJECT CROP PREPARATION")
print("=" * 60)

print("\nLoading TACO annotations...")

with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as file:
    taco = json.load(file)

print(f"Images      : {len(taco['images'])}")
print(f"Annotations : {len(taco['annotations'])}")
print(f"Categories  : {len(taco['categories'])}")


# =========================================================
# CATEGORY LOOKUPS
# =========================================================

category_id_to_name = {
    category["id"]: category["name"]
    for category in taco["categories"]
}

category_to_class = {}

for nirmaan_class, taco_categories in TACO_MAPPING.items():
    for category_name in taco_categories:
        category_to_class[category_name] = nirmaan_class


image_lookup = {
    image["id"]: image
    for image in taco["images"]
}


# =========================================================
# LOAD SELECTION CSV
# =========================================================

print("\nLoading TACO selection metadata...")

selection_lookup = {}

with open(SELECTION_FILE, "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        image_id = int(row["image_id"])

        selection_lookup[image_id] = {
            "class": row["assigned_class"],
            "taco_filename": row["taco_filename"],
        }


# =========================================================
# FIND HELD-OUT IMAGES
# =========================================================

print("\nFinding held-out images...")

heldout_files = []

for class_dir in HELDOUT_DIR.iterdir():

    if not class_dir.is_dir():
        continue

    for image_path in class_dir.iterdir():

        if image_path.is_file():
            heldout_files.append(
                (
                    class_dir.name,
                    image_path
                )
            )


print(f"Held-out images found: {len(heldout_files)}")


# =========================================================
# PREPARE OUTPUT
# =========================================================

if OUTPUT_DIR.exists():

    import shutil

    shutil.rmtree(OUTPUT_DIR)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


for class_name in TACO_MAPPING:

    (
        OUTPUT_DIR / class_name
    ).mkdir(
        parents=True,
        exist_ok=True
    )


# =========================================================
# CREATE CROPS
# =========================================================

successful = 0
failed = 0

print("\nCreating object crops...\n")


for expected_class, image_path in heldout_files:

    filename = image_path.name

    # Expected filename:
    # taco_<image_id>_<original_filename>

    parts = filename.split("_", 2)

    if len(parts) < 3:
        print(f"[SKIP] Cannot determine image ID: {filename}")
        failed += 1
        continue

    try:
        image_id = int(parts[1])
    except ValueError:
        print(f"[SKIP] Invalid image ID: {filename}")
        failed += 1
        continue


    # -----------------------------------------------------
    # Check metadata
    # -----------------------------------------------------

    if image_id not in selection_lookup:

        print(f"[SKIP] No selection metadata: {filename}")
        failed += 1
        continue


    assigned_class = selection_lookup[image_id]["class"]

    if assigned_class != expected_class:

        print(
            f"[SKIP] Class mismatch: "
            f"{filename} -> "
            f"{expected_class} / {assigned_class}"
        )

        failed += 1
        continue


    # -----------------------------------------------------
    # Find relevant annotations
    # -----------------------------------------------------

    relevant_boxes = []

    for annotation in taco["annotations"]:

        if annotation["image_id"] != image_id:
            continue

        category_name = category_id_to_name[
            annotation["category_id"]
        ]

        mapped_class = category_to_class.get(
            category_name
        )

        if mapped_class != assigned_class:
            continue

        bbox = annotation.get("bbox")

        if not bbox:
            continue

        relevant_boxes.append(bbox)


    if not relevant_boxes:

        print(
            f"[SKIP] No mapped bounding box: "
            f"{filename}"
        )

        failed += 1
        continue


    # -----------------------------------------------------
    # Open image
    # -----------------------------------------------------

    try:

        image = Image.open(image_path).convert("RGB")

    except Exception as error:

        print(
            f"[SKIP] Cannot open {filename}: {error}"
        )

        failed += 1
        continue


    width, height = image.size

    # -----------------------------------------------------
    # TACO annotations use the ORIGINAL image dimensions.
    # The downloaded Flickr image may be resized.
    # Therefore scale the bounding boxes accordingly.
    # -----------------------------------------------------

    original_width = image_lookup[image_id]["width"]
    original_height = image_lookup[image_id]["height"]

    scale_x = width / original_width
    scale_y = height / original_height

    print(
        f"    Original size : "
        f"{original_width}x{original_height}"
    )

    print(
        f"    Download size : "
        f"{width}x{height}"
    )

    print(
        f"    Scale         : "
        f"{scale_x:.4f}, {scale_y:.4f}"
    )


    # -----------------------------------------------------
    # Combine all relevant boxes
    # -----------------------------------------------------

    x1 = min(
        box[0]
        for box in relevant_boxes
    )

    y1 = min(
        box[1]
        for box in relevant_boxes
    )

    x2 = max(
        box[0] + box[2]
        for box in relevant_boxes
    )

    y2 = max(
        box[1] + box[3]
        for box in relevant_boxes
    )


    # -----------------------------------------------------
    # Scale bounding box to downloaded image dimensions
    # -----------------------------------------------------

    x1 = x1 * scale_x
    y1 = y1 * scale_y
    x2 = x2 * scale_x
    y2 = y2 * scale_y


    # -----------------------------------------------------
    # Clip to image boundaries
    # -----------------------------------------------------

    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(width, int(x2))
    y2 = min(height, int(y2))


    if x2 <= x1 or y2 <= y1:

        print(
            f"[SKIP] Invalid crop: {filename}"
        )

        failed += 1
        continue


    # -----------------------------------------------------
    # Crop
    # -----------------------------------------------------

    crop = image.crop(
        (
            x1,
            y1,
            x2,
            y2
        )
    )


    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    output_path = (
        OUTPUT_DIR
        / assigned_class
        / filename
    )

    crop.save(
        output_path,
        quality=95
    )

    successful += 1

    print(
        f"[OK] {assigned_class:10s} "
        f"{filename}"
    )


# =========================================================
# FINAL REPORT
# =========================================================

print("\n" + "=" * 60)
print("TACO OBJECT CROP PREPARATION COMPLETE")
print("=" * 60)

print(f"Input images  : {len(heldout_files)}")
print(f"Crops created : {successful}")
print(f"Failed        : {failed}")

print("\nCrop counts:")

for class_name in TACO_MAPPING:

    class_dir = OUTPUT_DIR / class_name

    count = len(
        list(class_dir.glob("*"))
    )

    print(
        f"  {class_name:10s}: {count}"
    )

print("\nOutput directory:")
print(OUTPUT_DIR)
import csv
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from urllib.request import Request, urlopen

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ANNOTATIONS_FILE = (
    PROJECT_ROOT / "data" / "raw" / "taco" / "data" / "annotations.json"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "messy"

SELECTION_FILE = (
    PROJECT_ROOT / "experiments" / "results" / "taco_selection.csv"
)

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

IMAGES_PER_CLASS = 18
RANDOM_SEED = 42

# ---------------------------------------------------------
# TACO -> Nirmaan AI class mapping
# Only categories with >=20 annotations and a clear mapping
# are included.
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Download helper
# ---------------------------------------------------------

def download_image(url, destination):
    """Download one image from TACO."""

    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0"
        },
    )

    try:
        with urlopen(request, timeout=30) as response:
            data = response.read()

        destination.write_bytes(data)
        return True

    except Exception as error:
        print(f"    Download failed: {error}")
        return False


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=" * 60)
    print("NIRMAAN AI - TACO MESSY DATASET PREPARATION")
    print("=" * 60)

    print("\nLoading TACO annotations...")

    with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as file:
        taco = json.load(file)

    print(f"Images in TACO: {len(taco['images'])}")
    print(f"Annotations: {len(taco['annotations'])}")
    print(f"Categories: {len(taco['categories'])}")

    # -----------------------------------------------------
    # Category lookup
    # -----------------------------------------------------

    category_id_to_name = {
        category["id"]: category["name"]
        for category in taco["categories"]
    }

    # -----------------------------------------------------
    # Convert category names -> Nirmaan class
    # -----------------------------------------------------

    category_to_class = {}

    for nirmaan_class, taco_categories in TACO_MAPPING.items():

        for category_name in taco_categories:
            category_to_class[category_name] = nirmaan_class

    # -----------------------------------------------------
    # Find classes present in each image
    # -----------------------------------------------------

    image_classes = defaultdict(set)

    image_categories = defaultdict(set)

    for annotation in taco["annotations"]:

        category_name = category_id_to_name[
            annotation["category_id"]
        ]

        if category_name not in category_to_class:
            continue

        nirmaan_class = category_to_class[category_name]

        image_id = annotation["image_id"]

        image_classes[image_id].add(nirmaan_class)
        image_categories[image_id].add(category_name)

    # -----------------------------------------------------
    # Keep ONLY images containing exactly one mapped class
    # -----------------------------------------------------

    single_class_images = defaultdict(list)

    image_lookup = {
        image["id"]: image
        for image in taco["images"]
    }

    for image_id, classes in image_classes.items():

        if len(classes) != 1:
            continue

        nirmaan_class = next(iter(classes))

        single_class_images[nirmaan_class].append(image_id)

    print("\nSingle-class TACO images:")

    for class_name in TACO_MAPPING:

        count = len(single_class_images[class_name])

        print(f"  {class_name:10s}: {count}")

    # -----------------------------------------------------
    # Check that every class has enough images
    # -----------------------------------------------------

    for class_name in TACO_MAPPING:

        available = len(single_class_images[class_name])

        if available < IMAGES_PER_CLASS:

            raise RuntimeError(
                f"Not enough images for {class_name}: "
                f"{available} available, "
                f"{IMAGES_PER_CLASS} required."
            )

    # -----------------------------------------------------
    # Reproducible sampling
    # -----------------------------------------------------

    random.seed(RANDOM_SEED)

    selected_images = {}

    for class_name in TACO_MAPPING:

        candidates = single_class_images[class_name].copy()

        random.shuffle(candidates)

        selected_images[class_name] = candidates[:IMAGES_PER_CLASS]

    # -----------------------------------------------------
    # Prepare output directories
    # -----------------------------------------------------

    print("\nPreparing output directories...")

    for class_name in TACO_MAPPING:

        class_dir = OUTPUT_DIR / class_name

        class_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    SELECTION_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # -----------------------------------------------------
    # Download selected images
    # -----------------------------------------------------

    print("\nDownloading selected TACO images...")

    selection_rows = []

    total = 0
    successful = 0
    failed = 0

    for class_name in TACO_MAPPING:

        print(f"\n[{class_name.upper()}]")

        for image_id in selected_images[class_name]:

            image_info = image_lookup[image_id]

            filename = Path(
                image_info["file_name"]
            ).name

            output_path = (
                OUTPUT_DIR
                / class_name
                / f"taco_{image_id}_{filename}"
            )

            # Prefer 640px Flickr URL when available.
            url = image_info.get("flickr_640_url")

            if not url:
                url = image_info.get("flickr_url")

            if not url:

                print(
                    f"  [{image_id}] No download URL"
                )

                failed += 1
                continue

            print(
                f"  [{image_id}] {filename}"
            )

            success = download_image(
                url,
                output_path
            )

            total += 1

            if success:

                successful += 1

                source_categories = sorted(
                    image_categories[image_id]
                )

                selection_rows.append(
                    {
                        "image_id": image_id,
                        "taco_filename": image_info["file_name"],
                        "assigned_class": class_name,
                        "source_categories": ";".join(
                            source_categories
                        ),
                        "download_url": url,
                    }
                )

            else:

                failed += 1

            # Small delay to avoid aggressive requests.
            time.sleep(0.2)

    # -----------------------------------------------------
    # Save selection CSV
    # -----------------------------------------------------

    print("\nSaving selection metadata...")

    with open(
        SELECTION_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        fieldnames = [
            "image_id",
            "taco_filename",
            "assigned_class",
            "source_categories",
            "download_url",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(selection_rows)

    # -----------------------------------------------------
    # Final report
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("TACO DATASET PREPARATION COMPLETE")
    print("=" * 60)

    print(f"Requested images : {IMAGES_PER_CLASS * 6}")
    print(f"Download attempts: {total}")
    print(f"Successful       : {successful}")
    print(f"Failed           : {failed}")

    print("\nDataset location:")
    print(OUTPUT_DIR)

    print("\nSelection metadata:")
    print(SELECTION_FILE)

    print("\nClass counts downloaded:")

    for class_name in TACO_MAPPING:

        class_dir = OUTPUT_DIR / class_name

        count = len(
            list(class_dir.glob("*"))
        )

        print(
            f"  {class_name:10s}: {count}"
        )


if __name__ == "__main__":
    main()
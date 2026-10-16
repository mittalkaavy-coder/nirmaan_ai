from pathlib import Path
import shutil
import random
from PIL import Image

# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "trashnet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "clean"

CLASSES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# --------------------------------------------------
# Utility functions
# --------------------------------------------------

def is_valid_image(image_path: Path) -> bool:
    """Check whether an image can be opened successfully."""

    try:
        with Image.open(image_path) as image:
            image.verify()
        return True

    except Exception:
        return False


def split_files(files):
    """Create reproducible train/validation/test split."""

    files = list(files)

    random.Random(SEED).shuffle(files)

    total = len(files)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]

    return train_files, val_files, test_files


def copy_files(files, destination):
    """Copy files into the destination directory."""

    destination.mkdir(parents=True, exist_ok=True)

    for source in files:
        shutil.copy2(source, destination / source.name)


# --------------------------------------------------
# Main preprocessing
# --------------------------------------------------

def main():

    print("=" * 60)
    print("TrashNet Clean Dataset Preparation")
    print("=" * 60)

    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"TrashNet dataset not found at: {RAW_DIR}"
        )

    print(f"Source: {RAW_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    # Remove previous processed dataset
    if OUTPUT_DIR.exists():
        print("Removing previous processed clean dataset...")
        shutil.rmtree(OUTPUT_DIR)

    total_images = 0
    total_valid = 0

    for class_name in CLASSES:

        class_dir = RAW_DIR / class_name

        if not class_dir.exists():
            print(f"WARNING: Missing class: {class_name}")
            continue

        image_files = [
            file
            for file in class_dir.iterdir()
            if file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        ]

        valid_files = []

        for image_file in image_files:

            total_images += 1

            if is_valid_image(image_file):
                valid_files.append(image_file)
                total_valid += 1
            else:
                print(f"Invalid image skipped: {image_file}")

        train_files, val_files, test_files = split_files(valid_files)

        copy_files(
            train_files,
            OUTPUT_DIR / "train" / class_name
        )

        copy_files(
            val_files,
            OUTPUT_DIR / "val" / class_name
        )

        copy_files(
            test_files,
            OUTPUT_DIR / "test" / class_name
        )

        print(
            f"{class_name:10s} | "
            f"total={len(valid_files):3d} | "
            f"train={len(train_files):3d} | "
            f"val={len(val_files):3d} | "
            f"test={len(test_files):3d}"
        )

    print()
    print("=" * 60)
    print("Dataset preparation complete")
    print("=" * 60)

    print(f"Total images found : {total_images}")
    print(f"Valid images       : {total_valid}")

    print()
    print("Output structure:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
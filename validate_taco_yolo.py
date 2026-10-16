
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent / "data" / "processed" / "taco_yolo"
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

errors = []
class_counts = Counter()
total_boxes = 0

for split in ["train", "val", "test"]:
    image_dir = ROOT / "images" / split
    label_dir = ROOT / "labels" / split

    images = [
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ]
    labels = list(label_dir.glob("*.txt"))

    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}

    for stem in image_stems - label_stems:
        errors.append(f"{split}: missing label for {stem}")
    for stem in label_stems - image_stems:
        errors.append(f"{split}: label without image for {stem}")

    for label_file in labels:
        for line_number, line in enumerate(
            label_file.read_text(encoding="utf-8").splitlines(), start=1
        ):
            parts = line.split()

            if len(parts) != 5:
                errors.append(f"{label_file.name}:{line_number}: expected 5 values")
                continue

            try:
                class_id = int(parts[0])
                x, y, w, h = map(float, parts[1:])
            except ValueError:
                errors.append(f"{label_file.name}:{line_number}: non-numeric value")
                continue

            if not 0 <= class_id < len(CLASS_NAMES):
                errors.append(f"{label_file.name}:{line_number}: invalid class ID")
                continue

            if not all(0 <= v <= 1 for v in (x, y, w, h)):
                errors.append(f"{label_file.name}:{line_number}: values outside 0..1")
                continue

            if w <= 0 or h <= 0:
                errors.append(f"{label_file.name}:{line_number}: zero-sized box")
                continue

            class_counts[class_id] += 1
            total_boxes += 1

    print(
        f"{split}: {len(images)} images, {len(labels)} label files"
    )

print(f"\nTotal valid boxes: {total_boxes}")
print("\nClass counts:")
for class_id, name in enumerate(CLASS_NAMES):
    print(f"  {name}: {class_counts[class_id]}")

print(f"\nValidation errors: {len(errors)}")
for error in errors[:20]:
    print(" -", error)

if errors:
    raise SystemExit("Validation failed. Do not train yet.")

print("\nPASS: image-label pairing and basic YOLO label checks succeeded.")
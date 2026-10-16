from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# NIRMAAN AI - PADDED CROP VISUAL CHECK
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "messy_padded_crops"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "messy_padded_crop_grid.png"
)

CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
]


print("=" * 60)
print("NIRMAAN AI - PADDED CROP VISUAL CHECK")
print("=" * 60)


images = []

for class_name in CLASS_NAMES:

    class_dir = INPUT_DIR / class_name

    files = sorted(
        [
            p
            for p in class_dir.iterdir()
            if p.suffix.lower()
            in {".jpg", ".jpeg", ".png", ".webp"}
        ]
    )

    for image_path in files:

        image = Image.open(image_path).convert("RGB")

        images.append(
            (
                class_name,
                image_path.name,
                image
            )
        )


print(f"Images loaded: {len(images)}")


# ------------------------------------------------------------
# Create 6 x 6 grid
# ------------------------------------------------------------

fig, axes = plt.subplots(
    6,
    6,
    figsize=(18, 18)
)

for ax, (class_name, filename, image) in zip(
    axes.flat,
    images
):

    ax.imshow(image)

    ax.set_title(
        f"{class_name}\n{filename}",
        fontsize=8
    )

    ax.axis("off")


# Hide unused axes, if any

for ax in axes.flat[len(images):]:
    ax.axis("off")


plt.tight_layout()

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

plt.savefig(
    OUTPUT_PATH,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


print()
print("Saved padded crop grid to:")
print(OUTPUT_PATH)
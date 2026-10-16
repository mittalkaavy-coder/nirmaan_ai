from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
import math


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CROP_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "messy_crops"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "messy_crop_grid.png"
)


# =========================================================
# SETTINGS
# =========================================================

CLASSES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
]

GRID_COLUMNS = 6
CELL_WIDTH = 240
CELL_HEIGHT = 260
PADDING = 10


# =========================================================
# LOAD IMAGES
# =========================================================

images = []

for class_name in CLASSES:

    class_dir = CROP_DIR / class_name

    files = sorted(
        [
            path
            for path in class_dir.iterdir()
            if path.is_file()
        ]
    )

    for image_path in files:

        try:
            image = Image.open(image_path).convert("RGB")

            images.append(
                (
                    class_name,
                    image,
                    image_path.name
                )
            )

        except Exception as error:

            print(
                f"Could not open {image_path}: {error}"
            )


print("=" * 60)
print("NIRMAAN AI - TACO CROP VISUAL CHECK")
print("=" * 60)

print(f"Images loaded: {len(images)}")


# =========================================================
# GRID SIZE
# =========================================================

rows = math.ceil(
    len(images) / GRID_COLUMNS
)

grid_width = (
    GRID_COLUMNS * CELL_WIDTH
    + (GRID_COLUMNS + 1) * PADDING
)

grid_height = (
    rows * CELL_HEIGHT
    + (rows + 1) * PADDING
)


canvas = Image.new(
    "RGB",
    (grid_width, grid_height),
    "white"
)


# =========================================================
# DRAW GRID
# =========================================================

draw = ImageDraw.Draw(canvas)

for index, (class_name, image, filename) in enumerate(images):

    row = index // GRID_COLUMNS
    column = index % GRID_COLUMNS

    x = (
        PADDING
        + column * CELL_WIDTH
    )

    y = (
        PADDING
        + row * CELL_HEIGHT
    )

    # ---------------------------------------------
    # Image area
    # ---------------------------------------------

    image_area_width = CELL_WIDTH - 2 * PADDING
    image_area_height = 205

    fitted = ImageOps.contain(
        image,
        (
            image_area_width,
            image_area_height
        )
    )

    image_x = (
        x
        + (CELL_WIDTH - fitted.width) // 2
    )

    image_y = y + 5

    canvas.paste(
        fitted,
        (image_x, image_y)
    )

    # ---------------------------------------------
    # Text
    # ---------------------------------------------

    text_y = y + 215

    draw.text(
        (
            x + 5,
            text_y
        ),
        f"{class_name}",
        fill="black"
    )

    draw.text(
        (
            x + 5,
            text_y + 18
        ),
        filename,
        fill="black"
    )


# =========================================================
# SAVE
# =========================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

canvas.save(
    OUTPUT_PATH,
    quality=95
)

print("\nSaved crop grid to:")
print(OUTPUT_PATH)
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MAMMOGPS = DATA / "mammogps" / "data" / "benchmark"
SOURCES = DATA / "sources"
LABELS_CSV = DATA / "labels.csv"
YOLO_DIR = DATA / "yolo"
MODELS = ROOT / "models"

# Detector classes; rare finding types are merged into "other".
FINDING_CLASSES = ["mass", "calcifications", "asymmetry", "architectural_distortion", "other"]

# Finding BI-RADS classes available in VinDr-Mammo annotations.
BIRADS_CLASSES = ["3", "4", "5"]

# Converted VinDr PNGs are stored at this height (breast-cropped); detection runs at IMAGE_HEIGHT.
STORE_HEIGHT = 1536
IMAGE_HEIGHT = 1280
VINDR = DATA / "vindr"


def birads_number(value) -> int | None:
    """Parse a BI-RADS value: 4, '4a', 'BI-RADS 4' -> 4."""
    m = re.search(r"\d", str(value)) if value is not None else None
    return int(m.group()) if m else None


def birads_to_index(value) -> int | None:
    """Map a VinDr finding BI-RADS value to an index in BIRADS_CLASSES."""
    n = birads_number(value)
    if n is None:
        return None
    return {3: 0, 4: 1, 5: 2}.get(n)


def finding_class(finding_type: str) -> str:
    return finding_type if finding_type in FINDING_CLASSES else "other"

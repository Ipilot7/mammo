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

# BI-RADS buckets predicted by the classifier.
BIRADS_CLASSES = ["0", "1-2", "3", "4", "5", "6"]

# Converted VinDr PNGs are stored at this height (breast-cropped); detection runs at IMAGE_HEIGHT.
STORE_HEIGHT = 1536
IMAGE_HEIGHT = 1280
VINDR = DATA / "vindr"


def birads_number(value) -> int | None:
    """Parse a BI-RADS value: 4, '4a', 'BI-RADS 4' -> 4."""
    m = re.search(r"\d", str(value)) if value is not None else None
    return int(m.group()) if m else None


def birads_to_index(value) -> int | None:
    """Map a raw BI-RADS value (0..6, '4a', 'BI-RADS 4', ...) to an index in BIRADS_CLASSES."""
    n = birads_number(value)
    return {0: 0, 1: 1, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}.get(n)


def finding_class(finding_type: str) -> str:
    return finding_type if finding_type in FINDING_CLASSES else "other"

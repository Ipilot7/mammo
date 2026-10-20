"""Lesion crops shared by classifier training and inference."""
import random

from PIL import Image

CROP_SIZE = 384
CONTEXT = 1.5  # crop side = max(bbox side) * CONTEXT
MIN_SIDE_FRAC = 0.1  # of image height: small calcification clusters still get context


def crop_lesion(img: Image.Image, box, jitter: float = 0.0) -> Image.Image:
    """Square crop around box (x1, y1, x2, y2) with context, resized to CROP_SIZE.

    jitter > 0 randomly shifts/scales the crop (training-time augmentation that also
    mimics imperfect detector boxes).
    """
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1, MIN_SIDE_FRAC * img.height) * CONTEXT
    if jitter:
        side *= random.uniform(1 - jitter, 1 + jitter)
        cx += random.uniform(-jitter, jitter) * side / 2
        cy += random.uniform(-jitter, jitter) * side / 2
    half = side / 2
    region = (int(cx - half), int(cy - half), int(cx + half), int(cy + half))
    # PIL pads out-of-bounds regions with black, which matches the mammogram background.
    return img.crop(region).resize((CROP_SIZE, CROP_SIZE), Image.BILINEAR)

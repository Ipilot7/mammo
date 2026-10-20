"""Shared image preprocessing: grayscale conversion and cropping to the breast region.

Every image (training sources and uploads) goes through crop_breast() so that sources with a
large black background (KAU-BCMD, DMID, raw DICOM) look like the tightly framed MammoGPS images.
Otherwise a model could tell sources apart by the amount of background instead of by pathology.
"""
import cv2
import numpy as np
from PIL import Image

from mammo.config import IMAGE_HEIGHT


def to_gray_u8(img: Image.Image) -> np.ndarray:
    """Any PIL image (L, I;16, RGB, RGBA) -> uint8 grayscale array."""
    if img.mode in ("I;16", "I;16B", "I", "F"):
        a = np.asarray(img, dtype=np.float32)
        a = (a - a.min()) / max(float(a.max() - a.min()), 1e-6)
        return (a * 255).astype(np.uint8)
    return np.asarray(img.convert("L"))


def breast_mask(gray: np.ndarray):
    """Mask of the largest bright component (the breast) on a <=512 px copy, and that copy's scale."""
    h, w = gray.shape
    small_scale = 512 / max(h, w)
    small = cv2.resize(gray, (max(1, int(w * small_scale)), max(1, int(h * small_scale))), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (5, 5), 0)
    # Everything clearly above the background is breast. Otsu would split dense from fatty tissue
    # and cut the fatty part off.
    lo, hi = np.percentile(small, 2), np.percentile(small, 99.5)
    mask = (small > lo + 0.06 * (hi - lo)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return None, small_scale
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == i).astype(np.uint8), small_scale


def breast_bbox(gray: np.ndarray, pad: float = 0.02) -> tuple[int, int, int, int]:
    """Bounding box (x1, y1, x2, y2) of the breast."""
    h, w = gray.shape
    mask, small_scale = breast_mask(gray)
    if mask is None:
        return 0, 0, w, h
    ys, xs = np.nonzero(mask)
    x, y = xs.min() / small_scale, ys.min() / small_scale
    bw, bh = (xs.max() + 1) / small_scale - x, (ys.max() + 1) / small_scale - y
    px, py = pad * w, pad * h
    return (int(max(0, x - px)), int(max(0, y - py)), int(min(w, x + bw + px)), int(min(h, y + bh + py)))


def breast_contour(gray: np.ndarray, max_points: int = 120) -> list[list[float]]:
    """Simplified outline of the breast as [[x, y], ...] in the input image's pixels."""
    mask, small_scale = breast_mask(gray)
    if mask is None:
        return []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(contours, key=cv2.contourArea)
    eps = 0.002 * cv2.arcLength(c, True)
    while True:
        poly = cv2.approxPolyDP(c, eps, True)
        if len(poly) <= max_points:
            break
        eps *= 1.5
    return [[float(x) / small_scale, float(y) / small_scale] for x, y in poly[:, 0]]


def crop_breast(gray: np.ndarray, height: int = IMAGE_HEIGHT):
    """Crop to the breast and resize to `height`.

    Returns (image, transform) where transform = (x0, y0, scale) maps output pixels back to the
    input: x_in = x_out / scale + x0.
    """
    x1, y1, x2, y2 = breast_bbox(gray)
    crop = gray[y1:y2, x1:x2]
    scale = height / crop.shape[0]
    out = cv2.resize(crop, (max(1, round(crop.shape[1] * scale)), height), interpolation=cv2.INTER_AREA)
    return Image.fromarray(out), (x1, y1, scale)


def dicom_to_gray(ds) -> np.ndarray:
    """DICOM dataset -> uint8 grayscale, using the stored VOI window like the radiologist's display.

    Follows the VinDr-Mammo export used by MammoGPS: rescale, linear window (WindowCenter/Width),
    MONOCHROME1 inversion, padding pixels set to black. Falls back to min-max without a window.
    """
    from pydicom.multival import MultiValue

    a = ds.pixel_array.astype(np.float32)
    a = a * float(getattr(ds, "RescaleSlope", 1) or 1) + float(getattr(ds, "RescaleIntercept", 0) or 0)
    pad = getattr(ds, "PixelPaddingValue", None)
    pad_mask = (a == float(pad)) if pad is not None else np.zeros(a.shape, bool)

    def first(v):  # WindowCenter/Width may hold several windows; use the default (first) one
        return float(v[0]) if isinstance(v, (list, tuple, MultiValue)) else float(v)

    center, width = getattr(ds, "WindowCenter", None), getattr(ds, "WindowWidth", None)
    if center is not None and width is not None:
        c, w = first(center), max(first(width), 2.0)
        a = np.clip(((a - (c - 0.5)) / (w - 1) + 0.5), 0, 1)
    else:
        valid = a[~pad_mask] if (~pad_mask).any() else a
        lo, hi = float(valid.min()), float(valid.max())
        a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
    if str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")) == "MONOCHROME1":
        a = 1.0 - a
    a[pad_mask] = 0
    return (a * 255).astype(np.uint8)

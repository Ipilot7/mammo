"""End-to-end inference for radiologist-derived BI-RADS proxy outputs."""
import hashlib
import hmac
import io
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms as T

from mammo.classifier import MEAN, STD, load_classifier
from mammo.config import BIRADS_CLASSES, IMAGE_HEIGHT, MODELS
from mammo.crops import crop_lesion
from mammo.preprocess import breast_contour, crop_breast, dicom_to_gray, to_gray_u8
from mammo.screening import (
    BREAST_BIRADS,
    load_screening,
    screening_image,
    suspicious_prob,
    to_tensor,
)

MAX_DECODED_PIXELS = 50_000_000
MAX_IMAGE_FRAMES = 1
ALLOWED_DICOM_TRANSFER_SYNTAXES = {
    "1.2.840.10008.1.2",    # Implicit VR Little Endian
    "1.2.840.10008.1.2.1",  # Explicit VR Little Endian
    "1.2.840.10008.1.2.2",  # Explicit VR Big Endian
}


def verify_detector_checkpoint(path: str | Path) -> str:
    """Verify the SHA-256 sidecar before loading an unsafe Ultralytics checkpoint."""
    path = Path(path)
    sidecar = Path(f"{path}.sha256")
    if not sidecar.is_file():
        raise FileNotFoundError(f"detector SHA-256 sidecar not found: {sidecar}")
    expected = sidecar.read_text().split(maxsplit=1)[0].lower()
    if re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise ValueError(f"invalid detector SHA-256 sidecar: {sidecar}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if not hmac.compare_digest(actual, expected):
        raise ValueError(f"detector SHA-256 mismatch: {path}")
    return actual


def write_detector_checkpoint_sidecar(path: str | Path) -> str:
    """Write a SHA-256 sidecar for a locally produced detector checkpoint."""
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    Path(f"{path}.sha256").write_text(f"{actual}  {path.name}\n")
    return actual

# Display level of a BI-RADS category: 1-2 lower concern, 0/3 follow-up, 4-6 higher concern.
BIRADS_LEVEL = {"0": "medium", "1": "normal", "1-2": "low", "2": "low", "3": "medium", "4": "high", "5": "high",
                "6": "high"}
# Finding BI-RADS bucket -> comparable breast category (for the overall verdict).
BUCKET_RANK = {"1-2": 2, "0": 3, "3": 3, "4": 4, "5": 5, "6": 5}


@dataclass
class Finding:
    bbox_xyxy: list[float]  # in original image pixels
    type: str
    type_conf: float
    p_birads_4_5: float
    birads: str
    birads_probs: dict[str, float]
    level: str


def validate_decoded_shape(width: int, height: int, frames: int = 1, samples: int = 1) -> None:
    """Bound decoded image allocation before loading pixel data."""
    if width <= 0 or height <= 0 or frames <= 0 or samples <= 0:
        raise ValueError("invalid image dimensions")
    if frames > MAX_IMAGE_FRAMES:
        raise ValueError(f"image frame limit exceeded: {frames} > {MAX_IMAGE_FRAMES}")
    if width * height * frames * samples > MAX_DECODED_PIXELS:
        raise ValueError("decoded pixel limit exceeded")


def validate_dicom_metadata(dataset) -> None:
    """Allow bounded, single-frame, uncompressed grayscale mammograms only."""
    transfer_syntax = str(getattr(getattr(dataset, "file_meta", None), "TransferSyntaxUID", ""))
    if transfer_syntax not in ALLOWED_DICOM_TRANSFER_SYNTAXES:
        raise ValueError(f"unsupported DICOM transfer syntax: {transfer_syntax or 'missing'}")
    samples = int(getattr(dataset, "SamplesPerPixel", 1))
    bits = int(getattr(dataset, "BitsAllocated", 0))
    photometric = str(getattr(dataset, "PhotometricInterpretation", ""))
    if photometric not in {"MONOCHROME1", "MONOCHROME2"}:
        raise ValueError(f"unsupported DICOM PhotometricInterpretation: {photometric or 'missing'}")
    if samples != 1 or bits not in {8, 16}:
        raise ValueError(f"unsupported DICOM pixel format: samples={samples}, bits={bits}")
    validate_decoded_shape(
        int(getattr(dataset, "Columns", 0)),
        int(getattr(dataset, "Rows", 0)),
        int(getattr(dataset, "NumberOfFrames", 1)),
        samples,
    )


def load_image(data: bytes) -> Image.Image:
    """Decode PNG/JPEG or DICOM bytes into an 8-bit grayscale PIL image."""
    if data[128:132] == b"DICM" or data[:4] == b"DICM":
        import pydicom

        dataset = pydicom.dcmread(io.BytesIO(data))
        validate_dicom_metadata(dataset)
        return Image.fromarray(dicom_to_gray(dataset))
    with Image.open(io.BytesIO(data)) as image:
        validate_decoded_shape(image.width, image.height, getattr(image, "n_frames", 1), len(image.getbands()))
        return image.convert("L")


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def screening_summary(probs: torch.Tensor, threshold: float) -> dict:
    """Describe the BI-RADS screening output without implying pathology-confirmed cancer risk."""
    p_suspicious = float(suspicious_prob(probs))
    birads = BREAST_BIRADS[int(probs.argmax())]
    return {
        "birads": birads,
        "birads_probs": {str(b): float(q) for b, q in zip(BREAST_BIRADS, probs)},
        "p_suspicious": p_suspicious,
        "p_birads_4_5": float(probs[3:].sum()),
        "threshold": threshold,
        "verdict": "suspicious" if p_suspicious >= threshold else "normal",
    }


class MammoPipeline:
    def __init__(self, detector_path=MODELS / "detector.pt", classifier_path=MODELS / "classifier.pt",
                 screening_path=MODELS / "screening.pt",
                 device: str | None = None, conf: float = 0.05, max_findings: int = 5):
        verify_detector_checkpoint(detector_path)
        from ultralytics import YOLO

        self.device = device or pick_device()
        self.detector = YOLO(str(detector_path))
        self.classifier = load_classifier(classifier_path, self.device)
        self.conf = conf
        self.max_findings = max_findings
        self.tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
        self.screening = None
        if Path(screening_path).exists():  # optional: older setups have no screening model
            self.screening, self.screening_threshold = load_screening(screening_path, self.device)

    @torch.no_grad()
    def screen(self, image: Image.Image) -> dict | None:
        """BI-RADS category of the whole breast (1-5) and the "needs attention" verdict."""
        if self.screening is None:
            return None
        x = to_tensor(screening_image(to_gray_u8(image))).unsqueeze(0).to(self.device)
        probs = torch.softmax(self.screening(x).float(), 1)[0].cpu()
        return screening_summary(probs, self.screening_threshold)

    @torch.no_grad()
    def analyze(self, image: Image.Image, conf: float | None = None) -> list[Finding]:
        # Same breast crop as the stored VinDr training images; boxes map back with (x0, y0, scale).
        small, (x0, y0, scale) = crop_breast(to_gray_u8(image), IMAGE_HEIGHT)
        small = small.convert("RGB")

        res = self.detector.predict(small, conf=conf if conf is not None else self.conf, imgsz=IMAGE_HEIGHT,
                                    max_det=self.max_findings, iou=0.45, agnostic_nms=True,
                                    device=self.device, verbose=False)[0]
        boxes = res.boxes
        if len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        crops = torch.stack([self.tf(crop_lesion(small, b)) for b in xyxy]).to(self.device)
        mal_logit, birads_logit = self.classifier(crops)
        mal = torch.sigmoid(mal_logit).cpu().numpy()
        birads = torch.softmax(birads_logit, 1).cpu().numpy()

        findings = []
        for i, box in enumerate(xyxy):
            cls = int(boxes.cls[i])
            findings.append(Finding(
                bbox_xyxy=[float(box[0] / scale + x0), float(box[1] / scale + y0),
                           float(box[2] / scale + x0), float(box[3] / scale + y0)],
                type=res.names[cls],
                type_conf=float(boxes.conf[i]),
                p_birads_4_5=float(mal[i]),
                birads=BIRADS_CLASSES[int(birads[i].argmax())],
                birads_probs={c: float(p) for c, p in zip(BIRADS_CLASSES, birads[i])},
                level=BIRADS_LEVEL[BIRADS_CLASSES[int(birads[i].argmax())]],
            ))
        findings.sort(key=lambda f: f.p_birads_4_5, reverse=True)
        return findings

    def analyze_bytes(self, data: bytes, conf: float | None = None) -> dict:
        image = load_image(data)
        screening = self.screen(image)
        findings = self.analyze(image, conf)
        # Overall BI-RADS: the more severe of the whole-breast category and the worst finding.
        ranks = [BUCKET_RANK[f.birads] for f in findings]
        if screening:
            ranks.append(screening["birads"])
        overall = max(ranks) if ranks else None
        return {
            "image": image,
            "findings": [asdict(f) for f in findings],
            "screening": screening,
            "breast_contour": breast_contour(to_gray_u8(image)),
            "overall_birads": overall,
            "overall_level": BIRADS_LEVEL[str(overall)] if overall else "none",
        }

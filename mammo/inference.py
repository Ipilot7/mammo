"""End-to-end inference: image file -> detected lesions with type, BI-RADS and malignancy level."""
import io
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from mammo.classifier import MEAN, STD, load_classifier
from mammo.config import BIRADS_CLASSES, IMAGE_HEIGHT, MODELS
from mammo.crops import crop_lesion
from mammo.preprocess import breast_contour, crop_breast, dicom_to_gray, to_gray_u8
from mammo.screening import BREAST_BIRADS, load_screening, screening_image, suspicious_prob, to_tensor

# Colour level of a BI-RADS category: 1-2 benign/normal, 0 and 3 follow-up, 4-6 suspicious for cancer.
BIRADS_LEVEL = {"0": "medium", "1": "normal", "1-2": "low", "2": "low", "3": "medium", "4": "high", "5": "high",
                "6": "high"}
# Finding BI-RADS bucket -> comparable breast category (for the overall verdict).
BUCKET_RANK = {"1-2": 2, "0": 3, "3": 3, "4": 4, "5": 5, "6": 5}


@dataclass
class Finding:
    bbox_xyxy: list[float]  # in original image pixels
    type: str
    type_conf: float
    malignancy_prob: float
    birads: str
    birads_probs: dict[str, float]
    level: str


def load_image(data: bytes) -> Image.Image:
    """Decode PNG/JPEG or DICOM bytes into an 8-bit grayscale PIL image."""
    if data[128:132] == b"DICM" or data[:4] == b"DICM":
        import pydicom

        return Image.fromarray(dicom_to_gray(pydicom.dcmread(io.BytesIO(data))))
    return Image.open(io.BytesIO(data)).convert("L")


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class MammoPipeline:
    def __init__(self, detector_path=MODELS / "detector.pt", classifier_path=MODELS / "classifier.pt",
                 screening_path=MODELS / "screening.pt",
                 device: str | None = None, conf: float = 0.05, max_findings: int = 5):
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
        p = float(suspicious_prob(probs))
        birads = BREAST_BIRADS[int(probs.argmax())]
        return {"birads": birads, "birads_probs": {str(b): float(q) for b, q in zip(BREAST_BIRADS, probs)},
                "p_suspicious": p, "p_cancer": float(probs[3:].sum()), "threshold": self.screening_threshold,
                "verdict": "suspicious" if p >= self.screening_threshold else "normal"}

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
                malignancy_prob=float(mal[i]),
                birads=BIRADS_CLASSES[int(birads[i].argmax())],
                birads_probs={c: float(p) for c, p in zip(BIRADS_CLASSES, birads[i])},
                level=BIRADS_LEVEL[BIRADS_CLASSES[int(birads[i].argmax())]],
            ))
        findings.sort(key=lambda f: f.malignancy_prob, reverse=True)
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

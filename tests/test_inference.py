import io
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from mammo import inference as inference_module
from mammo.classifier import load_classifier
from mammo.inference import Finding

ROOT = Path(__file__).resolve().parents[1]


def test_finding_probability_is_named_as_birads_proxy():
    finding = Finding(
        bbox_xyxy=[1.0, 2.0, 3.0, 4.0],
        type="mass",
        type_conf=0.9,
        p_birads_4_5=0.7,
        birads="4",
        birads_probs={"3": 0.1, "4": 0.7, "5": 0.2},
        level="high",
    )

    payload = asdict(finding)
    assert payload["p_birads_4_5"] == 0.7
    assert "malignancy_prob" not in payload


def test_classifier_loader_rejects_incompatible_birads_classes(tmp_path):
    checkpoint = tmp_path / "old-classifier.pt"
    torch.save(
        {
            "arch": "efficientnet_b3",
            "birads_classes": ["0", "1-2", "3", "4", "5", "6"],
            "state_dict": {},
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="BI-RADS classes"):
        load_classifier(checkpoint)


def test_viewer_uses_the_public_birads_proxy_fields():
    viewer = (ROOT / "app" / "static" / "viewer.js").read_text()
    index = (ROOT / "app" / "static" / "index.html").read_text()

    assert "p_birads_4_5" in viewer
    assert "p_cancer" not in viewer
    assert "malignancy_prob" not in viewer
    assert "рак" not in viewer.lower()
    assert "рак" not in index.lower()


def test_inference_module_does_not_claim_pathology_or_cancer_probability():
    source = (ROOT / "mammo" / "inference.py").read_text().lower()

    assert "malignancy level" not in source
    assert "suspicious for cancer" not in source


def test_load_image_rejects_excessive_decoded_pixels(monkeypatch):
    buf = io.BytesIO()
    Image.new("L", (100, 100)).save(buf, format="PNG")
    monkeypatch.setattr(inference_module, "MAX_DECODED_PIXELS", 9_999)

    with pytest.raises(ValueError, match="decoded pixel limit"):
        inference_module.load_image(buf.getvalue())


def test_dicom_metadata_rejects_unsupported_transfer_syntax():
    dataset = SimpleNamespace(
        Rows=100,
        Columns=100,
        NumberOfFrames=1,
        SamplesPerPixel=1,
        BitsAllocated=16,
        PhotometricInterpretation="MONOCHROME2",
        file_meta=SimpleNamespace(TransferSyntaxUID="1.2.3.4.5"),
    )

    with pytest.raises(ValueError, match="unsupported DICOM transfer syntax"):
        inference_module.validate_dicom_metadata(dataset)


def test_dicom_metadata_rejects_multiple_frames():
    dataset = SimpleNamespace(
        Rows=100,
        Columns=100,
        NumberOfFrames=2,
        SamplesPerPixel=1,
        BitsAllocated=16,
        PhotometricInterpretation="MONOCHROME2",
        file_meta=SimpleNamespace(TransferSyntaxUID="1.2.840.10008.1.2.1"),
    )

    with pytest.raises(ValueError, match="frame limit"):
        inference_module.validate_dicom_metadata(dataset)


@pytest.mark.parametrize("photometric", ["PALETTE COLOR", "RGB", ""])
def test_dicom_metadata_rejects_non_monochrome_photometric_interpretation(photometric):
    dataset = SimpleNamespace(
        Rows=100,
        Columns=100,
        NumberOfFrames=1,
        SamplesPerPixel=1,
        BitsAllocated=8,
        PhotometricInterpretation=photometric,
        file_meta=SimpleNamespace(TransferSyntaxUID="1.2.840.10008.1.2.1"),
    )

    with pytest.raises(ValueError, match="PhotometricInterpretation"):
        inference_module.validate_dicom_metadata(dataset)

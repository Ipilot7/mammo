import pickle
import sys
from types import SimpleNamespace

import pytest
import torch

from mammo import inference as inference_module
from mammo.classifier import load_classifier
from mammo.screening import load_screening


class UnsafePayload:
    pass


@pytest.mark.parametrize("loader", [load_classifier, load_screening])
def test_checkpoint_loaders_reject_non_tensor_python_objects(tmp_path, loader):
    checkpoint = tmp_path / "unsafe.pt"
    torch.save({"payload": UnsafePayload()}, checkpoint)

    with pytest.raises(pickle.UnpicklingError):
        loader(checkpoint)


def test_screening_loader_rejects_incompatible_birads_order(tmp_path):
    checkpoint = tmp_path / "wrong-order.pt"
    torch.save(
        {
            "arch": "efficientnet_b0",
            "birads_classes": [5, 4, 3, 2, 1],
            "state_dict": {},
            "threshold": 0.5,
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="BI-RADS classes"):
        load_screening(checkpoint)


def test_detector_checkpoint_rejects_sha256_mismatch(tmp_path):
    checkpoint = tmp_path / "detector.pt"
    checkpoint.write_bytes(b"trusted-model")
    checkpoint.with_suffix(".pt.sha256").write_text(f"{'0' * 64}  detector.pt\n")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        inference_module.verify_detector_checkpoint(checkpoint)


def test_pipeline_verifies_detector_before_ultralytics_load(monkeypatch, tmp_path):
    checkpoint = tmp_path / "detector.pt"
    checkpoint.write_bytes(b"unverified")
    loaded = False

    class FakeYOLO:
        def __init__(self, _path):
            nonlocal loaded
            loaded = True

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))

    with pytest.raises(FileNotFoundError, match="SHA-256 sidecar"):
        inference_module.MammoPipeline(
            detector_path=checkpoint,
            classifier_path=tmp_path / "classifier.pt",
            screening_path=tmp_path / "screening.pt",
            device="cpu",
        )

    assert loaded is False

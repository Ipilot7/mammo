import hashlib
import subprocess
import sys
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from mammo.inference import screening_summary
from train import evaluate as evaluate_module
from train.evaluate import append_image_level_predictions


def write_sha256_sidecar(path):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".pt.sha256").write_text(f"{digest}  {path.name}\n")


def test_missed_detection_is_counted_as_false_negative():
    rows = pd.DataFrame({"malignant": [1], "birads_idx": [2]})
    probs, ys, birads_pred, birads_true = [], [], [], []

    append_image_level_predictions(rows, [], probs, ys, birads_pred, birads_true)

    assert probs == [0.0]
    assert ys == [1]
    assert birads_pred == [-1]
    assert birads_true == [2]


def test_image_birads_uses_most_severe_finding_not_highest_proxy_score():
    rows = pd.DataFrame({"malignant": [1], "birads_idx": [2]})
    findings = [
        SimpleNamespace(p_birads_4_5=0.9, birads="3"),
        SimpleNamespace(p_birads_4_5=0.8, birads="5"),
    ]
    probs, ys, birads_pred, birads_true = [], [], [], []

    append_image_level_predictions(rows, findings, probs, ys, birads_pred, birads_true)

    assert probs == [0.9]
    assert birads_pred == [2]


def test_screening_summary_names_birads_proxy_without_claiming_cancer_probability():
    summary = screening_summary(torch.tensor([0.1, 0.1, 0.1, 0.3, 0.4]), threshold=0.5)

    assert summary["p_suspicious"] == pytest.approx(0.8)
    assert summary["p_birads_4_5"] == pytest.approx(0.7)
    assert "p_cancer" not in summary


def test_detector_evaluation_uses_deployed_image_height():
    kwargs = evaluate_module.detector_evaluation_kwargs(torch.device("cuda"))

    assert kwargs["imgsz"] == 1280


def test_lesion_matching_is_one_to_one():
    ground_truth = [(0, 0, 10, 10), (1, 1, 9, 9)]
    predictions = [(0, 0, 10, 10)]

    assert evaluate_module.count_lesion_matches(ground_truth, predictions, threshold=0.3) == 1


def test_evaluation_resume_refreshes_detector_from_best_checkpoint(tmp_path):
    models = tmp_path / "models"
    best = models / "runs" / "detector" / "weights" / "best.pt"
    target = models / "detector.pt"
    best.parent.mkdir(parents=True)
    best.write_bytes(b"new-best")
    write_sha256_sidecar(best)
    target.parent.mkdir(exist_ok=True)
    target.write_bytes(b"stale")

    result = evaluate_module.prepare_detector_checkpoint(models)

    assert result == target
    assert target.read_bytes() == b"new-best"
    assert target.with_suffix(".pt.sha256").read_text().endswith("  detector.pt\n")
    subprocess.run(
        ["sha256sum", "-c", "detector.pt.sha256"],
        cwd=models,
        check=True,
        capture_output=True,
        text=True,
    )


def test_evaluation_resume_fails_when_detector_checkpoints_are_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="detector checkpoint not found"):
        evaluate_module.prepare_detector_checkpoint(tmp_path / "models")


def test_direct_detector_evaluation_refreshes_checkpoint(monkeypatch, tmp_path):
    checkpoint = tmp_path / "detector.pt"
    checkpoint.write_bytes(b"checkpoint")
    write_sha256_sidecar(checkpoint)
    loaded = {}

    def prepare():
        loaded["prepared"] = True
        return checkpoint

    class FakeYOLO:
        def __init__(self, path):
            loaded["path"] = path

        def val(self, **kwargs):
            loaded["kwargs"] = kwargs
            return "metrics"

    monkeypatch.setattr(evaluate_module, "prepare_detector_checkpoint", prepare)
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))

    result = evaluate_module.evaluate_detector(torch.device("cuda"))

    assert result == "metrics"
    assert loaded["prepared"] is True
    assert loaded["path"] == str(checkpoint)
    assert loaded["kwargs"]["imgsz"] == 1280


def test_direct_detector_evaluation_accepts_explicit_candidate_without_promotion(monkeypatch, tmp_path):
    candidate = tmp_path / "candidate.pt"
    candidate.write_bytes(b"candidate")
    write_sha256_sidecar(candidate)
    loaded = {}

    def unexpected_prepare():
        raise AssertionError("explicit candidate must not refresh production checkpoint")

    class FakeYOLO:
        def __init__(self, path):
            loaded["path"] = path

        def val(self, **kwargs):
            return "candidate-metrics"

    monkeypatch.setattr(evaluate_module, "prepare_detector_checkpoint", unexpected_prepare)
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))

    result = evaluate_module.evaluate_detector(torch.device("cuda"), candidate)

    assert result == "candidate-metrics"
    assert loaded["path"] == str(candidate)


def test_explicit_candidate_requires_sha256_sidecar(tmp_path):
    candidate = tmp_path / "candidate.pt"
    candidate.write_bytes(b"candidate")

    with pytest.raises(FileNotFoundError, match="SHA-256 sidecar"):
        evaluate_module.resolve_detector_checkpoint(candidate)

from mammo.inference import verify_detector_checkpoint
from train import train_detector


def test_detector_promotion_creates_and_verifies_source_and_target_sidecars(tmp_path):
    best = tmp_path / "runs" / "detector" / "weights" / "best.pt"
    target = tmp_path / "detector.pt"
    best.parent.mkdir(parents=True)
    best.write_bytes(b"new-detector")

    result = train_detector.promote_detector_checkpoint(best, target)

    assert result == target
    assert target.read_bytes() == best.read_bytes()
    assert best.with_suffix(".pt.sha256").is_file()
    assert target.with_suffix(".pt.sha256").is_file()
    assert verify_detector_checkpoint(best) == verify_detector_checkpoint(target)

"""Evaluate detector, classifier and the full pipeline on the held-out test split."""
import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader

from mammo.classifier import load_classifier
from mammo.config import (
    BIRADS_CLASSES,
    DATA,
    IMAGE_HEIGHT,
    LABELS_CSV,
    MODELS,
    YOLO_DIR,
)
from mammo.inference import (
    MammoPipeline,
    verify_detector_checkpoint,
    write_detector_checkpoint_sidecar,
)
from mammo.screening import BREAST_BIRADS
from train.train_classifier import CropDataset, evaluate, pick_device
from train.train_screening import screening_metrics


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def detector_evaluation_kwargs(device) -> dict:
    return {
        "data": str(YOLO_DIR / "data.yaml"),
        "split": "test",
        "imgsz": IMAGE_HEIGHT,
        "device": str(device),
        "verbose": False,
    }


def count_lesion_matches(ground_truth, predictions, threshold=0.3) -> int:
    """Greedily count one-to-one localization matches by descending IoU."""
    candidates = sorted(
        (
            (iou(gt_box, pred_box), gt_idx, pred_idx)
            for gt_idx, gt_box in enumerate(ground_truth)
            for pred_idx, pred_box in enumerate(predictions)
            if iou(gt_box, pred_box) >= threshold
        ),
        reverse=True,
    )
    matched_gt, matched_pred = set(), set()
    for _, gt_idx, pred_idx in candidates:
        if gt_idx not in matched_gt and pred_idx not in matched_pred:
            matched_gt.add(gt_idx)
            matched_pred.add(pred_idx)
    return len(matched_gt)


def prepare_detector_checkpoint(models_dir=MODELS):
    """Refresh the deployable detector from the local best training checkpoint."""
    best = models_dir / "runs" / "detector" / "weights" / "best.pt"
    target = models_dir / "detector.pt"
    if best.is_file():
        verify_detector_checkpoint(best)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, target)
        write_detector_checkpoint_sidecar(target)
    if not target.is_file():
        raise FileNotFoundError(f"detector checkpoint not found: {best} or {target}")
    verify_detector_checkpoint(target)
    return target


def resolve_detector_checkpoint(checkpoint=None):
    """Resolve an explicit candidate without promoting it, or refresh production."""
    if checkpoint is None:
        checkpoint = prepare_detector_checkpoint()
    else:
        checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"detector checkpoint not found: {checkpoint}")
    verify_detector_checkpoint(checkpoint)
    return checkpoint


def evaluate_detector(device, checkpoint=None):
    """Evaluate a detector checkpoint at the deployed image size."""
    from ultralytics import YOLO

    detector_path = resolve_detector_checkpoint(checkpoint)
    return YOLO(str(detector_path)).val(**detector_evaluation_kwargs(device))


def append_image_level_predictions(rows, findings, probs, ys, birads_pred, birads_true):
    """Collect image-level metrics, including detector misses as false negatives."""
    top = findings[0] if findings else None
    if rows.malignant.notna().any():
        probs.append(top.p_birads_4_5 if top else 0.0)
        ys.append(int(rows.malignant.max()))
    if rows.birads_idx.notna().any():
        predicted_birads = max((BIRADS_CLASSES.index(f.birads) for f in findings), default=-1)
        birads_pred.append(predicted_birads)
        birads_true.append(int(rows.birads_idx.max()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default=None)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--detector-checkpoint", type=Path)
    args = p.parse_args()
    device = pick_device(args.device)
    detector_path = resolve_detector_checkpoint(args.detector_checkpoint)

    df = pd.read_csv(LABELS_CSV, dtype={"malignant": "Int64", "birads_idx": "Int64"})
    test = df[(df.split == "test") & np.array([(DATA / p).exists() for p in df.image_path])]
    if args.limit:
        keep = test.drop_duplicates("image_path").groupby("is_normal").head(args.limit).image_path
        test = test[test.image_path.isin(keep)]

    print("== Detector (YOLO val on test split) ==")
    m = evaluate_detector(device, detector_path)
    print(f"mAP50 {m.box.map50:.3f}  mAP50-95 {m.box.map:.3f}")

    print("\n== Classifier on ground-truth crops ==")
    labeled = test[(test.has_box == 1) & (test.malignant.notna() | test.birads_idx.notna())]
    model = load_classifier(MODELS / "classifier.pt", device)
    print(evaluate(model, DataLoader(CropDataset(labeled, False), 32), device))

    print("\n== Full pipeline (per image) ==")
    pipe = MammoPipeline(detector_path=detector_path, device=str(device))
    hits = n_boxes = 0
    probs, ys, b_pred, b_true = [], [], [], []
    normal_boxes = {0.05: [], 0.25: []}
    scr = {"probs": [], "birads": []}
    for path, rows in test.groupby("image_path", sort=False):
        img = Image.open(DATA / path)
        findings = pipe.analyze(img, conf=0.05)
        first = rows.iloc[0]
        if pipe.screening is not None and first.breast_birads in BREAST_BIRADS:
            scr["probs"].append([pipe.screen(img)["birads_probs"][str(b)] for b in BREAST_BIRADS])
            scr["birads"].append(int(first.breast_birads))
        if first.is_normal:
            for c, counts in normal_boxes.items():
                counts.append(sum(f.type_conf >= c for f in findings))
            continue
        ground_truth = [(r.x1, r.y1, r.x2, r.y2) for r in rows[rows.has_box == 1].itertuples()]
        predictions = [f.bbox_xyxy for f in findings]
        n_boxes += len(ground_truth)
        hits += count_lesion_matches(ground_truth, predictions)
        append_image_level_predictions(rows, findings, probs, ys, b_pred, b_true)
    print(f"lesions found (IoU>=0.3 among top-5): {hits}/{n_boxes} = {hits / max(n_boxes, 1):.3f}")
    if len(set(ys)) == 2:
        print(f"image-level BI-RADS 4-5 proxy AUC: {roc_auc_score(ys, probs):.3f}")
    if b_true:
        print(f"BI-RADS accuracy: {np.mean(np.array(b_pred) == np.array(b_true)):.3f}")
        labels = [-1, *range(len(BIRADS_CLASSES))]
        print("BI-RADS confusion (rows=true, cols=pred):", ["missed", *BIRADS_CLASSES])
        print(confusion_matrix(b_true, b_pred, labels=labels))
    for c, counts in normal_boxes.items():
        if counts:
            print(f"healthy images: {len(counts)}, false boxes per image at conf>={c}: {np.mean(counts):.2f}, "
                  f"images with any box: {np.mean(np.array(counts) > 0):.2f}")

    if scr["probs"]:
        print("\n== Whole-breast BI-RADS (screening model) ==")
        m = screening_metrics(scr["probs"], scr["birads"], threshold=pipe.screening_threshold)
        confusion = m.pop("confusion")
        print(m)
        print("confusion (rows = true BI-RADS 1-5, cols = predicted):")
        print(np.array(confusion))

if __name__ == "__main__":
    main()

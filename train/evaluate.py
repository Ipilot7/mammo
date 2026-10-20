"""Evaluate detector, classifier and the full pipeline on the held-out test split."""
import argparse

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader

from mammo.classifier import load_classifier
from mammo.config import BIRADS_CLASSES, DATA, LABELS_CSV, MODELS, YOLO_DIR
from mammo.inference import MammoPipeline
from train.train_classifier import CropDataset, evaluate, pick_device
from mammo.screening import BREAST_BIRADS
from train.train_screening import screening_metrics


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default=None)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    device = pick_device(args.device)

    df = pd.read_csv(LABELS_CSV, dtype={"malignant": "Int64", "birads_idx": "Int64"})
    test = df[(df.split == "test") & np.array([(DATA / p).exists() for p in df.image_path])]
    if args.limit:
        keep = test.drop_duplicates("image_path").groupby("is_normal").head(args.limit).image_path
        test = test[test.image_path.isin(keep)]

    print("== Detector (YOLO val on test split) ==")
    from ultralytics import YOLO
    m = YOLO(str(MODELS / "detector.pt")).val(data=str(YOLO_DIR / "data.yaml"), split="test",
                                              imgsz=1024, device=str(device), verbose=False)
    print(f"mAP50 {m.box.map50:.3f}  mAP50-95 {m.box.map:.3f}")

    print("\n== Classifier on ground-truth crops ==")
    labeled = test[(test.has_box == 1) & (test.malignant.notna() | test.birads_idx.notna())]
    model = load_classifier(MODELS / "classifier.pt", device)
    print(evaluate(model, DataLoader(CropDataset(labeled, False), 32), device))

    print("\n== Full pipeline (per image) ==")
    pipe = MammoPipeline(device=str(device))
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
            for c in normal_boxes:
                normal_boxes[c].append(sum(f.type_conf >= c for f in findings))
            continue
        for r in rows[rows.has_box == 1].itertuples():
            n_boxes += 1
            hits += any(iou(f.bbox_xyxy, (r.x1, r.y1, r.x2, r.y2)) >= 0.3 for f in findings)
        if not findings:
            continue
        top = findings[0]  # highest malignancy = image-level score
        if rows.malignant.notna().any():
            probs.append(top.malignancy_prob)
            ys.append(int(rows.malignant.max()))
        if rows.birads_idx.notna().any():
            b_pred.append(BIRADS_CLASSES.index(top.birads))
            b_true.append(int(rows.birads_idx.max()))
    print(f"lesions found (IoU>=0.3 among top-5): {hits}/{n_boxes} = {hits / max(n_boxes, 1):.3f}")
    if len(set(ys)) == 2:
        print(f"image-level malignancy AUC: {roc_auc_score(ys, probs):.3f}")
    if b_true:
        print(f"BI-RADS accuracy: {np.mean(np.array(b_pred) == np.array(b_true)):.3f}")
        print("BI-RADS confusion (rows=true, cols=pred):", BIRADS_CLASSES)
        print(confusion_matrix(b_true, b_pred, labels=range(len(BIRADS_CLASSES))))
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

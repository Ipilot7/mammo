"""Convert data/labels.csv into an Ultralytics YOLO dataset (symlinked images + txt labels).

Healthy images get an empty label file: YOLO treats them as background and learns not to fire on
normal tissue. Abnormal images without boxes (KAU-BCMD BI-RADS 3-5) are left out of detection.
"""
import argparse
import os
import shutil

import pandas as pd

from mammo.config import DATA, FINDING_CLASSES, LABELS_CSV, YOLO_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bg-ratio", type=float, default=1.0,
                    help="healthy background images per image with boxes in train/val (test keeps all)")
    args = ap.parse_args()
    df = pd.read_csv(LABELS_CSV)
    df = df[(df.has_box == 1) | (df.is_normal == 1)]
    # VinDr is ~90% healthy: keep every image with findings plus a random sample of healthy ones,
    # otherwise an epoch at 1280 px takes ~17 min on a T4.
    keep = []
    for split, g in df.groupby("split"):
        images = g.drop_duplicates("image_path")
        boxed, healthy = images[images.has_box == 1], images[images.has_box == 0]
        if split != "test":
            healthy = healthy.sample(min(len(healthy), int(len(boxed) * args.bg_ratio)), random_state=0)
        keep += list(boxed.image_path) + list(healthy.image_path)
    df = df[df.image_path.isin(set(keep))]
    if YOLO_DIR.exists():
        shutil.rmtree(YOLO_DIR)
    for split, g in df.groupby("split"):
        img_dir, lbl_dir = YOLO_DIR / "images" / split, YOLO_DIR / "labels" / split
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        n_bg = 0
        for i, (path, rows) in enumerate(g.groupby("image_path", sort=False)):
            src = DATA / path
            if not src.exists():
                continue
            first = rows.iloc[0]
            stem = f"{first.source}_{i:05d}"
            os.symlink(src.resolve(), img_dir / f"{stem}.png")
            lines = []
            for r in rows[rows.has_box == 1].itertuples():
                w, h = r.width, r.height
                x1, y1 = max(r.x1, 0), max(r.y1, 0)
                x2, y2 = min(r.x2, w), min(r.y2, h)
                if x2 - x1 < 1 or y2 - y1 < 1:
                    continue
                cls = FINDING_CLASSES.index(r.finding_class)
                lines.append(f"{cls} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
            n_bg += not lines
            (lbl_dir / f"{stem}.txt").write_text("".join(line + "\n" for line in lines))
        print(f"{split}: {len(list(img_dir.iterdir()))} images ({n_bg} healthy background)")

    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(FINDING_CLASSES))
    (YOLO_DIR / "data.yaml").write_text(
        f"path: {YOLO_DIR.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{names}\n"
    )
    print(f"wrote {YOLO_DIR / 'data.yaml'}")


if __name__ == "__main__":
    main()

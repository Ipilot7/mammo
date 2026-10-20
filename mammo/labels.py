"""Build data/labels.csv from VinDr-Mammo (after scripts/vindr_fetch.py).

One row per annotated finding (box), plus one box-less row for every image without findings.
- is_normal: breast BI-RADS 1-2 and no box -> healthy (detector background, screening label 0);
- malignant: VinDr has no biopsy results, so it is a proxy: finding BI-RADS 4-5 = suspicious for
  malignancy (1), BI-RADS 2-3 = probably benign (0);
- split: official VinDr test split; 15% of training studies (exams) go to val.
Boxes are converted to the coordinates of the stored breast-cropped PNGs.
image_path is relative to data/.
"""
import ast

import numpy as np
import pandas as pd

from mammo.config import BIRADS_CLASSES, DATA, LABELS_CSV, VINDR, birads_number, birads_to_index, finding_class
from mammo.split import assign_splits

CATEGORY_MAP = {
    "Mass": "mass",
    "Suspicious Calcification": "calcifications",
    "Architectural Distortion": "architectural_distortion",
    "Asymmetry": "asymmetry",
    "Focal Asymmetry": "asymmetry",
    "Global Asymmetry": "asymmetry",
}


def categories(value) -> list[str]:
    try:
        cats = ast.literal_eval(value) if isinstance(value, str) else []
    except (ValueError, SyntaxError):
        cats = [value]
    return [c for c in cats if c != "No Finding"]


def build() -> pd.DataFrame:
    images = pd.read_csv(VINDR / "breast-level_annotations.csv")
    transforms = pd.read_csv(VINDR / "transforms.csv")
    images = images.merge(transforms, on="image_id", suffixes=("_orig", ""))  # converted images only
    images["breast_birads"] = images["breast_birads"].map(birads_number)

    findings = pd.read_csv(VINDR / "finding_annotations.csv").dropna(subset=["xmin"])
    findings = findings[findings.image_id.isin(images.image_id)]
    t = images.set_index("image_id")
    rows = []
    for f in findings.itertuples():
        im = t.loc[f.image_id]
        x1, x2 = np.clip([(f.xmin - im.x0) * im.scale, (f.xmax - im.x0) * im.scale], 0, im.width)
        y1, y2 = np.clip([(f.ymin - im.y0) * im.scale, (f.ymax - im.y0) * im.scale], 0, im.height)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        cats = categories(f.finding_categories)
        b = birads_number(f.finding_birads)
        rows.append(dict(image_id=f.image_id, x1=x1, y1=y1, x2=x2, y2=y2,
                         primary_finding_type="|".join(cats),
                         finding_class=finding_class(CATEGORY_MAP.get(cats[0], "other")) if cats else "other",
                         birads_idx=birads_to_index(b), malignant=None if b is None else int(b >= 4)))
    boxes = pd.DataFrame(rows, columns=["image_id", "x1", "y1", "x2", "y2", "primary_finding_type",
                                        "finding_class", "birads_idx", "malignant"])

    boxed = set(boxes.image_id)
    no_box = images.loc[~images.image_id.isin(boxed), ["image_id"]]
    df = pd.concat([boxes, no_box], ignore_index=True).merge(images, on="image_id")
    df["has_box"] = df["x1"].notna().astype(int)
    df["is_normal"] = ((df["breast_birads"] <= 2) & ~df["image_id"].isin(boxed)).astype(int)
    df["image_path"] = "vindr/png/" + df["image_id"] + ".png"
    df["source"] = "vindr"
    df["patient_id"] = "vindr:" + df["study_id"]  # one exam per woman in VinDr
    df = df.rename(columns={"view_position": "view"})
    df["split"] = np.where(df["split"] == "test", "test", assign_splits(df["patient_id"], val=0.15, test=0.0))
    df["birads_idx"] = df["birads_idx"].astype("Int64")
    df["malignant"] = df["malignant"].astype("Int64")
    cols = ["image_path", "source", "patient_id", "split", "view", "laterality", "width", "height",
            "breast_birads", "is_normal", "has_box", "x1", "y1", "x2", "y2", "primary_finding_type",
            "finding_class", "birads_idx", "malignant"]
    return df[cols]


def main():
    df = build()
    df.to_csv(LABELS_CSV, index=False)
    print(f"wrote {LABELS_CSV} ({len(df)} rows)")
    images = df.drop_duplicates("image_path")
    print(images.groupby("split").agg(images=("image_path", "size"), healthy=("is_normal", "sum"),
                                      suspicious=("breast_birads", lambda b: int((b >= 3).sum()))))
    boxes = df[df.has_box == 1]
    print("boxes by class:", boxes.finding_class.value_counts().to_dict())
    print("boxes by finding BI-RADS:", boxes.birads_idx.map(lambda i: BIRADS_CLASSES[i] if pd.notna(i) else "?")
          .value_counts().to_dict())
    print("breast BI-RADS:", images.breast_birads.value_counts().sort_index().to_dict())
    leak = df.groupby("patient_id")["split"].nunique().max()
    print(f"max splits per study: {leak} (must be 1)")
    missing = sum(not (DATA / p).exists() for p in images.image_path)
    print(f"missing image files: {missing}")


if __name__ == "__main__":
    main()

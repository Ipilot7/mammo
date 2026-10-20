"""Build data/labels.csv: one row per finding, plus one box-less row per healthy image.

Sources:
- MammoGPS (CBIS-DDSM, CMMD, INbreast) boxes joined with BI-RADS / pathology from the source datasets;
- DMID: healthy images + findings with boxes and pathology (scripts/download_extra.py);
- KAU-BCMD: healthy images (BI-RADS 1) + BI-RADS 3-5 images without boxes.
image_path is relative to data/.
"""
import json
import re

import pandas as pd

from mammo.config import DATA, LABELS_CSV, MAMMOGPS, SOURCES, birads_to_index, finding_class
from mammo.split import assign_splits

CBIS_KEY = re.compile(r"^((?:Calc|Mass)-(?:Training|Test)_(P_\d+)_(LEFT|RIGHT)_(CC|MLO))")
CMMD_KEY = re.compile(r"^(D\d-\d{4})_")
INBREAST_KEY = re.compile(r"AllDICOMs_(\d+)_([0-9a-f]+)_MG_")


def load_core() -> pd.DataFrame:
    rows = [json.loads(line) for line in (MAMMOGPS / "core.jsonl").open()]
    df = pd.DataFrame(rows)
    df["source"] = df["image_path"].str.split("/").str[2]
    df = df[df["source"] != "vindr"].reset_index(drop=True)  # VinDr images are not distributed
    df["finding_class"] = df["primary_finding_type"].map(finding_class)
    return df


def load_cbis() -> dict:
    """image key (e.g. Mass-Training_P_00001_LEFT_CC) -> (birads_idx, malignant)."""
    frames = [pd.read_csv(p) for p in sorted(SOURCES.glob("*_case_description_*_set.csv"))]
    if not frames:
        return {}
    df = pd.concat(frames, ignore_index=True)
    df["key"] = df["image file path"].str.split("/").str[0]
    df["malignant"] = (df["pathology"].str.upper() == "MALIGNANT").astype(int)
    df["birads_idx"] = df["assessment"].map(birads_to_index)
    out = {}
    for key, g in df.groupby("key"):
        # Several abnormalities per image: take the most severe.
        out[key] = (g["birads_idx"].max(), int(g["malignant"].max()))
    return out


def load_cmmd() -> dict:
    """(ID1, laterality) -> malignant."""
    path = SOURCES / "CMMD_clinicaldata_revision.xlsx"
    if not path.exists():
        return {}
    df = pd.read_excel(path)
    df["malignant"] = (df["classification"].str.lower() == "malignant").astype(int)
    return {(r.ID1, r.LeftRight): int(r.malignant) for r in df.itertuples()}


def load_inbreast() -> dict:
    """file name (int as str) -> birads_idx."""
    path = SOURCES / "INbreast.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip().lower() for c in df.columns]
    return {str(r["file name"]).split(".")[0]: birads_to_index(r["bi-rads"]) for _, r in df.iterrows()}


def build() -> pd.DataFrame:
    df = load_core()
    cbis, cmmd, inbreast = load_cbis(), load_cmmd(), load_inbreast()

    patient, birads, malignant = [], [], []
    for r in df.itertuples():
        name = r.image_path.split("/")[-1]
        b = m = None
        pid = name
        if r.source == "cbis_ddsm" and (mt := CBIS_KEY.match(name)):
            pid = mt.group(2)
            if mt.group(1) in cbis:
                b, m = cbis[mt.group(1)]
        elif r.source == "cmmd" and (mt := CMMD_KEY.match(name)):
            pid = mt.group(1)
            m = cmmd.get((pid, r.laterality))
        elif r.source == "inbreast" and (mt := INBREAST_KEY.search(name)):
            pid = mt.group(2)
            b = inbreast.get(mt.group(1))
            if b is not None:
                m = int(b >= 3)  # BI-RADS >= 4 treated as malignant (INbreast has no pathology)
        patient.append(f"{r.source}:{pid}")
        birads.append(b)
        malignant.append(m)

    df["patient_id"] = patient
    df["birads_idx"] = pd.array(birads, dtype="Int64")
    df["malignant"] = pd.array(malignant, dtype="Int64")
    x1, y1, x2, y2 = zip(*df["primary_bbox_xyxy"])
    df["x1"], df["y1"], df["x2"], df["y2"] = x1, y1, x2, y2
    df["image_path"] = str(MAMMOGPS.relative_to(DATA)) + "/" + df["image_path"]
    df["is_normal"] = 0
    df = pd.concat([df, load_dmid(), load_kau()], ignore_index=True)
    df["finding_class"] = df["finding_class"].astype(object)
    df["birads_idx"] = df["birads_idx"].astype("Int64")
    df["malignant"] = df["malignant"].astype("Int64")
    df["has_box"] = df["x1"].notna().astype(int)
    df["split"] = assign_splits(df["patient_id"])
    cols = ["image_path", "source", "patient_id", "split", "view", "laterality", "width", "height",
            "is_normal", "has_box", "x1", "y1", "x2", "y2", "primary_finding_type", "finding_class",
            "birads_idx", "malignant"]
    return df[cols]


def load_dmid() -> pd.DataFrame:
    path = DATA / "extra" / "dmid.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["source"] = "dmid"
    df["patient_id"] = "dmid:" + df["image"]  # DMID has no patient ids
    df["primary_finding_type"] = df["finding_type"]
    df["finding_class"] = df["finding_type"].map(lambda t: finding_class(t) if isinstance(t, str) else None)
    df["birads_idx"] = None
    return df


def load_kau() -> pd.DataFrame:
    path = DATA / "extra" / "kau.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["source"] = "kau"
    df["patient_id"] = "kau:" + df["patient"].astype(str)
    df["is_normal"] = (df["birads"] == 1).astype(int)
    df["birads_idx"] = df["birads"].map(birads_to_index)
    df["malignant"] = None  # image-level BI-RADS only, no pathology and no boxes
    df["primary_finding_type"] = df["finding_class"] = None
    return df


def main():
    df = build()
    df.to_csv(LABELS_CSV, index=False)
    print(f"wrote {LABELS_CSV} ({len(df)} rows)")
    images = df.drop_duplicates("image_path")
    print(images.groupby("source").agg(images=("image_path", "size"), normal=("is_normal", "sum")).join(
        df.groupby("source").agg(boxes=("has_box", "sum"), with_malignant=("malignant", "count"),
                                 with_birads=("birads_idx", "count"))))
    print(images.groupby("split").agg(images=("image_path", "size"), normal=("is_normal", "sum")))
    leak = df.groupby("patient_id")["split"].nunique().max()
    print(f"max splits per patient: {leak} (must be 1)")


if __name__ == "__main__":
    main()

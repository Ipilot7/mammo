"""Download datasets with healthy breasts: KAU-BCMD (CC0) and DMID (CC-BY-4.0) from Hugging Face.

Each image is downloaded, cropped to the breast, resized to IMAGE_HEIGHT, saved as PNG and the
original (up to 15 MB TIFF) is deleted, so ~14 GB of sources become a few hundred MB.
Re-running skips images that are already converted. Writes data/extra/{kau,dmid}.csv manifests.
"""
import argparse
import re
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mammo.config import DATA  # noqa: E402
from mammo.preprocess import crop_breast, to_gray_u8  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
OUT = DATA / "extra"
KAU_REPO = "MyTwinLab/KAU-BCMD_Mamography_dataset"
DMID_REPO = "MyTwinLab/DMID_Breast_Cancer_Mammography_Dataset"

KAU_NAME = re.compile(r"^(?P<patient>.+?)_\s*(?P<view>CC|MLO)_(?P<lat>[LR])\.png$", re.IGNORECASE)
DMID_TYPES = {"CIRC": "mass", "SPIC": "mass", "MISC": "mass", "CALC": "calcifications", "CLAC": "calcifications",
              "ARCH": "architectural_distortion", "ASYM": "asymmetry"}


def convert(repo: str, remote: str, dst: Path, tmp: str):
    """Download one file, crop + resize, save PNG. Returns (width, height, transform)."""
    src = hf_hub_download(repo, remote, repo_type="dataset", local_dir=tmp)
    try:
        img, transform = crop_breast(to_gray_u8(Image.open(src)))
        dst.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst)
        return img.width, img.height, transform
    finally:
        Path(src).unlink(missing_ok=True)


def run_parallel(jobs, workers, label):
    done, failed = 0, 0
    with ThreadPoolExecutor(workers) as ex:
        futures = {ex.submit(fn): key for key, fn in jobs}
        results = {}
        for f in as_completed(futures):
            try:
                results[futures[f]] = f.result()
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"  failed {futures[f]}: {e}", flush=True)
            done += 1
            if done % 50 == 0 or done == len(futures):
                print(f"{label}: {done}/{len(futures)} (failed {failed})", flush=True)
    return results


def kau(limit: int, workers: int, tmp: str):
    files = [f for f in HfApi().list_repo_files(KAU_REPO, repo_type="dataset") if f.endswith(".png")]
    if limit:  # smoke tests: a few images of each class
        files = [f for folder in sorted({f.split("/")[0] for f in files})
                 for f in [x for x in files if x.startswith(folder + "/")][:limit]]
    rows, jobs = [], []
    for f in files:
        folder, name = f.split("/", 1)
        m = KAU_NAME.match(name)
        if not m:
            print("  skip unparsed name", f)
            continue
        dst = OUT / "kau" / folder / name.replace(" ", "")
        rows.append(dict(image_path=str(dst.relative_to(DATA)), patient=m["patient"].strip(), view=m["view"].upper(),
                         laterality=m["lat"].upper(), birads=int(folder.replace("birads", "")), remote=f))
        if not dst.exists():
            jobs.append((f, lambda f=f, dst=dst: convert(KAU_REPO, f, dst, tmp)))
    run_parallel(jobs, workers, "KAU-BCMD")
    df = pd.DataFrame(rows)
    df = df[[(DATA / p).exists() for p in df.image_path]]
    sizes = [Image.open(DATA / p).size for p in df.image_path]
    df["width"], df["height"] = [s[0] for s in sizes], [s[1] for s in sizes]
    df.drop(columns="remote").to_csv(OUT / "kau.csv", index=False)
    print(f"KAU-BCMD: {len(df)} images, by BI-RADS: {df.birads.value_counts().to_dict()}")


def parse_dmid_metadata(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, header=None)
    raw = raw[raw[0].astype(str).str.strip().str.match(r"^IMG\d+$")]
    rows = []
    for r in raw.itertuples(index=False):
        view = str(r[1]).strip().upper()
        cls = str(r[3]).strip().upper().replace(" ", "")
        sev = str(r[4]).strip().upper()
        normal = cls == "NORM"
        first = cls.split("+")[0]
        rows.append(dict(
            image=str(r[0]).strip(), view="CC" if view.startswith("CC") else "MLO",
            laterality="L" if view.endswith("LT") else "R", is_normal=int(normal),
            finding_type=None if normal else DMID_TYPES.get(first, "other"),
            malignant={"M": 1, "B": 0}.get(sev),
            cx=pd.to_numeric(r[5], errors="coerce"), cy=pd.to_numeric(r[6], errors="coerce"),
            radius=pd.to_numeric(r[7], errors="coerce"),
        ))
    return pd.DataFrame(rows)


def dmid(limit: int, workers: int, tmp: str):
    meta = parse_dmid_metadata(Path(hf_hub_download(DMID_REPO, "Metadata.xlsx", repo_type="dataset", local_dir=tmp)))
    images = list(dict.fromkeys(meta.image))
    if limit:
        normal = meta[meta.is_normal == 1].image.unique().tolist()[:limit]
        abnormal = meta[meta.is_normal == 0].image.unique().tolist()[:limit]
        images = normal + abnormal

    transforms_file = OUT / "dmid_transforms.csv"  # remember crop transforms for resumed runs
    known = pd.read_csv(transforms_file).set_index("image").to_dict("index") if transforms_file.exists() else {}
    jobs = []
    for img in images:
        dst = OUT / "dmid" / f"{img}.png"
        if not (dst.exists() and img in known):
            jobs.append((img, lambda img=img, dst=dst: convert(DMID_REPO, f"TIFF Images/TIFF Images/{img}.tif", dst, tmp)))
    for img, (w, h, (x0, y0, s)) in run_parallel(jobs, workers, "DMID").items():
        known[img] = dict(width=w, height=h, x0=x0, y0=y0, scale=s)
    pd.DataFrame.from_dict(known, orient="index").rename_axis("image").reset_index().to_csv(transforms_file, index=False)

    rows = []
    for r in meta[meta.image.isin(known)].itertuples():
        t = known[r.image]
        row = dict(image_path=f"extra/dmid/{r.image}.png", image=r.image, view=r.view, laterality=r.laterality,
                   width=t["width"], height=t["height"], is_normal=r.is_normal, finding_type=r.finding_type,
                   malignant=r.malignant, x1=None, y1=None, x2=None, y2=None)
        if not r.is_normal and pd.notna(r.cx) and pd.notna(r.radius):
            # Metadata: centre and radius in original pixels, origin top-left.
            s, x0, y0 = t["scale"], t["x0"], t["y0"]
            row.update(x1=max(0.0, (r.cx - r.radius - x0) * s), y1=max(0.0, (r.cy - r.radius - y0) * s),
                       x2=min(t["width"], (r.cx + r.radius - x0) * s), y2=min(t["height"], (r.cy + r.radius - y0) * s))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "dmid.csv", index=False)
    print(f"DMID: {df.image.nunique()} images ({df[df.is_normal == 1].image.nunique()} normal), "
          f"{df.x1.notna().sum()} findings with boxes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="images per class (smoke tests)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", choices=["kau", "dmid"], default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="mammo_extra_")
    try:
        if args.only in (None, "dmid"):
            dmid(args.limit, args.workers, tmp)
        if args.only in (None, "kau"):
            kau(args.limit, args.workers, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()

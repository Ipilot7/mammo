"""Download MammoGPS core annotations and raw images (CBIS-DDSM, CMMD, INbreast) from Hugging Face."""
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "mammovlmbench/mammogps"
SOURCES = ["cbis_ddsm", "cmmd", "inbreast"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/mammogps")
    args = parser.parse_args()

    patterns = ["data/benchmark/core.jsonl"] + [f"data/benchmark/images/raw/{s}/*" for s in SOURCES]
    path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=args.out,
        allow_patterns=patterns,
        max_workers=16,
    )
    root = Path(path) / "data" / "benchmark"
    for s in SOURCES:
        n = len(list((root / "images" / "raw" / s).glob("*.png")))
        print(f"{s}: {n} images")


if __name__ == "__main__":
    main()

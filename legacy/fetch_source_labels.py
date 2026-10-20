"""Download label metadata of the source datasets (BI-RADS / pathology) into data/sources/."""
import urllib.request
from pathlib import Path

OUT = Path("data/sources")
TCIA = "https://www.cancerimagingarchive.net/wp-content/uploads/"
FILES = [
    "mass_case_description_train_set.csv",
    "mass_case_description_test_set.csv",
    "calc_case_description_train_set.csv",
    "calc_case_description_test_set.csv",
    "CMMD_clinicaldata_revision.xlsx",
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dst = OUT / name
        if dst.exists() and dst.stat().st_size > 0:
            print(f"ok      {name}")
            continue
        try:
            urllib.request.urlretrieve(TCIA + name, dst)
            print(f"fetched {name} ({dst.stat().st_size} bytes)")
        except Exception as e:  # noqa: BLE001
            print(f"FAILED  {name}: {e}\n        download manually from {TCIA + name} into {OUT}/")

    if not (OUT / "INbreast.csv").exists():
        print(
            "\nINbreast.csv not found (optional). It is distributed with the INbreast release / Kaggle\n"
            "(e.g. kaggle.com/datasets/ramanathansp20/inbreast-dataset). Put INbreast.csv into data/sources/\n"
            "to get BI-RADS labels for INbreast images; without it they are used for detection only."
        )


if __name__ == "__main__":
    main()

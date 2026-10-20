import hashlib

import pandas as pd


def assign_splits(patient_ids: pd.Series, val: float = 0.15, test: float = 0.15) -> pd.Series:
    """Deterministic patient-level split: all images of a patient land in the same split."""

    def bucket(pid: str) -> str:
        h = int(hashlib.md5(pid.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        if h < test:
            return "test"
        if h < test + val:
            return "val"
        return "train"

    return patient_ids.map(bucket)

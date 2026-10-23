import pandas as pd

from mammo.config import BIRADS_CLASSES, birads_to_index
from mammo.labels import birads_class_name


def test_finding_birads_classes_match_vindr_annotations():
    assert BIRADS_CLASSES == ["3", "4", "5"]
    assert {value: birads_to_index(value) for value in range(7)} == {
        0: None,
        1: None,
        2: None,
        3: 0,
        4: 1,
        5: 2,
        6: None,
    }


def test_birads_class_name_accepts_nullable_pandas_values():
    assert birads_class_name(0.0) == "3"
    assert birads_class_name(1.0) == "4"
    assert birads_class_name(2.0) == "5"
    assert birads_class_name(pd.NA) == "?"

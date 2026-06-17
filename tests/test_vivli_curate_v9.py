import pandas as pd

from ghlobus.ingestion.vivli_curate_v9 import (
    build_curated_prototype,
    build_exam_metadata,
    normalize_lie,
    normalize_nof,
)


def test_normalize_lie():
    assert normalize_lie("1") == 0
    assert normalize_lie("Cephalic") == 0
    assert normalize_lie("2") == 1
    assert normalize_lie("Transverse") == 1
    assert pd.isna(normalize_lie("98"))
    assert pd.isna(normalize_lie(""))


def test_normalize_nof():
    assert normalize_nof("1", "") == 1
    assert normalize_nof("2", "0") == 2
    assert normalize_nof("", "1") == 2
    assert normalize_nof("", "0") == 1


def test_build_exam_metadata_prefers_sr_then_crf():
    crf = pd.DataFrame({
        "pidscan": ["FA3-001-1", "FA3-002-1"],
        "pid": ["FA3-001", "FA3-002"],
        "ega": ["100", "120"],
        "us_lie": ["1", "2"],
        "us_twin": ["1", "2"],
        "us_efw": ["500", "600"],
        "us_bpa": ["2", ""],
        "us_bpb": ["4", ""],
        "us_hca": ["10", "11"],
        "us_hcb": ["12", "13"],
        "us_aca": ["8", "9"],
        "us_acb": ["10", "11"],
        "us_fla": ["1", "2"],
        "us_flb": ["3", "4"],
    })
    sr = pd.DataFrame({
        "pidscan": ["FA3-001-1"],
        "pid": ["FA3-001"],
        "ega": ["101"],
        "multiple_gest": ["0"],
        "mean_bpd": ["5"],
        "mean_hc": ["14"],
        "mean_ac": ["9"],
        "mean_fl": ["2"],
        "mean_crl": ["7"],
    })

    out = build_exam_metadata(crf, sr)

    row1 = out[out["StudyID"] == "FA3-001-1"].iloc[0]
    assert row1["GA"] == 101
    assert row1["BPD"] == 5
    assert row1["lie"] == 0
    assert row1["NOF"] == 1

    row2 = out[out["StudyID"] == "FA3-002-1"].iloc[0]
    assert row2["GA"] == 120
    assert row2["BPD"] != row2["BPD"]
    assert row2["lie"] == 1
    assert row2["NOF"] == 2


def test_build_curated_prototype_preserves_preprocessing_columns():
    prototype = pd.DataFrame({
        "PID": ["FA3-001"],
        "StudyID": ["FA3-001-1"],
        "filename": ["video.mp4"],
        "exam_dir": ["FA3-001-1_20230101_010101_Butterfly_iQ+"],
        "fail_reason": ["Good_video"],
        "tag": ["M"],
    })
    exam = pd.DataFrame({
        "PID": ["FA3-001"],
        "StudyID": ["FA3-001-1"],
        "GA": [101],
        "BPD": [5],
        "lie": [0],
    })

    out = build_curated_prototype(prototype, exam)

    assert out.loc[0, "exam_dir"] == "FA3-001-1_20230101_010101_Butterfly_iQ+"
    assert out.loc[0, "fail_reason"] == "Good_video"
    assert out.loc[0, "GA"] == 101
    assert out.loc[0, "lie"] == 0

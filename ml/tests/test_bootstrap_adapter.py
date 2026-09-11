from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from ml.adapters.bootstrap_adapter import load_byu_tracks, normalize_iceberg_id


def _zip(tmp_path: Path, files: dict[str, str]) -> Path:
    p = tmp_path / "byu.zip"
    with zipfile.ZipFile(p, "w") as zf:
        for name, text in files.items():
            zf.writestr(f"updated7_consol/{name}", text)
    return p


def test_normalize_id() -> None:
    assert normalize_iceberg_id(" a-23a ") == "A23A"
    assert normalize_iceberg_id("B22_F") == "B22F"


def test_cell53_semantics(tmp_path: Path) -> None:
    csv = (
        "ascat_1,ascat_2,ascat_3,date,nic_1,nic_2,nic_3,size_1,size_2\n"
        "0.0,0.0,0,2017244,-67.9333,-60.85,1,82,25\n"  # nic used (priority)
        "-67.5,-60.5,1,2017245,0.0,0.0,0,0,0\n"  # nic 0/0 placeholder -> ascat
        "0.0,0.0,0,2017246,0.0,0.0,0,0,0\n"  # no position -> dropped
        "0.0,0.0,0,2017246,-68.0,-60.0,1,0,0\n"  # same date, second row -> kept (first valid)
        "-10.0,5.0,1,2017247,0.0,0.0,0,0,0\n"  # lat outside [-90,-45] -> dropped
    )
    tracks = load_byu_tracks(_zip(tmp_path, {"a68a.csv": csv}))
    assert list(tracks["iceberg_id"].unique()) == ["A68A"]
    assert len(tracks) == 3
    first = tracks.iloc[0]
    assert str(first["date"].date()) == "2017-09-01"  # day 244 of 2017
    assert first["position_source"] == "nic"
    assert tracks.iloc[1]["position_source"] == "ascat"
    assert tracks.iloc[0]["size_1"] == 82
    assert tracks["x_m"].notna().all()


def test_empty_zip_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        load_byu_tracks(_zip(tmp_path, {"x.csv": "foo,bar\n1,2\n"}))

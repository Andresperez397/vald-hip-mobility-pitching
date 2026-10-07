"""Data audit: structural checks on the pitcher table and the effect of each cleaning rule.

    VALD_DATA=path/to/master.csv PYTHONPATH=src python scripts/01_data_audit.py
    PYTHONPATH=src python scripts/01_data_audit.py --synthetic     # runs on the synthetic stand-in

Writes aggregate counts only (reports/tables/data_audit.json); no pitcher-level values are written.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vald import data

ROOT = Path(__file__).resolve().parents[1]


def source_path(synthetic: bool) -> Path:
    if synthetic:
        return ROOT / "data" / "synthetic_master.csv"
    return Path(os.environ["VALD_DATA"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    d = data.load(source_path(a.synthetic))
    flags = data.audit_flags(d)
    c = data.clean(d)
    q = lambda s: {k: round(float(v), 2) for k, v in s.quantile([0, 0.25, 0.5, 0.75, 1]).items()}  # noqa: E731
    out = {
        "rows": int(len(d)),
        "unique_pitchers": int(d["player_id"].nunique()),
        "static_tests_in_table": int(sum(x in d.columns for x in data.VALD_ALL)),
        "corrupt_rows": int(flags["corrupt"].sum()),
        "corrupt_rows_detail": {
            "release_speed_under_60": int(d[data.SPEED].lt(data.MIN_SPEED_MPH).sum()),
            "height_over_3": int(d["height"].gt(data.MAX_HEIGHT_M).sum()),
        },
        "placeholder_mass_rows": int(flags["placeholder_mass"].sum()),
        "missing_release_speed_after_cleaning": int(c[data.SPEED].isna().sum()),
        "rows_after_cleaning": int(len(c)),
        "rows_with_usable_speed": int(c[data.SPEED].notna().sum()),
        "rows_with_usable_torque": int(c[data.TORQUE].notna().sum()),
        "raw_spread": {
            "mass_kg": q(d["mass"]),
            "height_m": q(d["height"]),
            "release_speed_mph": q(d[data.SPEED]),
            "elbow_torque_nm": q(d[data.TORQUE]),
        },
        "clean_spread": {
            "mass_kg": q(c["mass"].dropna()),
            "height_m": q(c["height"]),
            "release_speed_mph": q(c[data.SPEED].dropna()),
            "elbow_torque_nm": q(c[data.TORQUE].dropna()),
        },
        "static_test_missing_share": {k: round(float(c[k].isna().mean()), 3) for k in data.VALD_ALL if k in c.columns},
        "usable_hip_tests": data.usable_predictors(c, [x for x in data.VALD_HIP if x in c.columns]),
        "usable_shoulder_tests": data.usable_predictors(c, [x for x in data.VALD_SHOULDER if x in c.columns]),
        "hip_mocap_targets": len(data.mocap_targets(c, data.VALD_ALL)),
        "left_handed": int(c["throws_left"].sum()),
        "mlb_level": int(c["level_mlb"].sum()),
        "capture_labels": c["pitch_type"].str.replace(r"^(exp|slidestep)_", "", regex=True).value_counts().to_dict(),
    }
    # How badly the uncleaned table distorts a naive look at the outcome: the elbow-torque scale clusters.
    out["torque_scale_clusters_raw"] = {
        "under_1_nm": int(d[data.TORQUE].lt(1).sum()),
        "1_to_60_nm": int(d[data.TORQUE].between(1, 60, inclusive="left").sum()),
        "over_60_nm": int(d[data.TORQUE].ge(60).sum()),
    }
    path = ROOT / "reports" / "tables" / ("data_audit_synthetic.json" if a.synthetic else "data_audit.json")
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if not isinstance(v, dict)}, indent=2, default=lambda o: int(o)))
    _ = np  # keep numpy imported for dtype handling in json


if __name__ == "__main__":
    main()

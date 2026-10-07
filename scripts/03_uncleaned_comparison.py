"""What the audit prevents: the same models on the table with the corrupt and placeholder rows left in.

    VALD_DATA=path/to/master.csv PYTHONPATH=src python scripts/03_uncleaned_comparison.py
    PYTHONPATH=src python scripts/03_uncleaned_comparison.py --synthetic

The raw table mixes three scales of elbow torque (about 0.1, about 25-45, and about 50-280 Nm) because of capture
errors, not physiology. A model only has to recognize which scale a row is on to look accurate.
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
from vald.models import cv_r2, make_folds

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    path = ROOT / "data" / "synthetic_master.csv" if a.synthetic else Path(os.environ["VALD_DATA"])
    d = data.load(path)
    d["mass_missing"] = 0.0  # the raw analysis had no such flag: placeholder mass is just "1 kg"
    d["throws_left"] = data.throws_left(d["pitch_type"])
    d["level_mlb"] = d["level"].str.lower().eq("mlb").astype(float)
    base = ["height", "mass", "mass_missing", "throws_left", "level_mlb"]
    hip = data.usable_predictors(d, data.VALD_HIP)
    out = {}
    for outcome in (data.TORQUE, data.SPEED):
        r = d[d[outcome].notna()].reset_index(drop=True)
        y = r[outcome].to_numpy(float)
        folds = make_folds(len(r), 10, 20)
        res = {"n": int(len(r)), "outcome_sd": round(float(np.std(y, ddof=1)), 2)}
        for name, cols in (("body_size_and_handedness", base), ("plus_hip_tests", base + hip)):
            r2, (p, _) = cv_r2(r[cols].to_numpy(float), y, folds)
            res[name] = {"cv_r2": round(r2, 3), "rmse": round(float(np.sqrt(np.mean((y - p) ** 2))), 2)}
        out[outcome] = res
    path_out = (
        ROOT / "reports" / "tables" / ("uncleaned_synthetic.json" if a.synthetic else "uncleaned_comparison.json")
    )
    path_out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

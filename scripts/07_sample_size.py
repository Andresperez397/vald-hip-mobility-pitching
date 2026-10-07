"""How many pitchers would a follow-up study need?

    VALD_DATA=path/to/master.csv python scripts/07_sample_size.py
    python scripts/07_sample_size.py --synthetic

Same simulation as 06_power.py (a planted gain from the static tests, the project's decision rule), but the
cohort is enlarged. Each simulated pitcher is a real pitcher's body-size covariates and static tests drawn with
replacement, with the static tests jittered by 25% of their standard deviation so no two simulated pitchers are
identical. The outcome is rebuilt from the planted effect plus fresh noise, so the only thing borrowed from the
real data is the realistic joint distribution of the predictors. Results are a planning guide, not a guarantee.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vald import data, models  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ["height", "mass", "mass_missing", "throws_left", "level_mlb"]
SIZES = [100, 150, 200, 300, 450]
DELTAS = [0.05, 0.10, 0.15]
SIMS, BOOT = 150, 300


def simulate(d, outcome: str, n: int, delta: float, base_r2: float, rng, sims: int = SIMS, boot: int = BOOT) -> float:
    d = d[d[outcome].notna()].reset_index(drop=True)
    hip = data.usable_predictors(d, data.VALD_HIP)
    sho = data.usable_predictors(d, data.VALD_SHOULDER)
    Xb0 = d[BASELINE].to_numpy(float)
    V0 = d[hip + sho].to_numpy(float)
    Xb0 = np.where(np.isnan(Xb0), np.nanmedian(Xb0, axis=0), Xb0)
    med = np.nanmedian(V0, axis=0)
    sd = np.nanstd(V0, axis=0)
    hits = 0
    folds = models.make_folds(n, 10, 2)
    for _ in range(sims):
        i = rng.integers(0, len(d), n)
        Xb = Xb0[i].copy()
        V = V0[i] + rng.normal(0, 0.25, (n, V0.shape[1])) * sd  # jitter; NaN stays NaN
        Vi = np.where(np.isnan(V), med, V)
        Vi = np.clip(Vi, np.percentile(Vi, 2.5, axis=0), np.percentile(Vi, 97.5, axis=0))
        Vs = (Vi - Vi.mean(0)) / Vi.std(0)
        # body-size signal from a fixed random direction (so the baseline carries `base_r2`)
        zb = Xb @ rng.normal(size=Xb.shape[1])
        zb = (zb - zb.mean()) / zb.std()
        zv = Vs @ rng.normal(size=Vs.shape[1])
        zv = zv - zb * (zb @ zv) / (zb @ zb)
        zv = (zv - zv.mean()) / zv.std()
        y = np.sqrt(base_r2) * zb + np.sqrt(delta) * zv + np.sqrt(1 - base_r2 - delta) * rng.normal(size=n)
        Xf = np.hstack([Xb, V])
        rb, (pb, m) = models.cv_r2(Xb, y, folds)
        _, (pf, _) = models.cv_r2(Xf, y, folds)
        lo, _, _ = models.bootstrap_gain(y, pf, pb, m, reps=boot, seed=int(rng.integers(1e9)))
        hits += lo > 0
    return hits / sims


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    path = ROOT / "data" / "synthetic_master.csv" if a.synthetic else Path(os.environ["VALD_DATA"])
    c = data.clean(data.load(path))
    rng = np.random.default_rng(2026)
    out = {"release_speed": {}, "elbow_torque": {}, "settings": {"sims": SIMS, "bootstrap": BOOT, "cv": "10-fold x 2"}}
    for outcome, key, base in ((data.SPEED, "release_speed", 0.125), (data.TORQUE, "elbow_torque", 0.255)):
        for delta in DELTAS:
            out[key][str(delta)] = {str(n): round(simulate(c, outcome, n, delta, base, rng), 2) for n in SIZES}
            print(key, delta, out[key][str(delta)], flush=True)
    name = "sample_size_synthetic.json" if a.synthetic else "sample_size.json"
    (ROOT / "reports" / "tables" / name).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

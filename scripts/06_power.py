"""How big would a real effect of the static tests have to be for this design to see it?

    VALD_DATA=path/to/master.csv python scripts/06_power.py
    python scripts/06_power.py --synthetic

Plants a known effect: the outcome is rebuilt as  sqrt(r2_base) * (body-size signal) + sqrt(delta) * (a random
combination of the usable static tests, uncorrelated with body size) + noise, scaled to the real outcome's spread. The
real predictor matrix and the real sample are used, so missingness, correlation between tests, and n are all realistic.
For each planted gain `delta` it repeats the analysis (10-fold CV x 3 repeats, ridge, pitcher bootstrap) and records how
often the 95% interval for the gain over body size excludes zero, which is this project's decision rule.
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
DELTAS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
SIMS, BOOT = 120, 400


def one_design(d, outcome: str, rng, deltas=None, sims=None, boot=None) -> dict:
    deltas, sims, boot = deltas or DELTAS, sims or SIMS, boot or BOOT
    d = d[d[outcome].notna()].reset_index(drop=True)
    hip = data.usable_predictors(d, data.VALD_HIP)
    sho = data.usable_predictors(d, data.VALD_SHOULDER)
    Xb = d[BASELINE].to_numpy(float)
    Xf = d[BASELINE + hip + sho].to_numpy(float)
    y0 = d[outcome].to_numpy(float)
    n = len(d)
    # baseline signal: in-sample fitted values from body size, standardized, with the observed baseline R^2 rescaled
    Xb_i = np.where(np.isnan(Xb), np.nanmedian(Xb, axis=0), Xb)
    A = np.column_stack([np.ones(n), Xb_i])
    fit = A @ np.linalg.lstsq(A, y0, rcond=None)[0]
    zb = (fit - fit.mean()) / fit.std()
    V = d[hip + sho].to_numpy(float)
    V = np.where(np.isnan(V), np.nanmedian(V, axis=0), V)
    V = np.clip(V, np.percentile(V, 2.5, axis=0), np.percentile(V, 97.5, axis=0))
    V = (V - V.mean(0)) / V.std(0)
    folds = models.make_folds(n, 10, 3)
    out = {}
    base_r2 = 0.125 if outcome == data.SPEED else 0.255  # observed baseline R^2 (results.json)
    for delta in deltas:
        hits = 0
        gains = []
        for _ in range(sims):
            w = rng.normal(size=V.shape[1])
            zv = V @ w
            zv = zv - zb * (zb @ zv) / (zb @ zb)  # uncorrelated with the body-size signal
            zv = (zv - zv.mean()) / zv.std()
            eps = rng.normal(size=n)
            y = np.sqrt(base_r2) * zb + np.sqrt(delta) * zv + np.sqrt(max(1 - base_r2 - delta, 0.01)) * eps
            y = y * y0.std() + y0.mean()
            rb, (pb, m) = models.cv_r2(Xb, y, folds)
            rf, (pf, _) = models.cv_r2(Xf, y, folds)
            lo, hi_, _ = models.bootstrap_gain(y, pf, pb, m, reps=boot, seed=int(rng.integers(1e9)))
            gains.append(rf - rb)
            hits += lo > 0
        out[str(delta)] = {"power": round(hits / sims, 2), "mean_observed_gain": round(float(np.mean(gains)), 3)}
    return {"n": int(n), "tests_used": len(hip) + len(sho), "by_planted_gain": out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    path = ROOT / "data" / "synthetic_master.csv" if a.synthetic else Path(os.environ["VALD_DATA"])
    c = data.clean(data.load(path))
    rng = np.random.default_rng(2026)
    res = {"release_speed": one_design(c, data.SPEED, rng), "elbow_torque": one_design(c, data.TORQUE, rng),
           "settings": {"sims_per_gain": SIMS, "bootstrap": BOOT, "cv": "10-fold x 3", "seed": 2026}}  # fmt: skip
    name = "power_synthetic.json" if a.synthetic else "power.json"
    (ROOT / "reports" / "tables" / name).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

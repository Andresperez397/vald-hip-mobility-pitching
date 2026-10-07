"""Run both questions and write aggregate tables to reports/tables.

    VALD_DATA=path/to/master.csv PYTHONPATH=src python scripts/02_run_analysis.py
    PYTHONPATH=src python scripts/02_run_analysis.py --synthetic

Question 1: do static VALD tests predict release speed and peak elbow varus torque for a pitcher the model has
            not seen, beyond body size?  (10-fold CV x 20 repeats, ridge)
Question 2: do static hip tests explain how a pitcher's hip and pelvis move in the delivery, across ~240
            motion-capture measurements?  (5-fold CV x 3 repeats, permutation null for the best target)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vald import data, models
from vald.models import cv_r2, make_folds

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ["height", "mass", "mass_missing", "throws_left", "level_mlb"]
Q1_FOLDS, Q1_REPEATS = 10, 20
Q2_FOLDS, Q2_REPEATS, Q2_PERMS = 5, 3, 200


def rmse(y, p):
    ok = ~np.isnan(y)
    return float(np.sqrt(np.mean((y[ok] - p[ok]) ** 2)))


def question1(c, outcome: str) -> dict:
    d = c[c[outcome].notna()].reset_index(drop=True)
    y = d[outcome].to_numpy(float)
    hip = data.usable_predictors(d, data.VALD_HIP)
    sho = data.usable_predictors(d, data.VALD_SHOULDER)
    sets = {
        "body_size_and_handedness": BASELINE,
        "plus_hip_tests": BASELINE + hip,
        "plus_hip_and_shoulder_tests": BASELINE + hip + sho,
    }
    folds = make_folds(len(d), Q1_FOLDS, Q1_REPEATS)
    res: dict = {"n": int(len(d)), "outcome_sd": round(float(np.std(y, ddof=1)), 2), "models": {}}
    first = {}
    for name, cols in sets.items():
        X = d[cols].to_numpy(float)
        mean_r2, (p, m) = cv_r2(X, y, folds)
        res["models"][name] = {"predictors": len(cols), "cv_r2": round(mean_r2, 3), "rmse": round(rmse(y, p), 2)}
        first[name] = (p, m)
    res["models"]["mean_only"] = {
        "predictors": 0,
        "cv_r2": 0.0,
        "rmse": round(rmse(y, first["body_size_and_handedness"][1]), 2),
    }
    base = first["body_size_and_handedness"]
    for name in ("plus_hip_tests", "plus_hip_and_shoulder_tests"):
        lo, hi, _ = models.bootstrap_gain(y, first[name][0], base[0], base[1])
        res["models"][name]["gain_over_body_size"] = round(
            res["models"][name]["cv_r2"] - res["models"]["body_size_and_handedness"]["cv_r2"], 3
        )
        res["models"][name]["gain_ci_first_repeat"] = [round(lo, 3), round(hi, 3)]
    lo, hi, _ = models.bootstrap_gain(y, base[0], base[1], base[1])
    res["models"]["body_size_and_handedness"]["gain_over_mean_ci_first_repeat"] = [round(lo, 3), round(hi, 3)]
    # Simple, readable companion: the correlation of each static test with the outcome (shown, not selected on).
    cors = {k: round(float(d[k].corr(d[outcome], method="spearman")), 2) for k in hip + sho}
    res["spearman_with_outcome"] = dict(sorted(cors.items(), key=lambda kv: -abs(kv[1])))
    res["spearman_n_by_test"] = {k: int(d[k].notna().sum()) for k in hip + sho}
    return res


def question2(c) -> dict:
    hip = data.usable_predictors(c, data.VALD_HIP)
    targets = data.mocap_targets(c, data.VALD_ALL)
    d = c.reset_index(drop=True)
    Xb = d[BASELINE].to_numpy(float)
    V = d[hip].to_numpy(float)
    Y = d[targets].to_numpy(float)
    folds = make_folds(len(d), Q2_FOLDS, Q2_REPEATS)
    gain = models.target_gain(Xb, np.hstack([Xb, V]), Y, folds)
    # R^2 of baseline and of full model per target (first repeat), for the table
    pb, m = models.oof_predictions(Xb, Y, folds[0])
    pf, _ = models.oof_predictions(np.hstack([Xb, V]), Y, folds[0])
    r2b, r2f = models.r2(Y, pb, m), models.r2(Y, pf, m)
    null = models.permutation_null(Xb, V, Y, folds, perms=Q2_PERMS)
    cutoff = float(np.percentile(null, 95))
    order = np.argsort(-gain)
    top = [
        {
            "target": targets[i],
            "r2_body_size": round(float(r2b[i]), 3),
            "r2_with_hip_tests": round(float(r2f[i]), 3),
            "gain": round(float(gain[i]), 3),
        }
        for i in order[:12]
    ]
    return {
        "n_pitchers": int(len(d)),
        "hip_tests": hip,
        "targets": len(targets),
        "median_gain": round(float(np.median(gain)), 3),
        "share_of_targets_with_positive_gain": round(float(np.mean(gain > 0)), 3),
        "best_gain": round(float(gain.max()), 3),
        "null_best_gain_median": round(float(np.median(null)), 3),
        "null_best_gain_p95": round(cutoff, 3),
        "targets_beating_null_p95": int(np.sum(gain > cutoff)),
        "permutations": Q2_PERMS,
        "p_value_best_target": round(float((1 + np.sum(null >= gain.max())) / (1 + len(null))), 4),
        "top_targets": top,
        "gain_all_targets": [round(float(g), 4) for g in gain],
        "null_best_gains": [round(float(g), 4) for g in null],
        "gain_quantiles": {k: round(float(np.quantile(gain, k)), 3) for k in (0.05, 0.25, 0.5, 0.75, 0.95)},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    import os

    path = ROOT / "data" / "synthetic_master.csv" if a.synthetic else Path(os.environ["VALD_DATA"])
    c = data.clean(data.load(path))
    out = {
        "release_speed": question1(c, data.SPEED),
        "elbow_torque": question1(c, data.TORQUE),
        "hip_mocap": question2(c),
        "settings": {
            "q1": f"{Q1_FOLDS}-fold CV x {Q1_REPEATS} repeats, ridge (penalty chosen inside each training fold)",
            "q2": f"{Q2_FOLDS}-fold CV x {Q2_REPEATS} repeats, ridge; {Q2_PERMS} permutations of the static tests",
            "seed": 2026,
        },
    }
    name = "results_synthetic.json" if a.synthetic else "results.json"
    (ROOT / "reports" / "tables" / name).write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("release_speed", "elbow_torque")}, indent=1)[:3500])
    print(
        json.dumps(
            {
                k: v
                for k, v in out["hip_mocap"].items()
                if k not in ("top_targets", "gain_all_targets", "null_best_gains")
            },
            indent=1,
        )
    )
    print(json.dumps(out["hip_mocap"]["top_targets"][:6], indent=1))


if __name__ == "__main__":
    main()

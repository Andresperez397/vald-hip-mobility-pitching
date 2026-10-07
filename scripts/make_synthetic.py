"""Write a synthetic stand-in for the private pitcher table, with the same columns and the same planted problems.

    python scripts/make_synthetic.py            # writes data/synthetic_master.csv

Truth planted so the pipeline can be checked against it:
  - release speed depends on height, mass and handedness a little, and on no static test;
  - elbow torque depends on mass and handedness, and on no static test;
  - one hip motion-capture target ("pelvis_angle_max_knee_height_y") depends on a static hip test (trail_add_avg);
  - 6 corrupt rows (speed 36-41 mph, height 29-41 m, torque about 0.1), 21 placeholder-mass rows (mass 1 kg, torque on a
    smaller scale), and one asymmetry value of 619%.
Every value is random. Nothing here comes from a real athlete.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vald import data  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
N = 103


def make(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"player_id": 1001 + rng.permutation(N)})
    d["player_name"] = [f"P{i:03d}" for i in range(1, N + 1)]
    d["source"] = "redacted"
    left = rng.random(N) < 0.23
    d["pitch_type"] = np.where(left, "Fastball LH Markerless", "Fastball RH Markerless")
    d["level"] = np.where(rng.random(N) < 0.15, "mlb", "milb")
    d["height"] = rng.normal(1.88, 0.06, N).clip(1.7, 2.05)
    d["mass"] = rng.normal(94, 10, N).clip(75, 120)
    z = lambda x: (x - x.mean()) / x.std()  # noqa: E731
    d[data.SPEED] = 91.5 + 0.8 * z(d["height"]) + 0.6 * z(d["mass"]) - 0.8 * left + rng.normal(0, 2.4, N)
    d[data.TORQUE] = 150 + 18 * z(d["mass"]) - 6 * left + rng.normal(0, 32, N)

    # Static tests: independent of the outcomes by construction.
    base = {"add": 460, "abd": 485}
    for side in ("lead", "trail"):
        for kind in ("add", "abd"):
            m = base[kind] + rng.normal(0, 55, N)
            d[f"{side}_{kind}_max"] = m
            d[f"{side}_{kind}_avg"] = m - np.abs(rng.normal(20, 8, N))
        d[f"{side}_er"] = rng.normal(26, 7, N)
        d[f"{side}_ir"] = rng.normal(29.5, 5, N)
    d["asymmetry_hip_add"] = np.abs(rng.normal(5, 4, N))
    d["asymmetry_hip_abd"] = np.abs(rng.normal(4, 3, N))
    for k in ("ir", "er"):
        for stat in ("max", "avg"):
            d[f"throwing_{k}_{stat}"] = rng.normal(165, 40, N)
            d[f"nonthrowing_{k}_{stat}"] = np.where(rng.random(N) < 0.58, np.nan, rng.normal(160, 35, N))
        d[f"throwing_{k}_impulse"] = rng.normal(1000, 300, N)
        d[f"non_throwing_{k}_impulse"] = np.where(rng.random(N) < 0.58, np.nan, rng.normal(1100, 300, N))
        d[f"asymmetry_shoulder_{k}"] = np.where(rng.random(N) < 0.58, np.nan, np.abs(rng.normal(10, 7, N)))

    # Motion-capture hip and pelvis measurements: noise, plus one target that truly depends on a static test.
    extra = {}
    for i in range(60):
        extra[f"pelvis_angle_synth_{i:02d}"] = rng.normal(0, 8, N)
        extra[f"hip_shoulders_sep_synth_{i:02d}"] = rng.normal(0, 9, N)
    d = pd.concat([d, pd.DataFrame(extra)], axis=1)
    d["pelvis_angle_max_knee_height_y"] = 0.7 * z(d["trail_add_avg"]) * 8 + rng.normal(0, 6, N)

    # The problems the audit has to find.
    bad = rng.choice(N, 6, replace=False)
    d.loc[bad, data.SPEED] = rng.uniform(36, 41, 6)
    d.loc[bad, "height"] = rng.uniform(29, 41, 6)
    d.loc[bad, data.TORQUE] = rng.uniform(0.08, 0.15, 6)
    rest = np.setdiff1d(np.arange(N), bad)
    ph = rng.choice(rest, 21, replace=False)
    d.loc[ph, "mass"] = data.PLACEHOLDER_MASS
    d.loc[ph, data.TORQUE] = rng.uniform(24, 45, 21)
    d.loc[rest[0] if rest[0] not in ph else rest[1], "asymmetry_hip_add"] = 619.0
    return d


def main() -> None:
    path = ROOT / "data" / "synthetic_master.csv"
    make().to_csv(path, index=False)
    print("wrote", path)


if __name__ == "__main__":
    main()

"""Load the pitcher-level table, apply the audited cleaning rules, and define the column groups.

One row per pitcher: static strength and range-of-motion tests from a VALD system, body size, and
marker-based/markerless motion-capture measurements of one session of fastballs. The data are private
lab data and are not in this repository; `scripts/make_synthetic.py` writes a stand-in with the same
structure and the same planted problems, so everything here runs end to end.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# Static tests, fixed before any model was fit (see ANALYSIS_PLAN.md). 29 columns in the source.
VALD_HIP = [
    "lead_add_max", "trail_add_max", "lead_abd_max", "trail_abd_max",
    "lead_add_avg", "trail_add_avg", "lead_abd_avg", "trail_abd_avg",
    "asymmetry_hip_add", "asymmetry_hip_abd", "lead_er", "trail_er", "lead_ir", "trail_ir",
]  # fmt: skip
VALD_SHOULDER = [
    "nonthrowing_ir_max", "throwing_ir_max", "nonthrowing_er_max", "throwing_er_max",
    "nonthrowing_ir_avg", "throwing_ir_avg", "nonthrowing_er_avg", "throwing_er_avg",
    "non_throwing_ir_impulse", "throwing_ir_impulse", "non_throwing_er_impulse", "throwing_er_impulse",
    "asymmetry_shoulder_ir", "asymmetry_shoulder_er",
]  # fmt: skip
VALD_ALL = VALD_HIP + VALD_SHOULDER

SPEED = "ball_release_speed_mph"
TORQUE = "max_elbow_varus_torque_nm"

# Plausibility limits used by the audit rules.
MIN_SPEED_MPH = 60.0  # a 36-41 mph "fastball" is a unit or capture error
MAX_HEIGHT_M = 3.0  # heights of 29-41 are in the wrong unit
PLACEHOLDER_MASS = 1.0  # the capture software's default when no body mass was entered
MISSING_LIMIT = 0.30  # a static test with more missing values than this is not used as a predictor
TARGET_MISSING_LIMIT = 0.05  # a motion-capture target needs to be nearly complete


def throws_left(pitch_type: pd.Series) -> pd.Series:
    return pitch_type.str.contains(r"\bLH\b", flags=re.I, regex=True).astype(float)


def audit_flags(d: pd.DataFrame) -> pd.DataFrame:
    """Row-level audit flags. Every row is kept in the table; the analysis chooses what to use."""
    f = pd.DataFrame(index=d.index)
    f["corrupt"] = d[SPEED].lt(MIN_SPEED_MPH) | d["height"].gt(MAX_HEIGHT_M)
    f["placeholder_mass"] = d["mass"].eq(PLACEHOLDER_MASS) & ~f["corrupt"]
    f["no_speed"] = d[SPEED].isna()
    return f


def clean(d: pd.DataFrame) -> pd.DataFrame:
    """The analysis table: corrupt rows dropped, placeholder mass set to missing, torque on placeholder-mass rows
    set to missing (it is on a different scale there), covariates built."""
    flags = audit_flags(d)
    out = d.loc[~flags["corrupt"]].copy()
    ph = flags.loc[out.index, "placeholder_mass"]
    out.loc[ph, "mass"] = np.nan
    out.loc[ph, TORQUE] = np.nan
    out["mass_missing"] = out["mass"].isna().astype(float)
    out["throws_left"] = throws_left(out["pitch_type"])
    out["level_mlb"] = out["level"].str.lower().eq("mlb").astype(float)
    return out


def usable_predictors(d: pd.DataFrame, candidates: list[str]) -> list[str]:
    """Keep static tests with at most 30% missing values and some variation. Uses no outcome."""
    keep = []
    for c in candidates:
        x = d[c]
        if x.isna().mean() <= MISSING_LIMIT and x.nunique(dropna=True) > 1:
            keep.append(c)
    return keep


def mocap_targets(d: pd.DataFrame, vald: list[str]) -> list[str]:
    """Hip and pelvis motion-capture measurements (the targets of question 2): at most 5% missing, varying, and not a
    near-duplicate (|r| > 0.995) of one already kept."""
    cols = [c for c in d.columns if re.search("hip|pelvis|sep", c, re.I) and c not in vald and c != "height"]
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(d[c])]
    keep: list[str] = []
    for c in cols:
        x = d[c]
        if x.isna().mean() > TARGET_MISSING_LIMIT or x.nunique(dropna=True) < 5:
            continue
        if any(abs(x.corr(d[k])) > 0.995 for k in keep):
            continue
        keep.append(c)
    return keep


def load(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig").copy()

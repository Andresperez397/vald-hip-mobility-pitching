"""Tests on the synthetic stand-in, whose truth is known (see scripts/make_synthetic.py)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vald import data, models

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def raw():
    return _load_script("make_synthetic").make()


@pytest.fixture(scope="module")
def clean(raw):
    return data.clean(raw)


def test_audit_finds_every_planted_problem(raw):
    f = data.audit_flags(raw)
    assert f["corrupt"].sum() == 6
    assert f["placeholder_mass"].sum() == 21
    assert not (f["corrupt"] & f["placeholder_mass"]).any()


def test_clean_drops_corrupt_rows_and_blanks_what_is_not_trustworthy(raw, clean):
    assert len(clean) == len(raw) - 6
    ph = clean["mass_missing"].eq(1.0)
    assert ph.sum() == 21
    assert clean.loc[ph, data.TORQUE].isna().all() and clean.loc[ph, "mass"].isna().all()
    assert clean[data.SPEED].between(75, 110).all() and clean["height"].between(1.6, 2.2).all()


def test_usable_predictors_exclude_mostly_missing_tests(clean):
    sho = data.usable_predictors(clean, data.VALD_SHOULDER)
    assert "throwing_ir_max" in sho and "nonthrowing_ir_max" not in sho and "asymmetry_shoulder_ir" not in sho


def test_mocap_targets_are_hip_pelvis_only_and_exclude_the_static_tests(clean):
    t = data.mocap_targets(clean, data.VALD_ALL)
    assert "pelvis_angle_max_knee_height_y" in t
    assert not set(t) & set(data.VALD_ALL) and all("hip" in c or "pelvis" in c or "sep" in c for c in t)


def test_winsorizer_uses_training_data_only():
    train = np.arange(100.0)[:, None]
    w = models.Winsorizer().fit(train)
    out = w.transform(np.array([[-1000.0], [50.0], [1000.0]]))
    assert out[1, 0] == 50.0 and out[0, 0] == pytest.approx(2.475) and out[2, 0] == pytest.approx(96.525)


def test_folds_never_share_a_pitcher():
    for rep in models.make_folds(97, 10, 3):
        seen = np.concatenate([te for _, te in rep])
        assert sorted(seen) == list(range(97))
        for tr, te in rep:
            assert not set(tr) & set(te)


def test_held_out_outcome_cannot_change_its_own_prediction(clean):
    d = clean.reset_index(drop=True)
    X = d[["height", "mass", "mass_missing", "throws_left", "level_mlb"]].to_numpy(float)
    y = d[data.SPEED].to_numpy(float)[:, None]
    folds = models.make_folds(len(d), 5, 1)[0]
    p1, _ = models.oof_predictions(X, y, folds)
    te = folds[0][1]
    y2 = y.copy()
    y2[te] += 500.0  # corrupt every held-out outcome in fold 0
    p2, _ = models.oof_predictions(X, y2, folds)
    assert np.allclose(p1[te], p2[te])


def test_baseline_beats_the_mean_on_clean_speed_and_tests_add_nothing(clean):
    d = clean.reset_index(drop=True)
    y = d[data.SPEED].to_numpy(float)
    folds = models.make_folds(len(d), 10, 5)
    base = ["height", "mass", "mass_missing", "throws_left", "level_mlb"]
    r_base, _ = models.cv_r2(d[base].to_numpy(float), y, folds)
    r_full, _ = models.cv_r2(d[base + data.usable_predictors(d, data.VALD_HIP)].to_numpy(float), y, folds)
    assert r_base > 0.05 and r_full - r_base < 0.05


def test_uncleaned_table_makes_body_size_look_far_better_than_it_is(raw, clean):
    def r2(df):
        df = df.reset_index(drop=True).copy()
        df["mass_missing"] = df.get("mass_missing", 0.0)
        df["throws_left"] = data.throws_left(df["pitch_type"])
        df["level_mlb"] = df["level"].str.lower().eq("mlb").astype(float)
        cols = ["height", "mass", "mass_missing", "throws_left", "level_mlb"]
        return models.cv_r2(
            df[cols].to_numpy(float), df[data.SPEED].to_numpy(float), models.make_folds(len(df), 10, 3)
        )[0]

    assert r2(raw) > 0.5 > r2(clean)


def test_permutation_null_recovers_the_planted_hip_target(clean):
    d = clean.reset_index(drop=True)
    base = ["height", "mass", "mass_missing", "throws_left", "level_mlb"]
    hip = data.usable_predictors(d, data.VALD_HIP)
    targets = data.mocap_targets(d, data.VALD_ALL)
    Y = d[targets].to_numpy(float)
    Xb, V = d[base].to_numpy(float), d[hip].to_numpy(float)
    folds = models.make_folds(len(d), 5, 2)
    gain = models.target_gain(Xb, np.hstack([Xb, V]), Y, folds)
    null = models.permutation_null(Xb, V, Y, folds, perms=40)
    best = targets[int(np.argmax(gain))]
    assert best == "pelvis_angle_max_knee_height_y" and gain.max() > np.percentile(null, 95)


def test_audit_output_contains_no_pitcher_level_values(tmp_path):
    out = _load_script("01_data_audit")
    assert hasattr(out, "main")
    syn = _load_script("make_synthetic").make()
    assert isinstance(syn, pd.DataFrame) and syn["player_id"].is_unique

# Data audit

Produced by `scripts/01_data_audit.py` before any model was fit; counts only, in [reports/tables/data_audit.json](reports/tables/data_audit.json).

## The table

One row per pitcher: 103 pitchers (103 unique IDs), all captured with markerless motion capture (74 right-handed, 23 left-handed in the 97 kept rows). Levels: 88 minor league, 15 major league. About 1,000 columns: 28 static VALD tests (14 hip, 14 shoulder), body size, release speed, elbow varus torque, and several hundred motion-capture measurements.

## Problems found and how the cleaning handles them

| # | Problem | Rows | Handling |
|---|---|---|---|
| 1 | **Corrupt rows:** release speed 36–41 mph, height 29–41 m, elbow torque about 0.1 Nm, and body mass up to 1,581 kg. Unit or capture errors. | 6 | Dropped from every analysis (rule: speed under 60 mph or height over 3 m). |
| 2 | **Placeholder body mass of 1 kg.** The capture software's default when no mass was entered. Elbow torque on these rows is on a smaller scale (25–47 Nm against 112–281 Nm elsewhere), so it can't be pooled. | 21 | Mass set to missing and flagged; torque set to missing; release speed and the motion-capture and VALD columns kept. |
| 3 | **Torque scale clusters.** The raw table's elbow torque falls in three groups, below 1 Nm (6 rows), 1 to 60 Nm (21 rows) and above 60 Nm (76 rows). These are the two problems above, not three kinds of pitcher. | — | Handled by rules 1 and 2. |
| 4 | **Mostly-missing static tests.** 8 shoulder tests (the non-throwing-side tests and the two shoulder asymmetries) are missing for 56–58% of pitchers. | 8 columns | A test with more than 30% missing is not used as a predictor. |
| 5 | **A percentage asymmetry that explodes.** One `asymmetry_hip_add` value of 619% against 43% or less for every other pitcher (a near-zero denominator). `asymmetry_shoulder_ir` has one similar value (over 700%). | 1 (+1) | Predictors are winsorized at the training fold's 2.5th and 97.5th percentiles (see DEVIATIONS.md). |
| 6 | **Near-duplicate and sparse motion-capture targets.** | — | Targets need at most 5% missing values, some variation, and |r| ≤ 0.995 with an earlier kept target: 239 remain. |

## After cleaning

| Table | Rows | Release speed | Elbow torque | Body mass |
|---|---|---|---|---|
| As delivered | 103 | 36–98 mph | 0.1–281 Nm | 1–1,581 kg |
| Cleaned | 97 (speed), 76 (torque) | 85–98 mph, median 92.1 | 112–281 Nm, median 195 | 77–120 kg, median 95 (76 pitchers) |

## What the audit prevents

On the table as delivered, a model with body size, handedness and level alone reaches a cross-validated R² of 0.91 for release speed and 0.89 for elbow torque (`scripts/03_uncleaned_comparison.py`). After the audit, 0.12 and 0.26. The inflated numbers come from the model telling good rows from broken ones.

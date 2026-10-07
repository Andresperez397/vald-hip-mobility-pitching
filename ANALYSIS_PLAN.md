# Analysis plan

Written 2026-10-07 after the data audit and after the eight earlier exploratory reports on this data, and before the models in this repository were run. Because those reports existed, this is **not a pre-registration**: the questions and outcomes were already known to me. What the plan does fix is the design (below) before this repository's results existed. One change made after the first run is in [DEVIATIONS.md](DEVIATIONS.md).

## Questions

1. **Prediction:** do static VALD hip and shoulder tests predict release speed and peak elbow varus torque for a pitcher the model has not seen, beyond body size?
2. **Mechanism:** do static hip tests explain how a pitcher's hips and pelvis move in the delivery?

## Data and cleaning

As in [DATA_AUDIT.md](DATA_AUDIT.md). One row per pitcher.

## Predictor sets (fixed)

- **Baseline:** height, mass (with a missing indicator), throws left, level (major league). No test enters here.
- **Hip tests:** 14 (lead and trail hip adduction and abduction, peak and average; hip adduction and abduction asymmetry; lead and trail hip external and internal rotation).
- **Shoulder tests:** 14, of which 6 pass the 30% missing rule.

## Question 1

- **Outcomes:** release speed (97 pitchers) and peak elbow varus torque (76).
- **Models:** the training mean; baseline; baseline + hip tests; baseline + hip and shoulder tests. Ridge regression with winsorizing, median imputation and scaling fit inside each training fold, and the penalty chosen by efficient leave-one-out inside the fold.
- **Validation:** 10-fold cross-validation, 20 repeats, seed 2026.
- **Metric:** out-of-fold R² against the training-fold mean. The main comparison is the gain over the baseline model, with a 95% interval from 2,000 resamples of pitchers applied to the first repeat's predictions.
- **Decision rule:** a block of tests counts as informative only if the 95% interval of its gain over baseline excludes zero.

## Question 2

- **Targets:** all hip, pelvis and hip-shoulder separation motion-capture measurements that pass the missing, variation and duplicate rules (239).
- **Models:** baseline versus baseline + hip tests, ridge as above; 5-fold CV, 3 repeats.
- **Metric:** out-of-fold R² gain per target.
- **Multiplicity:** the largest gain across all targets is compared with the distribution of the largest gain when the hip tests are shuffled across pitchers (200 shuffles, baseline columns left in place). A target counts only if its gain beats the 95th percentile of that null.

## Checks

- The raw-table comparison (`03_uncleaned_comparison.py`).
- Tests on a synthetic table with planted problems and planted truth.

## What would change the conclusions

A larger sample, a second capture session per pitcher, or tests on the same pitchers' later injuries. With 97 pitchers and 25 correlated predictors, small effects are not detectable here: a simulation (`scripts/06_power.py`) puts the detectable gain at about 0.25 R² or more.

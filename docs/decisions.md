# Modelling Pipeline Decisions

## Research goal

The goal is not only to predict transport mode, but to test how well the model generalizes to unseen users.

Main questions:

- Can we predict transport mode from smartphone IMU data?
- How much does performance drop when testing on an unseen user?
- Which full pipeline performs best?

A pipeline is defined as:

```text
feature set + model + validation strategy
```

We compare complete pipelines, not isolated algorithms.

---

## Validation strategy

Three evaluation levels of increasing difficulty.

### 1. Random window split (E0)

- Easy optimistic baseline.
- Windows from the same session and user appear in train and test.
- Expected to overestimate performance because of 50% window overlap.

### 2. Session-based split (E1)

- Grouped k-fold by `session_id`, stratified by mode.
- All windows from one session stay in a single fold.
- Tests generalization to new recordings from known users.

### 3. Leave-One-Subject-Out (E2, LOSO)

Three folds with three participants:

```text
Fold 1: train on Athina + Xiomara, test on Babis
Fold 2: train on Babis + Xiomara, test on Athina
Fold 3: train on Babis + Athina, test on Xiomara
```

- `participant_id` is used for splitting only, never as a model feature.
- LOSO results conflate user, device, and sampling-rate shift. This is documented in the report rather than corrected.

---

## Data representation

Classical ML:

```text
raw sensor data
→ preprocessing
→ windowing
→ engineered features
→ model
```

TCN:

```text
raw sensor data
→ preprocessing
→ windowing
→ raw multichannel sequence tensor
→ TCN
```

Windowing: 5 seconds, 50% overlap, 50 Hz, giving 250 samples per window across 9 IMU channels (acc xyz, lin_acc xyz, gyro xyz).

---

## Feature engineering plan

Feature sets are designed as an ablation chain, not as independent alternatives:

```text
A ⊂ B ⊂ C
```

Each addition tests whether a feature family carries discriminative signal beyond the previous one.

### Feature Set A: time-domain core

Time-domain statistics on accelerometer axes, gyroscope axes, accelerometer magnitude, and gyroscope magnitude:

- mean, standard deviation, median, IQR, RMS, min, max, range

This is the minimum feature set and serves as the ablation baseline.

### Feature Set B: A + frequency-domain features

Frequency-domain features computed from FFT on linear acceleration magnitude and gyroscope magnitude:

- dominant frequency and its amplitude
- spectral centroid
- spectral entropy
- band powers: 0.2–1 Hz, 1–3 Hz, 3–8 Hz, 8–20 Hz

Motivation: walking and cycling have rhythmic signatures; vehicular modes have lower-frequency irregular vibration.

### Feature Set C: B + cross-axis and distribution-shape features

Adds:

- pairwise correlations between axes for accelerometer and gyroscope
- skewness and kurtosis on magnitudes
- jerk statistics (mean, std, max of absolute jerk)
- signal magnitude area

Motivation: these features target the public-transport cluster, where simple magnitude statistics are not enough.

### Selection inside Set C

Inside each LOSO fold:

- remove constant or near-constant features
- remove features with > 5% missing values
- remove highly correlated pairs (Pearson > 0.95)
- optional: permutation importance for diagnostic, not for selection in the main pipeline

Selection must be fitted on training users only. The held-out user remains unseen at all stages.

### Barometer ablation (separate)

Pressure features are excluded from the main feature table. A separate Babis + Xiomara subset analysis tests whether barometer features improve metro detection. This is reported as a side experiment, not as part of the main comparison.

---

## Model strategy

Four models, no LightGBM:

```text
01_baseline_majority
02_logistic_regression
03_random_forest
04_xgboost
```

Rationale:

- Majority baseline anchors the lower bound.
- Logistic regression anchors a linear model and tests whether complexity is needed.
- Random Forest provides interpretable feature importance.
- XGBoost is the modern gradient-boosting standard.

LightGBM is excluded because its expected difference from XGBoost is smaller than the LOSO fold-to-fold variance on this dataset, and including it would dilute the comparison without changing the conclusion.

### Preprocessing per model

- Tree models (RF, XGBoost): no standardization.
- Logistic regression: standardization fitted on training folds only.
- Imputation (if needed): median fitted on training folds only.

### Hyperparameter tuning: nested CV

LOSO results must not be inflated by tuning on the held-out user.

Protocol for each LOSO fold:

```text
outer fold: hold out one user
inner CV: grouped k-fold by session on the two training users
search: small grid over key hyperparameters
select: best inner-CV macro-F1
evaluate: single fit on training users, single evaluation on held-out user
```

Grid sizes are kept small (3–5 values per hyperparameter, 1–2 hyperparameters per model). Larger grids do not help with three users and a small dataset.

For Random splits and Session splits, the same nested protocol is used with the corresponding outer grouping.

---

## Main comparison table

Every cell is a single full pipeline:

| Pipeline | Random F1 | Session F1 | LOSO F1 | Drop R→S | Drop S→L |
|---|---:|---:|---:|---:|---:|
| Majority baseline | | | | | |
| LR + A | | | | | |
| RF + A | | | | | |
| RF + B | | | | | |
| RF + C | | | | | |
| XGBoost + A | | | | | |
| XGBoost + B | | | | | |
| XGBoost + C | | | | | |

Two drop columns instead of one:

- Random → Session captures session-level optimism (driven mainly by window overlap).
- Session → LOSO captures user-level generalization.

These answer different questions and both belong in the table.

---

## Selection criterion

Primary criterion, committed before running:

```text
mean LOSO macro-F1 across the three folds
tie-breaker: standard deviation across folds (lower is better)
```

Secondary diagnostics (used for discussion, not for selection):

- per-class precision and recall
- confusion matrix structure
- drop magnitude
- agreement between RF and XGBoost feature importance

Differences smaller than one LOSO-fold standard deviation are reported as ties, not as wins.

---

## Confusion matrices

Confusion matrices are a primary deliverable, not a supplementary one. The report will include:

- one detailed normalized confusion matrix for the winning pipeline on LOSO
- per-class macro-F1 breakdown across all pipelines as a secondary table
- short qualitative discussion of bus / metro / tram confusion

The interesting science of this project lives in the public-transport cluster. The confusion matrix is where that story is told.

---

## Ensemble

Soft-voting ensemble of RF + XGBoost (with their best feature sets and best hyperparameters) is run unconditionally:

```text
final probability = mean of RF and XGBoost class probabilities
```

This is cheap once the individual models exist. If the ensemble outperforms both base models on LOSO, it suggests complementary errors. If not, the strongest individual pipeline wins. Either outcome is informative.

---

## TCN plan

The TCN is a separate pipeline operating on raw sequences.

### Input

```text
shape: batch × channels × time_steps
example: 32 × 9 × 250
```

Channels are the same nine IMU signals used for feature engineering, standardized per channel using training-fold statistics only.

### Architecture (fixed before training)

- TCN backbone with dilated causal convolutions
- depth and dilation chosen to cover the full 250-sample receptive field
- dropout for regularization
- global pooling + linear classifier on the temporal output

### Training

- macro-F1 on a held-out validation portion of the training users as the early-stopping criterion
- Adam optimizer, fixed learning rate schedule
- batch size and epoch budget fixed before running

### Evaluation

Same validation strategy as classical models: random, session, LOSO. Same nested protocol for any tuning. The final report compares:

```text
best classical pipeline  vs  TCN
```

with explicit acknowledgement that classical models use hand-crafted features while TCN extracts features implicitly from raw sequences.

---

## Statistical noise

With three LOSO folds, fold-to-fold standard deviation is the relevant noise floor.

Reporting rules:

- always show per-fold results, not only aggregates
- mark differences below one standard deviation as ties
- do not claim a winner from a difference that vanishes inside the noise

---

## Final workflow

```text
1. Preprocess and align sensor data (done)
2. Window the data, build Feature Sets A, B, C
3. Train baseline, LR, RF, XGBoost across feature sets
4. Run nested CV for hyperparameter tuning
5. Build the comparison table with both drop columns
6. Select the winning pipeline by mean LOSO macro-F1
7. Run soft-voting ensemble of RF + XGBoost
8. Train and evaluate the TCN under the same validation logic
9. Compare best classical pipeline vs TCN vs ensemble
10. Run barometer ablation on Babis + Xiomara subset
11. Report results, drops, confusion matrices, and feature importance
```

---

## Report framing

Suggested wording:

> Because the research question concerns generalization as much as raw accuracy, each combination of feature set, model, and validation strategy was treated as a complete prediction pipeline. Pipelines were evaluated under random, session-based, and leave-one-subject-out splits in order to measure both classification performance and the performance drop for unseen users. Hyperparameter selection was performed inside each training fold to avoid leakage into the held-out evaluation.

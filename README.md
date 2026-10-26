[README.md](https://github.com/user-attachments/files/33970976/README.md)

# Transport Mode Detection from Smartphone Sensors

Classifying six transport modes (stationary, walking, cycling, bus, metro, tram) from smartphone IMU data, with a structured evaluation that degrades from random-window splits to leave-one-subject-out to external validation on unseen participants. Random Forest with 129 hand-crafted features reaches **macro-F1 = 0.57 under LOSO** and **0.60 on three completely new users**, while a Temporal Convolutional Network collapses to 0.18 — and counterintuitively gets *worse* with more training data per subject.

### Evaluation ladder

| Level | Description | RF + Set A | TCN (best) |
|-------|-------------|-----------|------------|
| E0 | Random window split | 0.90 | 0.95 |
| E1 | Session-grouped split | 0.64 | — |
| E2 | Leave-one-subject-out (LOSO) | 0.57 | 0.18 |
| E4 | External validation (3 new users) | 0.60 | — |

The drop from E0 to E2 is the generalization gap. For the Random Forest it's 0.33; for the TCN it's 0.77. That differential is the central empirical finding.

## The problem

Transport mode detection from phone sensors is a well-studied problem in human activity recognition, used in mobility analytics, context-aware systems, and personal activity tracking. What makes it harder than standard classification: sensor signals are noisy, motion patterns depend on how the phone is carried, and substantial variability exists across users, devices, and recording sessions. A model that scores 95% on random windows may fail completely on a new person with a different phone in a different pocket.

Most published results report only random-split accuracy, which inflates performance because overlapping windows from the same session leak into both train and test. This project evaluates every model at multiple difficulty levels, from easy (E0, random windows) to realistic (E2, leave-one-subject-out), to ground-truth deployment (E4, entirely new participants).

## Dataset

Data was collected specifically for this project using the [phyphox](https://phyphox.org/) smartphone sensing app. Three participants recorded six transport modes across three separate days each, producing 54 sessions of 8–12 minutes. Three additional participants contributed one full day each for external validation.

| Property | Value |
|----------|-------|
| Internal participants | 3 (different phone models) |
| External participants | 3 (unseen during training) |
| Transport modes | 6 |
| Sessions | 54 internal + 18 external |
| Sampling rate | 50 Hz (resampled from device-native rates) |
| Window size | 5 s with 50% overlap |
| Total windows | 14,513 internal + 5,274 external |
| Sensors | Accelerometer, linear accelerometer, gyroscope |

Phone placement (front trouser pocket) was specified but not strictly controlled. One participant's average linear-acceleration magnitude during dynamic modes was 1.8–2.0× higher than the others — a real carry-style difference, not a calibration error. The models had to handle this amplitude shift at test time.

Raw participant data is not included in this repository.

## Approach

### Feature engineering (129 features)

Each 5-second window (250 samples at 50 Hz) was converted to a feature vector across seven signal channels (linear accelerometer xyz, gyroscope xyz, and linear-acceleration magnitude). The raw accelerometer was excluded because its magnitude is dominated by gravity and encodes phone orientation rather than transport-related motion.

| Family | Count | Captures |
|--------|-------|----------|
| Time-domain basic | 72 | Mean, std, median, IQR, RMS, min, max, range |
| Frequency-domain | 20 | Dominant frequency, spectral entropy, centroid, band powers |
| Shape and energy | 9 | Skewness, kurtosis, energy summaries |
| Slope | 8 | Short-term increasing or decreasing trends |
| Cross-axis correlation | 7 | Co-movement between sensor axes |
| Axis std ratio | 6 | How motion energy distributes across axes |
| Peak count | 3 | Local maxima in selected signals |
| Signal magnitude area | 2 | Total movement intensity |
| Zero-crossing rate | 2 | Oscillation or sign changes |

Three feature sets were compared: Set A (80 time-domain features), Set B (A + 20 frequency features), and Set C (all 129). Under LOSO, Set A matched Set B while using 20 fewer features. Set C produced the highest random-split scores but the *lowest* LOSO scores — the extra features fit same-user windows better but failed to transfer across users.

### Random Forest

The winning pipeline uses a Random Forest with Set A (80 time-domain features), 300 trees, and session-grouped nested cross-validation for hyperparameter selection. Under LOSO, the held-out participant is never seen during tuning or training. Macro-F1 is the primary metric because the six classes are nearly balanced (largest-to-smallest ratio = 1.17).

The per-class structure tells a clear story: stationary, walking, and cycling are well-separated (F1 0.72–0.78), while bus, metro, and tram form a tight confusion block (F1 0.35–0.56). This asymmetry motivated the hierarchical cascade experiment.

### Hierarchical cascade

A four-level cascade decomposes the six-class problem into subproblems matched to the confusion structure: active vs. passive → walking vs. cycling / stationary vs. vehicle → bus vs. metro vs. tram. Each level uses a tailored feature representation (per-session normalization for the walking/cycling split, event-burstiness features for stationary/vehicle, multi-scale 15-second context for the vehicle subclasses).

Result: LOSO macro-F1 = 0.572, within noise of the flat baseline (0.568). The decomposition doesn't break through the ceiling because the public-transport bottleneck (bus/metro/tram) is sensor-bound, not method-bound. A barometer ablation confirmed this: adding pressure features moved the vehicle-subclass F1 by +0.031, the first evidence that the ceiling shifts with sensor coverage rather than modeling sophistication.

### Temporal Convolutional Network

The TCN receives raw 5-second multivariate sequences (9 channels × 250 timesteps) and learns temporal representations end-to-end. Three dilation schedules were tested, with receptive fields from 0.58 s to 2.50 s.

Under random splits, the TCN reaches 0.952 in 17 seconds — confirming the pipeline works. Under LOSO, all configurations collapse to 0.13–0.18. The drop of 0.77 macro-F1 (81% of the sanity-check score) is more than double the Random Forest's drop of 0.33. Per-window z-scoring, necessary to prevent cross-user leakage, strips the absolute amplitude information that was the Random Forest's strongest feature group. The TCN is left trying to classify from temporal *shape*, which turns out to be the most user-specific part of the signal.

## Key findings

**Subject diversity is the binding constraint, not data volume or model architecture.** A subject-count ablation (one day per participant, 3→6 subjects under LOSO) shows the Random Forest improving from 0.583 to 0.655 as the training pool grows from three to five subjects. More users help more than more sessions.

**The TCN gets worse with more data from the same subjects.** With one day per subject, the TCN reaches macro-F1 = 0.447. With three days per subject (3× the data), it drops to 0.184. Additional sessions sharpen user-specific signatures rather than diversifying the training distribution. This is user-specific memorization: the network learns *who* is moving rather than *how* they're moving. Hand-crafted features encoding physical priors (magnitude, spread, periodicity) transfer across users because they're designed to be amplitude-invariant. End-to-end sequence learning does not transfer at this dataset scale.

**The generalization gap is the real metric.** Reporting E0 accuracy (0.90+) without E2 or E4 overstates what the model can do in deployment. The 0.33 drop from E0 to E2 for the Random Forest — and the 0.77 drop for the TCN — quantifies how much of the apparent performance is just same-user leakage through overlapping windows. External validation (E4 = 0.60) confirms the LOSO estimate is calibrated: the model performs comparably on completely new people.

## Project structure

```
transport-mode-detection/
├── README.md
├── config.yaml                             # Sampling rate, window params, sensor list
├── requirements.txt
├── notebooks/
│   ├── 01_data_quality.ipynb               # File inventory, sampling rates, gap detection
│   ├── 02_preprocessing.ipynb              # Resampling to 50 Hz, merging, windowing
│   ├── 03_eda.ipynb                        # Magnitude distributions, FFT, cross-participant
│   ├── 04_feature_engineering.ipynb         # 129 features across 9 families
│   ├── 04b_normalized_features.ipynb       # Per-session normalized features for cascade
│   ├── 04c_multiscale_features.ipynb       # 15-second multi-scale context features
│   ├── 05_classical_ml.ipynb               # RF, XGBoost, LogReg across E0/E1/E2
│   ├── 06_tcn.ipynb                        # Temporal Convolutional Network
│   ├── 07_three_day_update.ipynb           # RF retuned on full 3-day data
│   ├── 08_hierarchical_cascade.ipynb       # Four-level cascade decomposition
│   ├── 09_external_validation.ipynb        # E4 on 3 unseen participants
│   └── 10_subject_ablation.ipynb           # Subject-count ablation (3→6 users)
├── src/
│   ├── features.py                         # Feature extraction functions (129 features)
│   ├── preprocessing.py                    # Resampling, gap handling, windowing
│   ├── data_loading.py                     # Session discovery and loading
│   └── smoke_test_features.py              # Feature pipeline validation
├── figures/                                # EDA plots, confusion matrices, ablation curves
├── docs/
│   ├── approach.md                         # Design decisions and methodology
│   ├── feature_engineering.md              # Feature family rationale
│   └── decisions.md                        # Modeling pipeline choices
├── data/                                   # Raw sensor data (not tracked)
└── outputs/                                # Generated features, models, results
```

## Quick start

```bash
git clone https://github.com/YOUR_USERNAME/transport-mode-detection.git
cd transport-mode-detection

pip install -r requirements.txt

# Place raw phyphox recordings in data/raw/
# Run notebooks in order: 01 → 10
```

**Requirements:** Python ≥ 3.10, pandas, numpy, scikit-learn, torch, matplotlib, seaborn, scipy, pywt.

**Data:** Raw participant recordings are not included for privacy. The pipeline reproduces all results from the raw phyphox CSV exports. Contact the author for data access.

## Limitations

Three internal participants is a small pool — typical HAR studies use 10–30. The external validation on three additional users provides some evidence of generalization, but the per-participant variance (external F1 ranges from 0.46 to 0.77) shows that the model's performance is still heavily user-dependent. Uncontrolled phone placement is the largest single source of cross-user variance. Future work should target a larger, more device-diverse participant pool and explicit modeling of phone placement as a domain factor.

## Acknowledgments

Developed collaboratively during my MSc in Business Analytics at Vrije Universiteit Amsterdam.

## License

MIT

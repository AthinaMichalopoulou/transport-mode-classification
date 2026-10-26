# Feature Engineering Design

## How the book maps to our design

Hoogendoorn and Funk (Chapter 4) define the feature engineering vocabulary we will use. Their framing has three families: time domain numerical features (means, standard deviations, slopes over a window of size $\lambda$), time domain categorical features (temporal patterns with succession and co-occurrence relations), and frequency domain features derived from the Discrete Fourier Transform.

Our problem only uses numerical sensors. The categorical temporal patterns from Section 4.1.2 do not apply because we have no categorical attributes to mine. Mixed data discretization from Section 4.1.3 is technically possible (we could bin magnitudes into low, medium, high and then look at sequences), but it would add complexity without addressing the hard problem in this dataset, which is the public transport cluster. We skip both and concentrate on time domain numerical features and frequency domain features.

The book's case study (CrowdSignals) uses 30 second windows for time domain features and 10 second windows for frequency features. We deviate from this for reasons specific to transport mode recognition, explained below.

---

## Window and overlap

Window length is 5 seconds (250 samples at 50 Hz). Overlap is 50% (step size 2.5 seconds).

Three justifications for the 5 second choice:

1. **Frequency resolution.** A 5 second window gives a frequency resolution of $f_0 = 1/5 = 0.2$ Hz. Walking cadence sits in the 1.5 to 2.5 Hz band. A 0.2 Hz resolution puts roughly five FFT bins inside that band, which is sufficient to locate the dominant frequency. Going below 5 seconds would smear the walking peak. Going above 5 seconds would average across longer behavioural variation without improving the bin spacing meaningfully for our purposes.

2. **Behavioural granularity.** Transport mode recognition literature typically uses 1 to 10 second windows. The book's 30 second choice was driven by long, prolonged labels in their dataset (sitting, driving, on table) where a coarser window made sense. Our sessions are also long but the within mode behaviour fluctuates faster, especially for bus and tram where braking events and road surface bursts last a few seconds.

3. **Sample count for stable statistics.** 250 samples is enough for mean, standard deviation, and percentile estimates to be stable. It is also enough for the FFT to produce a usable power spectrum.

The 50% overlap follows the book's explicit recommendation (Section 4.4.3, citing the established HAR practice). It doubles the number of training instances at the cost of strong correlation between adjacent windows. The correlation cost is acknowledged in the validation protocol: random window splits will be optimistic, and the headline results come from session and LOSO splits where adjacent overlapping windows stay inside the same fold.

We do not use the book's 90% overlap. They used it only because their dataset was small. Our 36 sessions at around 60 to 90 seconds of usable data each give us roughly 6,000 to 8,000 windows at 50% overlap, which is enough.

---

## Channels and derived signals

Raw channels (9 total):

```
acc_x, acc_y, acc_z              # accelerometer (includes gravity)
lin_ax, lin_ay, lin_az           # linear accelerometer (gravity removed)
gx, gy, gz                       # gyroscope
```

Derived orientation invariant magnitudes (3 total):

```
acc_mag    = sqrt(acc_x^2 + acc_y^2 + acc_z^2)
linacc_mag = sqrt(lin_ax^2 + lin_ay^2 + lin_az^2)
gyro_mag   = sqrt(gx^2 + gy^2 + gz^2)
```

Derived dynamic signal (1):

```
jerk_linacc = first finite difference of linacc_mag
```

The magnitudes are orientation invariant, which matters for LOSO because phone orientation drifts between users and sessions. The per axis channels are orientation dependent but carry structural information that magnitudes lose. The jerk captures transient events (braking, impacts, footstrikes) that smooth magnitude statistics miss.

We compute features at two levels:

- per axis features on the six linear acceleration and gyroscope axes
- magnitude features on the three magnitudes plus jerk

The raw accelerometer (with gravity) is used only for two features that capture orientation context: mean and standard deviation of `acc_mag`. We do not double up all features on both accelerometer types.

---

## Feature Set A: time domain

This is the book's Section 4.1.1 feature family. For each signal $s$ within a window:

```
mean, std, median, IQR, RMS, min, max, range
```

Eight statistics per signal. Three of these (mean, std, min, max) come directly from the book's equations 4.1 to 4.4. Median and IQR are robust counterparts to mean and std and matter here because public transport windows have heavy tailed distributions: most of the window is quiet, with occasional bursts from bumps or braking. RMS is the orientation independent energy measure standard in HAR. Range and the implicit (max minus min) descriptor sit alongside std as variability indicators.

Applied to:

- 6 per axis signals (lin_ax, lin_ay, lin_az, gx, gy, gz): 48 features
- 3 magnitude signals (acc_mag, linacc_mag, gyro_mag): 24 features

We also add the book's slope feature (Section 4.1.3, used there as a categorical discretiser) on the three magnitudes: 3 features. Slope captures trend within the window, which is useful for catching the start of acceleration events in bus or tram.

**Feature Set A total: 75 features.**

---

## Feature Set B: A + frequency domain

Frequency features come from the Discrete Fourier Transform applied to the window. We follow the book's Section 4.2.2 but choose summarised features over raw amplitudes.

Why summarised features instead of all amplitudes: the book's case study uses both. Their 40 sample window at 10 Hz sampling gives them around 20 frequency bins per signal. Our 250 sample windows give 125 unique amplitude bins per signal. With 4 signals (linacc_mag, gyro_mag, and two more we pick), that is 500 raw amplitude features. On 36 sessions, this is heavy and noisy. We follow the book's three aggregate features (max frequency, frequency weighted signal average, power spectral entropy) and add band powers and centroid.

For each of `linacc_mag` and `gyro_mag`, we compute:

- **Dominant frequency**: the frequency with the highest amplitude (book equation 4.13)
- **Dominant frequency amplitude**: the value of that peak
- **Frequency weighted signal average**: the average frequency weighted by amplitude (book equation 4.14). This is the spectral centroid in another name; we use the book's term for consistency.
- **Power spectral entropy**: how spread out the energy is across frequencies (book equations 4.15 to 4.17). High entropy means broadband noise. Low entropy means a few sharp peaks.
- **Band powers** in four bands: 0.2 to 1 Hz (slow oscillation, vehicle sway, pedalling), 1 to 3 Hz (walking cadence), 3 to 8 Hz (fast movement, road vibration), 8 to 20 Hz (high frequency vibration).
- **Band power ratio**: ratio of the 1 to 3 Hz band to total power. This is the single most discriminative feature we expect for walking.

That gives 10 features per signal, 20 features total across the two magnitude signals.

**Feature Set B total: 75 + 20 = 95 features.**

We do not include raw FFT amplitudes. If the modelling stage shows that frequency features carry weight, we can revisit this in an extension.

---

## Feature Set C: B + cross-axis and shape

This set targets the public transport cluster, where Sets A and B leave most of the work undone.

**Cross axis correlation features.** Pearson correlation between each pair of axes:

- linacc xy, xz, yz: 3 features
- gyro xy, xz, yz: 3 features

Walking has strong coupling between axes because the gait cycle drives multiple axes in phase. Cycling has a different coupling pattern. Vehicular modes have low and incoherent coupling. These six features are orientation dependent in absolute terms but their structure (which pairs are coupled) is informative.

**Distribution shape features.** On the three magnitudes (acc_mag, linacc_mag, gyro_mag):

- skewness, kurtosis: 6 features

The book mentions skewness as one of the additional features used in literature (Section 4.2.2, citing [7] and [118]). Heavy tailed bus/metro/tram distributions show high kurtosis and skewness because they are mostly quiet with occasional bursts.

**Jerk features.** On `jerk_linacc`:

- mean, std, max of absolute value: 3 features

Maximum absolute jerk captures the strongest transient in the window. This separates bus (frequent strong jerks from road surface) from metro (rare jerks, mostly smooth rail).

**Signal magnitude area.** One feature, computed as the sum of absolute values across the three linear acceleration axes divided by window length. This is a classical HAR feature and provides an orientation independent energy measure complementary to RMS.

**Peak count and zero crossing rate.** On `linacc_mag`:

- peak count (number of local maxima above 0.5 * std of window)
- zero crossing rate of the demeaned magnitude

2 features. Peak count is high and regular for walking. Zero crossing rate is high for oscillatory modes (walking, cycling) and low for vehicular modes.

**Axis std ratios.** Ratios of standard deviations between axis pairs:

- std(lin_ay) / std(lin_ax)
- std(lin_az) / std(lin_ax)
- std(gy) / std(gx)
- std(gz) / std(gx)

4 features. These encode the directional profile of motion without depending on absolute scale, which helps when participants have different overall movement intensities.

**Feature Set C total: 95 + 6 + 6 + 3 + 1 + 2 + 4 = 117 features.**

---

## What we leave out and why

**Categorical temporal patterns (book Section 4.1.2).** Our problem has no categorical attributes. The closest analogue would be to discretise magnitudes into low, medium, high and mine succession patterns, but the time scales of interest within a 5 second window are too short for the support based pattern mining to find anything useful. Adding this would inflate the feature count without addressing the bus vs metro vs tram problem.

**Raw FFT amplitudes (book Section 4.2.2).** Already explained: too many features for our sample size. The summarised frequency features capture what matters.

**Magnetometer features.** Excluded by design (project decision), to prevent the model from learning route signatures via heading drift.

**GPS features.** Excluded by design, same reason.

**Barometer features.** Excluded from the main feature table because one participant has no barometer. A separate ablation table will include pressure mean, pressure range, pressure std, pressure slope on the Babis plus Xiomara subset only.

---

## Leakage discipline at the feature stage

Feature engineering itself is leakage safe as long as every feature uses only the samples inside its window. There is no global statistic computed across the dataset, no normalisation, no scaling, no imputation at this stage. The CSV that comes out of the feature engineering notebook is raw feature values.

Standardisation, imputation of any feature that ends up with NaN, and feature selection all happen inside the cross validation folds in the modelling notebook. This is the only way to keep LOSO results honest.

One concrete leakage trap to flag: if we compute, for example, "z score of the window mean relative to the participant's overall mean," we have leaked participant identity into the feature. We do not do this. All features are window local.

---

## Output artefact

A single CSV file:

```
data/processed/features_common_no_baro.csv
```

Schema:

```
participant         # one of {Athina, Babis, Xiomara}
session_id          # unique session identifier
mode                # one of {stationary, walking, cycling, metro, bus, tram}
mode_id             # integer 0 to 5
window_index        # 0-based index within session
window_start_s      # seconds from session start
window_end_s        # seconds from session start
<117 feature columns>
```

The 117 feature columns are named with the pattern `<signal>_<feature>` for time domain, `<signal>_<freq_feature>` for frequency, and explicit names for cross-axis and shape features (e.g. `corr_linacc_xy`, `skew_linacc_mag`, `jerk_max_abs`, `sma_linacc`, `peak_count_linacc`, `zcr_linacc`, `std_ratio_linacc_yx`).

A second smaller CSV for the barometer ablation:

```
data/processed/features_baro_subset.csv
```

Containing the same 117 features plus pressure features, restricted to Babis and Xiomara sessions.

---

## Implementation order

When we move to code, the order is:

1. Build the windowing function. Confirm window count matches expectation (sessions × duration / 2.5 seconds with edge effects).
2. Implement Feature Set A. Sanity check: stationary windows have near zero values across motion features; walking windows have the highest std and RMS.
3. Add Feature Set B. Sanity check: walking dominant frequency lands in 1.5 to 2.5 Hz; stationary dominant frequency is noise.
4. Add Feature Set C. Sanity check: walking shows high peak count and high axis correlations; metro shows low values across all of these.
5. Run a single Random Forest at the end of the notebook with random split, just to confirm the feature table is well formed and the pipeline runs end to end. This is a smoke test, not a result.
6. Save the CSV and move to the modelling notebook.

The modelling notebook consumes the CSV and runs the ablation A vs B vs C inside the proper validation protocol.

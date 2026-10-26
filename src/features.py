"""Feature engineering for the transport-mode-detection project.

Public API (most useful entry points):

    Constants:
        SAMPLING_RATE, WINDOW_SECONDS, OVERLAP_FRAC,
        WINDOW_SAMPLES, STEP_SAMPLES, FREQ_BANDS,
        MODES, MODE_TO_ID, AXIS_COLS, ACC_COLS

    Building blocks (operate on one numpy array):
        basic_stats(arr) -> dict
        shape_and_energy(arr) -> dict
        slope(arr) -> float
        frequency_features(arr, fs=SAMPLING_RATE) -> dict
        peak_count(arr) -> int
        zero_crossing_rate(arr) -> float
        acceleration_event_features(arr, fs, low_threshold, high_threshold) -> dict

    Window-level helpers (operate on a DataFrame of one window):
        derive_magnitudes(window) -> dict
        pairwise_correlations(window, axis_cols, prefix) -> dict
        axis_std_ratios(window, axis_cols, prefix) -> dict
        signal_magnitude_area(window, axis_cols) -> float

    Dispatchers (produce the full feature dict for one window):
        compute_window_features(window) -> dict
            Standard set. 129 features when acc_mag absent, 141 when present.
        compute_event_features(window) -> dict
            12 burstiness/quietness features for the stationary vs vehicle binary.

    Session helpers:
        iterate_windows(n_samples, win, step)
        featurise_session(session_path, session_meta, include_event_features=False)
            Returns (feature_rows, dropped_rows).

Design notes:
    - Functions take numpy arrays or pandas DataFrames; they do not touch I/O.
    - All features are computed strictly within one window so the pipeline is
      leakage-safe by construction. Standardization happens later, per fold.
    - The `event_features` group is opt-in via `include_event_features=True`.
      Backwards compatible with the original notebook 04 by default.
"""

from __future__ import annotations

from typing import Iterable, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy.fft import rfft, rfftfreq
from scipy.stats import skew, kurtosis


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLING_RATE = 50          # Hz, common grid after resampling
WINDOW_SECONDS = 5.0
OVERLAP_FRAC = 0.5
WINDOW_SAMPLES = int(WINDOW_SECONDS * SAMPLING_RATE)        # 250
STEP_SAMPLES = int(WINDOW_SAMPLES * (1 - OVERLAP_FRAC))     # 125

# Frequency bands (Hz)
FREQ_BANDS = {
    'band_0p2_1':  (0.2, 1.0),    # slow oscillation, vehicle sway, pedalling
    'band_1_3':    (1.0, 3.0),    # walking cadence
    'band_3_8':    (3.0, 8.0),    # fast movement, road vibration
    'band_8_20':   (8.0, 20.0),   # high frequency vibration
}

MODES = ['stationary', 'walking', 'cycling', 'metro', 'bus', 'tram']
MODE_TO_ID = {m: i for i, m in enumerate(MODES)}

AXIS_COLS = ['lin_ax', 'lin_ay', 'lin_az', 'gx', 'gy', 'gz']
ACC_COLS = ['acc_x', 'acc_y', 'acc_z']

# Default thresholds for event features (tuned to the sensor noise floors
# observed in the dataset). They can be overridden per call if needed.
EVENT_THRESHOLDS = {
    'linacc_mag': (0.05, 0.5),   # m/s^2: quiet, active
    'gyro_mag':   (0.01, 0.1),   # rad/s: quiet, active
}


# ---------------------------------------------------------------------------
# Derived signals
# ---------------------------------------------------------------------------

def derive_magnitudes(window: pd.DataFrame) -> dict:
    """Compute orientation-invariant magnitudes and the jerk signal for one window."""
    out = {}
    if all(c in window.columns for c in ACC_COLS):
        out['acc_mag'] = np.sqrt(
            window['acc_x'].values ** 2
            + window['acc_y'].values ** 2
            + window['acc_z'].values ** 2
        )
    out['linacc_mag'] = np.sqrt(
        window['lin_ax'].values ** 2
        + window['lin_ay'].values ** 2
        + window['lin_az'].values ** 2
    )
    out['gyro_mag'] = np.sqrt(
        window['gx'].values ** 2
        + window['gy'].values ** 2
        + window['gz'].values ** 2
    )
    out['jerk_linacc'] = (
        np.diff(out['linacc_mag'], prepend=out['linacc_mag'][0]) * SAMPLING_RATE
    )
    return out


# ---------------------------------------------------------------------------
# Building blocks (operate on one 1D array)
# ---------------------------------------------------------------------------

def basic_stats(arr: np.ndarray) -> dict:
    """Eight time-domain statistics."""
    arr = np.asarray(arr, dtype=float)
    q25, q50, q75 = np.percentile(arr, [25, 50, 75])
    return {
        'mean':   float(np.mean(arr)),
        'std':    float(np.std(arr, ddof=0)),
        'median': float(q50),
        'iqr':    float(q75 - q25),
        'rms':    float(np.sqrt(np.mean(arr ** 2))),
        'min':    float(np.min(arr)),
        'max':    float(np.max(arr)),
        'range':  float(np.max(arr) - np.min(arr)),
    }


def shape_and_energy(arr: np.ndarray) -> dict:
    """Distribution shape and energy: skewness, kurtosis, mean squared value."""
    arr = np.asarray(arr, dtype=float)
    return {
        'skew':     float(skew(arr, bias=False)) if np.std(arr) > 1e-9 else 0.0,
        'kurtosis': float(kurtosis(arr, fisher=True, bias=False)) if np.std(arr) > 1e-9 else 0.0,
        'energy':   float(np.mean(arr ** 2)),
    }


def slope(arr: np.ndarray) -> float:
    """Linear regression slope of the signal against sample index."""
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    y_mean = arr.mean()
    denom = np.sum((x - x_mean) ** 2)
    if denom < 1e-12:
        return 0.0
    return float(np.sum((x - x_mean) * (arr - y_mean)) / denom)


def frequency_features(arr: np.ndarray, fs: int = SAMPLING_RATE) -> dict:
    """FFT-based features. Mean is removed first so DC does not dominate."""
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    arr = arr - np.mean(arr)
    freqs = rfftfreq(n, d=1.0 / fs)
    spectrum = np.abs(rfft(arr))
    power = (spectrum ** 2) / n

    out = {}
    if len(spectrum) > 1 and spectrum[1:].sum() > 1e-12:
        k = int(np.argmax(spectrum[1:])) + 1
        out['dominant_freq'] = float(freqs[k])
        out['dominant_amp']  = float(spectrum[k])
    else:
        out['dominant_freq'] = 0.0
        out['dominant_amp']  = 0.0

    amp_sum = spectrum.sum()
    if amp_sum > 1e-12:
        out['freq_weighted_avg'] = float(np.sum(freqs * spectrum) / amp_sum)
    else:
        out['freq_weighted_avg'] = 0.0

    p_sum = power.sum()
    if p_sum > 1e-12:
        p_norm = power / p_sum
        p_norm = p_norm[p_norm > 0]
        out['spec_entropy'] = float(-np.sum(p_norm * np.log(p_norm)))
    else:
        out['spec_entropy'] = 0.0

    total_power = power.sum() + 1e-12
    band_powers = {}
    for name, (f_lo, f_hi) in FREQ_BANDS.items():
        mask = (freqs >= f_lo) & (freqs < f_hi)
        bp = float(power[mask].sum())
        out[f'power_{name}'] = bp
        band_powers[name] = bp

    out['band_ratio_1_3_total'] = band_powers['band_1_3'] / total_power
    out['band_ratio_3_8_to_1_3'] = band_powers['band_3_8'] / (band_powers['band_1_3'] + 1e-12)
    return out


def peak_count(arr: np.ndarray) -> int:
    """Number of local maxima above the midpoint between mean and max.

    Scale-free: works for both small (stationary) and large (walking) signals
    without an absolute threshold.
    """
    arr = np.asarray(arr, dtype=float)
    if len(arr) < 3:
        return 0
    arr_max = float(np.max(arr))
    arr_mean = float(np.mean(arr))
    excursion = arr_max - arr_mean
    if excursion < 1e-9:
        return 0
    peaks, _ = scipy_signal.find_peaks(
        arr, height=arr_mean + 0.5 * excursion, distance=5
    )
    return int(len(peaks))


def zero_crossing_rate(arr: np.ndarray) -> float:
    """Zero crossings of the demeaned signal, normalized by length."""
    arr = np.asarray(arr, dtype=float) - float(np.mean(arr))
    return float(np.sum(np.diff(np.sign(arr)) != 0)) / len(arr)


def acceleration_event_features(
    arr: np.ndarray,
    fs: int = SAMPLING_RATE,
    low_threshold: float = 0.05,
    high_threshold: float = 0.5,
) -> dict:
    """Features for the stationary vs vehicle distinction.

    The physical intuition: stationary windows have near-zero magnitude
    *everywhere* in the window, while vehicle modes have intermittent
    acceleration events between quieter passages. These six features capture
    that quiet/active structure directly.

    Parameters
    ----------
    arr : array-like
        Magnitude signal in the window (e.g. linacc_mag or gyro_mag).
    fs : int
        Sampling rate in Hz, used to convert run lengths to seconds.
    low_threshold : float
        Signal below this is treated as "quiet" (sensor-noise floor).
    high_threshold : float
        Signal above this is treated as an active motion event.
    """
    arr = np.abs(np.asarray(arr, dtype=float))
    n = len(arr)
    out = {}

    # Fraction of samples below the noise floor: ~1.0 for stationary,
    # much less for vehicles whose ambient vibration sits above the floor.
    out['quiet_frac'] = float(np.sum(arr < low_threshold) / n)

    # Fraction of samples above the active threshold: ~0.0 for stationary,
    # non-zero for vehicles with road bumps, brakes, rail joints, etc.
    out['active_frac'] = float(np.sum(arr > high_threshold) / n)

    # Longest consecutive stretch below the noise floor, in seconds.
    # Stationary should approach the window length; vehicles much shorter.
    is_quiet = arr < low_threshold
    if is_quiet.any():
        padded = np.concatenate(([False], is_quiet, [False])).astype(np.int8)
        diffs = np.diff(padded)
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        run_lengths = ends - starts
        longest = int(run_lengths.max()) if len(run_lengths) > 0 else 0
    else:
        longest = 0
    out['longest_quiet_run_s'] = float(longest / fs)

    # Number of distinct burst events: count rising edges into the active band.
    is_active = arr > high_threshold
    if is_active.any():
        out['burst_count'] = int(np.sum(np.diff(is_active.astype(np.int8)) == 1))
        if not is_active[0]:
            pass  # already counts the first burst correctly
        else:
            out['burst_count'] += 1   # array starts in active state
    else:
        out['burst_count'] = 0

    # 95th percentile of |signal|: catches the peak excursion level.
    # Tiny for stationary, larger for any vehicle with at least one event.
    out['p95'] = float(np.percentile(arr, 95))

    # Spread between 95th and 5th percentile: captures dynamic range.
    out['p95_minus_p5'] = float(np.percentile(arr, 95) - np.percentile(arr, 5))

    return out


# ---------------------------------------------------------------------------
# Window-level helpers (operate on a DataFrame slice)
# ---------------------------------------------------------------------------

def pairwise_correlations(window: pd.DataFrame, axis_cols: list, prefix: str) -> dict:
    """Pearson correlations between the three axes of a 3D signal."""
    out = {}
    pairs = [('x', 'y'), ('x', 'z'), ('y', 'z')]
    arrs = {c[-1]: window[c].values.astype(float) for c in axis_cols}
    for a, b in pairs:
        sa = arrs[a]
        sb = arrs[b]
        if np.std(sa) < 1e-9 or np.std(sb) < 1e-9:
            out[f'{prefix}_corr_{a}{b}'] = 0.0
        else:
            out[f'{prefix}_corr_{a}{b}'] = float(np.corrcoef(sa, sb)[0, 1])
    return out


def axis_std_ratios(window: pd.DataFrame, axis_cols: list, prefix: str) -> dict:
    """Standard deviation ratios between axes (orientation-independent structure)."""
    sx = float(np.std(window[axis_cols[0]].values))
    sy = float(np.std(window[axis_cols[1]].values))
    sz = float(np.std(window[axis_cols[2]].values))
    eps = 1e-9
    return {
        f'{prefix}_std_ratio_yx': sy / (sx + eps),
        f'{prefix}_std_ratio_zx': sz / (sx + eps),
        f'{prefix}_std_ratio_zy': sz / (sy + eps),
    }


def signal_magnitude_area(window: pd.DataFrame, axis_cols: list) -> float:
    """Sum of absolute values across three axes divided by window length."""
    total = 0.0
    for c in axis_cols:
        total += float(np.sum(np.abs(window[c].values)))
    return total / len(window)


# ---------------------------------------------------------------------------
# Dispatchers
# ---------------------------------------------------------------------------

def compute_window_features(window: pd.DataFrame) -> dict:
    """Standard feature set for one window.

    Returns 129 features when raw accelerometer columns are absent, 141 when
    present. Matches the exact output of the original notebook 04 code.
    """
    feats = {}
    mags = derive_magnitudes(window)

    # Block 1: per-axis basic stats
    for ax in AXIS_COLS:
        arr = window[ax].values.astype(float)
        for k, v in basic_stats(arr).items():
            feats[f'{ax}_{k}'] = v

    # Block 2: magnitude basic stats
    for mag_name, mag_arr in mags.items():
        for k, v in basic_stats(mag_arr).items():
            feats[f'{mag_name}_{k}'] = v

    # Block 3: shape and energy on magnitudes
    for mag_name, mag_arr in mags.items():
        for k, v in shape_and_energy(mag_arr).items():
            feats[f'{mag_name}_{k}'] = v

    # Block 4: slope on axes and magnitudes (jerk excluded by design)
    for ax in AXIS_COLS:
        feats[f'{ax}_slope'] = slope(window[ax].values.astype(float))
    for mag_name in ['acc_mag', 'linacc_mag', 'gyro_mag']:
        if mag_name in mags:
            feats[f'{mag_name}_slope'] = slope(mags[mag_name])

    # Block 5: frequency-domain features on linacc_mag and gyro_mag
    for mag_name in ['linacc_mag', 'gyro_mag']:
        for k, v in frequency_features(mags[mag_name]).items():
            feats[f'{mag_name}_{k}'] = v

    # Block 6: cross-axis correlations
    feats.update(pairwise_correlations(window, ['lin_ax', 'lin_ay', 'lin_az'], 'linacc'))
    feats.update(pairwise_correlations(window, ['gx', 'gy', 'gz'], 'gyro'))

    # Block 7: axis std ratios
    feats.update(axis_std_ratios(window, ['lin_ax', 'lin_ay', 'lin_az'], 'linacc'))
    feats.update(axis_std_ratios(window, ['gx', 'gy', 'gz'], 'gyro'))

    # Block 8: SMA, peak count, ZCR
    feats['linacc_sma'] = signal_magnitude_area(window, ['lin_ax', 'lin_ay', 'lin_az'])
    feats['gyro_sma']   = signal_magnitude_area(window, ['gx', 'gy', 'gz'])
    feats['linacc_mag_peak_count'] = peak_count(mags['linacc_mag'])
    feats['gyro_mag_peak_count']   = peak_count(mags['gyro_mag'])
    feats['lin_ay_peak_count']     = peak_count(window['lin_ay'].values.astype(float))
    feats['linacc_mag_zcr'] = zero_crossing_rate(mags['linacc_mag'])
    feats['lin_ay_zcr']     = zero_crossing_rate(window['lin_ay'].values.astype(float))

    # Block 9: cross-magnitude correlation
    if np.std(mags['linacc_mag']) > 1e-9 and np.std(mags['gyro_mag']) > 1e-9:
        feats['linacc_gyro_mag_corr'] = float(
            np.corrcoef(mags['linacc_mag'], mags['gyro_mag'])[0, 1]
        )
    else:
        feats['linacc_gyro_mag_corr'] = 0.0

    return feats


def compute_event_features(window: pd.DataFrame) -> dict:
    """Event/burstiness features for the stationary vs vehicle distinction.

    Produces 12 features: six for linacc_mag and six for gyro_mag.
    All have the prefix `event_` so they are easy to identify in tables.
    """
    mags = derive_magnitudes(window)
    feats = {}
    for mag_name in ('linacc_mag', 'gyro_mag'):
        lo, hi = EVENT_THRESHOLDS[mag_name]
        sub = acceleration_event_features(
            mags[mag_name], fs=SAMPLING_RATE,
            low_threshold=lo, high_threshold=hi,
        )
        for k, v in sub.items():
            feats[f'event_{mag_name}_{k}'] = v
    return feats


# ---------------------------------------------------------------------------
# Multi-scale temporal features (15-second context window)
# ---------------------------------------------------------------------------

def compute_multiscale_features(
    session_df: pd.DataFrame,
    window_start: int,
    window_end: int,
    context_seconds: float = 15.0,
    burst_threshold_linacc: float = 0.5,
) -> dict:
    """Statistics computed on a longer context window centered on a 5-second window.

    Designed for the bus, metro, tram three-way classification at Level 3B of
    the hierarchical model. Vehicle dynamics play out over time scales longer
    than the 5-second windows used elsewhere in the pipeline: bus stop-go
    cycles are 10 to 20 seconds, tram rail joints repeat every few seconds,
    metro behaves smoothly between stations. A 15-second context window
    captures these patterns where 5 seconds truncates them.

    The context window is symmetrically centered on the 5-second window.
    Near session boundaries the context is truncated naturally; the features
    are still valid, just computed on fewer samples. Bad samples in the
    context are included as-is because the 5-second window itself has
    already passed the bad_sample guard.

    All features carry the prefix `ms_` so they are easy to identify in tables.

    Parameters
    ----------
    session_df : pd.DataFrame
        Full session DataFrame. Must contain lin_ax, lin_ay, lin_az and the
        three gyroscope axes.
    window_start, window_end : int
        Sample indices of the 5-second window this context surrounds.
    context_seconds : float
        Length of the context window in seconds. Default 15 seconds.
    burst_threshold_linacc : float
        Threshold on |linacc_mag| above which a sample counts as part of an
        active event. Default 0.5 m/s^2, matching the event_features defaults.

    Returns
    -------
    dict
        19 features with the `ms_` prefix. See the inline grouping below.
    """
    context_samples = int(context_seconds * SAMPLING_RATE)
    center = (window_start + window_end) // 2
    half = context_samples // 2

    ctx_start = max(0, center - half)
    ctx_end = min(len(session_df), center + half)
    ctx = session_df.iloc[ctx_start:ctx_end]

    mags = derive_magnitudes(ctx)
    out = {}

    # Block 1: amplitude statistics at the 15-second scale (5 stats x 2 mags = 10)
    for mag_name in ('linacc_mag', 'gyro_mag'):
        arr = mags[mag_name]
        q25, _, q75, q95 = np.percentile(arr, [25, 50, 75, 95])
        out[f'ms_{mag_name}_mean']  = float(np.mean(arr))
        out[f'ms_{mag_name}_std']   = float(np.std(arr))
        out[f'ms_{mag_name}_iqr']   = float(q75 - q25)
        out[f'ms_{mag_name}_p95']   = float(q95)
        out[f'ms_{mag_name}_range'] = float(np.max(arr) - np.min(arr))

    # Block 2: frequency features at the 15-second scale (2 features x 2 mags = 4)
    # A 15-second window gives 0.067 Hz resolution, three times finer than the
    # 0.2 Hz resolution of the 5-second windows. This helps distinguish
    # vehicle modes with low-frequency stop-go patterns.
    for mag_name in ('linacc_mag', 'gyro_mag'):
        ff = frequency_features(mags[mag_name])
        out[f'ms_{mag_name}_dominant_freq'] = ff['dominant_freq']
        out[f'ms_{mag_name}_spec_entropy']  = ff['spec_entropy']

    # Block 3: jerk variability over the 15-second context (2)
    jerk = mags['jerk_linacc']
    out['ms_jerk_linacc_max'] = float(np.max(np.abs(jerk)))
    out['ms_jerk_linacc_std'] = float(np.std(jerk))

    # Block 4: burst structure over the 15-second context (3)
    # Trams have regular rail-joint impacts (low interval std);
    # buses have irregular road bumps (high interval std);
    # metro has fewer, mostly station-related bursts.
    linacc_abs = np.abs(mags['linacc_mag'])
    is_active = linacc_abs > burst_threshold_linacc
    if is_active.any():
        edges = np.diff(is_active.astype(np.int8))
        n_bursts = int(np.sum(edges == 1))
        if is_active[0]:
            n_bursts += 1
        burst_starts = np.where(edges == 1)[0]
        if len(burst_starts) >= 2:
            intervals = np.diff(burst_starts)
            interval_std_s = float(np.std(intervals) / SAMPLING_RATE)
        else:
            interval_std_s = 0.0
        active_frac = float(is_active.mean())
    else:
        n_bursts = 0
        interval_std_s = 0.0
        active_frac = 0.0
    out['ms_burst_count']           = n_bursts
    out['ms_burst_interval_std_s']  = interval_std_s
    out['ms_active_frac']           = active_frac

    return out


# ---------------------------------------------------------------------------
# Per-session amplitude normalization
# ---------------------------------------------------------------------------

def per_session_normalize(session_df: pd.DataFrame) -> pd.DataFrame:
    """Divide linear accelerometer and gyroscope by their session-level RMS.

    The RMS is computed on the magnitude signal across the entire session,
    then applied to each axis individually. This removes the session-level
    amplitude scale (which is dominated by phone placement) while preserving
    the relative scale between axes (which encodes orientation information).

    After normalization, walking from a participant who carries the phone
    loosely (high amplitude) and walking from a participant with a tight
    pocket placement (low amplitude) produce similar normalized signals
    because the cadence pattern is amplitude-invariant.

    Trade-off: absolute amplitude information is lost, so this CSV is the
    wrong input for classifiers that need to distinguish quiet from active
    motion (stationary versus vehicle, for example). Use the standard
    features for those.

    Other columns (timestamps, flags, pressure, raw accelerometer) are
    unchanged. Raw accelerometer is left alone because it includes gravity,
    which is a constant bias rather than a session-level amplitude effect.
    """
    df = session_df.copy()

    if all(c in df.columns for c in ['lin_ax', 'lin_ay', 'lin_az']):
        linacc_mag = np.sqrt(
            df['lin_ax'].values ** 2
            + df['lin_ay'].values ** 2
            + df['lin_az'].values ** 2
        )
        linacc_rms = float(np.sqrt(np.mean(linacc_mag ** 2)))
        if linacc_rms > 1e-9:
            df['lin_ax'] = df['lin_ax'].values / linacc_rms
            df['lin_ay'] = df['lin_ay'].values / linacc_rms
            df['lin_az'] = df['lin_az'].values / linacc_rms

    if all(c in df.columns for c in ['gx', 'gy', 'gz']):
        gyro_mag = np.sqrt(
            df['gx'].values ** 2
            + df['gy'].values ** 2
            + df['gz'].values ** 2
        )
        gyro_rms = float(np.sqrt(np.mean(gyro_mag ** 2)))
        if gyro_rms > 1e-9:
            df['gx'] = df['gx'].values / gyro_rms
            df['gy'] = df['gy'].values / gyro_rms
            df['gz'] = df['gz'].values / gyro_rms

    return df


# ---------------------------------------------------------------------------
# Windowing and session-level featurisation
# ---------------------------------------------------------------------------

def iterate_windows(n_samples: int, win: int = WINDOW_SAMPLES, step: int = STEP_SAMPLES):
    """Yield (start, end) index pairs for a sliding window over n_samples."""
    start = 0
    while start + win <= n_samples:
        yield start, start + win
        start += step


def featurise_session(
    session_path: Path,
    session_meta: dict,
    include_event_features: bool = False,
    apply_session_normalization: bool = False,
) -> Tuple[list, list]:
    """Compute features for every valid window in one session.

    Parameters
    ----------
    session_path : Path
        Path to the interim per-session CSV (output of notebook 02).
    session_meta : dict
        Metadata to attach to every produced row (session_id, participant,
        mode, mode_id, day).
    include_event_features : bool
        If True, also compute the 12 burstiness features for Level 2B work.
    apply_session_normalization : bool
        If True, divide each axis by its session-level magnitude RMS before
        computing features. Useful for cross-user classifiers that should be
        invariant to phone placement amplitude differences. Do not combine
        with `include_event_features=True` (event features need absolute
        amplitudes to work; the combination is not meaningful).

    Returns
    -------
    (feature_rows, dropped_rows) : tuple of lists of dicts
    """
    df = pd.read_csv(session_path)
    df = df.reset_index(drop=True)

    if apply_session_normalization:
        df = per_session_normalize(df)

    has_bad = 'bad_sample' in df.columns
    has_flag = 'flagged_sample' in df.columns

    feature_rows = []
    dropped_rows = []

    for w_idx, (s, e) in enumerate(iterate_windows(len(df))):
        w = df.iloc[s:e]
        if has_bad and bool(w['bad_sample'].any()):
            dropped_rows.append({
                **session_meta,
                'window_index':   w_idx,
                'window_start_s': float(s / SAMPLING_RATE),
                'window_end_s':   float(e / SAMPLING_RATE),
                'n_bad_samples':  int(w['bad_sample'].sum()),
                'reason':         'bad_sample present',
            })
            continue

        try:
            feats = compute_window_features(w)
            if include_event_features:
                feats.update(compute_event_features(w))
        except Exception as exc:
            dropped_rows.append({
                **session_meta,
                'window_index':   w_idx,
                'window_start_s': float(s / SAMPLING_RATE),
                'window_end_s':   float(e / SAMPLING_RATE),
                'n_bad_samples':  0,
                'reason':         f'feature error: {exc}',
            })
            continue

        flagged_frac = float(w['flagged_sample'].mean()) if has_flag else 0.0

        row = {
            **session_meta,
            'window_index':    w_idx,
            'window_start_s':  float(s / SAMPLING_RATE),
            'window_end_s':    float(e / SAMPLING_RATE),
            'flagged_frac':    flagged_frac,
            **feats,
        }
        feature_rows.append(row)

    return feature_rows, dropped_rows

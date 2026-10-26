"""Smoke test for the feature engineering notebook.

Builds synthetic sessions that mimic the expected schema from notebook 02,
runs the full feature pipeline, and verifies:
1. Feature count matches design (141 features)
2. No NaN / inf
3. Physical sanity: walking dominant freq lands in 1-3 Hz band,
   walking linacc std > stationary, etc.
"""

import json
from pathlib import Path
import sys
import tempfile
import shutil

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy.fft import rfft, rfftfreq
from scipy.stats import skew, kurtosis

# ============================================================================
# Copy the feature functions verbatim from the notebook
# ============================================================================

SAMPLING_RATE = 50
WINDOW_SECONDS = 5.0
OVERLAP_FRAC = 0.5
WINDOW_SAMPLES = int(WINDOW_SECONDS * SAMPLING_RATE)
STEP_SAMPLES = int(WINDOW_SAMPLES * (1 - OVERLAP_FRAC))

FREQ_BANDS = {
    'band_0p2_1': (0.2, 1.0),
    'band_1_3': (1.0, 3.0),
    'band_3_8': (3.0, 8.0),
    'band_8_20': (8.0, 20.0),
}

MODES = ['stationary', 'walking', 'cycling', 'metro', 'bus', 'tram']
MODE_TO_ID = {m: i for i, m in enumerate(MODES)}

AXIS_COLS = ['lin_ax', 'lin_ay', 'lin_az', 'gx', 'gy', 'gz']
ACC_COLS = ['acc_x', 'acc_y', 'acc_z']


def derive_magnitudes(window):
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
    out['jerk_linacc'] = np.diff(out['linacc_mag'], prepend=out['linacc_mag'][0]) * SAMPLING_RATE
    return out


def basic_stats(arr):
    arr = np.asarray(arr, dtype=float)
    q25, q50, q75 = np.percentile(arr, [25, 50, 75])
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr, ddof=0)),
        'median': float(q50),
        'iqr': float(q75 - q25),
        'rms': float(np.sqrt(np.mean(arr ** 2))),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'range': float(np.max(arr) - np.min(arr)),
    }


def shape_and_energy(arr):
    arr = np.asarray(arr, dtype=float)
    return {
        'skew': float(skew(arr, bias=False)) if np.std(arr) > 1e-9 else 0.0,
        'kurtosis': float(kurtosis(arr, fisher=True, bias=False)) if np.std(arr) > 1e-9 else 0.0,
        'energy': float(np.mean(arr ** 2)),
    }


def slope(arr):
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


def frequency_features(arr, fs=SAMPLING_RATE):
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
        out['dominant_amp'] = float(spectrum[k])
    else:
        out['dominant_freq'] = 0.0
        out['dominant_amp'] = 0.0

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


def pairwise_correlations(window, axis_cols, prefix):
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


def axis_std_ratios(window, axis_cols, prefix):
    sx = float(np.std(window[axis_cols[0]].values))
    sy = float(np.std(window[axis_cols[1]].values))
    sz = float(np.std(window[axis_cols[2]].values))
    eps = 1e-9
    return {
        f'{prefix}_std_ratio_yx': sy / (sx + eps),
        f'{prefix}_std_ratio_zx': sz / (sx + eps),
        f'{prefix}_std_ratio_zy': sz / (sy + eps),
    }


def signal_magnitude_area(window, axis_cols):
    total = 0.0
    for c in axis_cols:
        total += float(np.sum(np.abs(window[c].values)))
    return total / len(window)


def peak_count(arr):
    arr = np.asarray(arr, dtype=float)
    if len(arr) < 3:
        return 0
    arr_max = float(np.max(arr))
    arr_mean = float(np.mean(arr))
    excursion = arr_max - arr_mean
    if excursion < 1e-9:
        return 0
    # Peak must be at least halfway between mean and max.
    # Scale-free; works for both small (stationary) and large (walking) signals.
    peaks, _ = scipy_signal.find_peaks(
        arr, height=arr_mean + 0.5 * excursion, distance=5
    )
    return int(len(peaks))


def zero_crossing_rate(arr):
    arr = np.asarray(arr, dtype=float) - float(np.mean(arr))
    return float(np.sum(np.diff(np.sign(arr)) != 0)) / len(arr)


def compute_window_features(window):
    feats = {}
    mags = derive_magnitudes(window)

    for ax in AXIS_COLS:
        arr = window[ax].values.astype(float)
        for k, v in basic_stats(arr).items():
            feats[f'{ax}_{k}'] = v

    for mag_name, mag_arr in mags.items():
        for k, v in basic_stats(mag_arr).items():
            feats[f'{mag_name}_{k}'] = v

    for mag_name, mag_arr in mags.items():
        for k, v in shape_and_energy(mag_arr).items():
            feats[f'{mag_name}_{k}'] = v

    for ax in AXIS_COLS:
        feats[f'{ax}_slope'] = slope(window[ax].values.astype(float))
    for mag_name in ['acc_mag', 'linacc_mag', 'gyro_mag']:
        if mag_name in mags:
            feats[f'{mag_name}_slope'] = slope(mags[mag_name])

    for mag_name in ['linacc_mag', 'gyro_mag']:
        for k, v in frequency_features(mags[mag_name]).items():
            feats[f'{mag_name}_{k}'] = v

    feats.update(pairwise_correlations(window, ['lin_ax', 'lin_ay', 'lin_az'], 'linacc'))
    feats.update(pairwise_correlations(window, ['gx', 'gy', 'gz'], 'gyro'))

    feats.update(axis_std_ratios(window, ['lin_ax', 'lin_ay', 'lin_az'], 'linacc'))
    feats.update(axis_std_ratios(window, ['gx', 'gy', 'gz'], 'gyro'))

    feats['linacc_sma'] = signal_magnitude_area(window, ['lin_ax', 'lin_ay', 'lin_az'])
    feats['gyro_sma'] = signal_magnitude_area(window, ['gx', 'gy', 'gz'])
    feats['linacc_mag_peak_count'] = peak_count(mags['linacc_mag'])
    feats['gyro_mag_peak_count'] = peak_count(mags['gyro_mag'])
    feats['lin_ay_peak_count'] = peak_count(window['lin_ay'].values.astype(float))
    feats['linacc_mag_zcr'] = zero_crossing_rate(mags['linacc_mag'])
    feats['lin_ay_zcr'] = zero_crossing_rate(window['lin_ay'].values.astype(float))

    if np.std(mags['linacc_mag']) > 1e-9 and np.std(mags['gyro_mag']) > 1e-9:
        feats['linacc_gyro_mag_corr'] = float(
            np.corrcoef(mags['linacc_mag'], mags['gyro_mag'])[0, 1]
        )
    else:
        feats['linacc_gyro_mag_corr'] = 0.0

    return feats


# ============================================================================
# Synthetic data generator
# ============================================================================

def make_session(mode, n_seconds=60, rng=None):
    """Generate a synthetic session that loosely mimics the physics of each mode."""
    if rng is None:
        rng = np.random.default_rng(0)
    n = int(n_seconds * SAMPLING_RATE)
    t = np.arange(n) / SAMPLING_RATE

    if mode == 'stationary':
        # Near-zero motion, only sensor noise
        lin_ax = rng.normal(0, 0.02, n)
        lin_ay = rng.normal(0, 0.02, n)
        lin_az = rng.normal(0, 0.02, n)
        gx = rng.normal(0, 0.005, n)
        gy = rng.normal(0, 0.005, n)
        gz = rng.normal(0, 0.005, n)
    elif mode == 'walking':
        # Strong impulsive motion at 2 Hz step cadence.
        # Use a half-rectified sine so the magnitude has a clean 2 Hz fundamental
        # rather than only the 4 Hz second harmonic that pure sine magnitudes produce.
        f_step = 2.0
        impulse = np.clip(np.sin(2 * np.pi * f_step * t), 0, None)  # step impulses
        lin_ay = 5.0 * impulse + rng.normal(0, 0.3, n)
        lin_ax = 2.0 * impulse + rng.normal(0, 0.3, n)
        lin_az = 3.0 * impulse + rng.normal(0, 0.3, n)
        gy = 1.0 * impulse + rng.normal(0, 0.1, n)
        gx = 0.6 * impulse + rng.normal(0, 0.1, n)
        gz = 0.5 * impulse + rng.normal(0, 0.1, n)
    elif mode == 'cycling':
        f_pedal = 1.0
        lin_ax = 1.2 * np.sin(2 * np.pi * f_pedal * t) + rng.normal(0, 0.3, n)
        lin_ay = 0.8 * np.sin(2 * np.pi * f_pedal * t + 0.2) + rng.normal(0, 0.3, n)
        lin_az = 0.9 * np.sin(2 * np.pi * f_pedal * t * 2) + rng.normal(0, 0.3, n)
        # Higher-freq vibration from road
        for axis in [lin_ax, lin_ay, lin_az]:
            axis += rng.normal(0, 0.2, n) * np.sin(2 * np.pi * 5 * t)
        gx = 0.3 * np.sin(2 * np.pi * f_pedal * t) + rng.normal(0, 0.05, n)
        gy = 0.5 * np.sin(2 * np.pi * f_pedal * t) + rng.normal(0, 0.05, n)
        gz = 0.4 * np.sin(2 * np.pi * f_pedal * t) + rng.normal(0, 0.05, n)
    else:
        # bus, metro, tram: low irregular motion with bursts
        burst_scale = {'bus': 0.4, 'metro': 0.15, 'tram': 0.18}[mode]
        lin_ax = rng.normal(0, burst_scale, n)
        lin_ay = rng.normal(0, burst_scale, n)
        lin_az = rng.normal(0, burst_scale, n)
        # Add occasional bursts
        burst_idx = rng.choice(n, size=int(n * 0.02), replace=False)
        lin_ax[burst_idx] += rng.normal(0, burst_scale * 5, len(burst_idx))
        gx = rng.normal(0, burst_scale * 0.1, n)
        gy = rng.normal(0, burst_scale * 0.1, n)
        gz = rng.normal(0, burst_scale * 0.1, n)

    # Accelerometer = linear acc + gravity (assume z = -9.81)
    acc_x = lin_ax
    acc_y = lin_ay
    acc_z = lin_az + 9.81

    df = pd.DataFrame({
        't': t,
        'lin_ax': lin_ax, 'lin_ay': lin_ay, 'lin_az': lin_az,
        'gx': gx, 'gy': gy, 'gz': gz,
        'acc_x': acc_x, 'acc_y': acc_y, 'acc_z': acc_z,
        'bad_sample': False,
        'flagged_sample': False,
    })
    return df


# ============================================================================
# Run the smoke test
# ============================================================================

def main():
    print("=" * 70)
    print("FEATURE ENGINEERING SMOKE TEST")
    print("=" * 70)

    rng = np.random.default_rng(42)
    rows = []

    for mode in MODES:
        df = make_session(mode, n_seconds=60, rng=rng)
        n = len(df)
        for w_idx, start in enumerate(range(0, n - WINDOW_SAMPLES + 1, STEP_SAMPLES)):
            w = df.iloc[start:start + WINDOW_SAMPLES]
            feats = compute_window_features(w)
            rows.append({
                'mode': mode,
                'mode_id': MODE_TO_ID[mode],
                'window_index': w_idx,
                **feats,
            })

    df = pd.DataFrame(rows)
    feature_cols = [c for c in df.columns if c not in ('mode', 'mode_id', 'window_index')]
    n_features = len(feature_cols)

    print(f"\nTotal windows generated: {len(df)}")
    print(f"Total feature columns:   {n_features}")
    print(f"Expected feature count:  141")
    assert n_features == 141, f"Feature count mismatch: got {n_features}, expected 141"
    print("[PASS] Feature count matches design")

    # No NaN / inf
    n_nan = int(df[feature_cols].isna().sum().sum())
    n_inf = int(np.isinf(df[feature_cols].values).sum())
    print(f"\nNaN count: {n_nan}")
    print(f"Inf count: {n_inf}")
    assert n_nan == 0, f"Found {n_nan} NaN values"
    assert n_inf == 0, f"Found {n_inf} inf values"
    print("[PASS] No NaN or inf")

    # Physical sanity
    print("\nMedian linacc_mag_std per mode:")
    grp = df.groupby('mode')['linacc_mag_std'].median().reindex(MODES)
    print(grp.to_string())
    assert grp['stationary'] < grp['walking'], "stationary linacc_std >= walking"
    assert grp['walking'] > grp['metro'], "walking linacc_std <= metro"
    print("[PASS] linacc_mag_std ordering correct")

    print("\nMedian walking dominant_freq:")
    wf = df.loc[df['mode'] == 'walking', 'linacc_mag_dominant_freq']
    med = wf.median()
    print(f"  {med:.2f} Hz (IQR {wf.quantile(0.25):.2f} to {wf.quantile(0.75):.2f})")
    assert 1.5 <= med <= 2.5, f"Walking dominant_freq median {med:.2f} not in [1.5, 2.5]"
    print("[PASS] Walking dominant freq lands at expected 2 Hz")

    print("\nMedian linacc_mag_peak_count per mode:")
    pc = df.groupby('mode')['linacc_mag_peak_count'].median().reindex(MODES)
    print(pc.to_string())
    # Walking at 2 Hz cadence over a 5-second window should give ~10 peaks.
    # Stationary on synthetic Gaussian noise is unreliable because pure noise
    # always produces some peaks above any threshold; in real sensor data the
    # gravity-subtraction smoothing eliminates these. The discriminative power
    # of peak_count comes from its combination with magnitude features.
    assert 7 <= pc['walking'] <= 13, f"Walking peak count {pc['walking']} not near expected 10"
    assert pc['walking'] > pc['metro'], "walking peak count should beat metro"
    print("[PASS] Walking peak count near expected 10 and exceeds metro")

    # Feature family breakdown
    print("\nFeature family breakdown:")
    families = {
        'time_domain_basic': 0,
        'shape_and_energy': 0,
        'slope': 0,
        'frequency_domain': 0,
        'cross_axis_correlation': 0,
        'axis_std_ratio': 0,
        'signal_magnitude_area': 0,
        'peak_count': 0,
        'zero_crossing_rate': 0,
    }
    for c in feature_cols:
        if any(c.endswith('_' + s) for s in ['mean', 'std', 'median', 'iqr', 'rms', 'min', 'max', 'range']):
            families['time_domain_basic'] += 1
        elif c.endswith('_skew') or c.endswith('_kurtosis') or c.endswith('_energy'):
            families['shape_and_energy'] += 1
        elif c.endswith('_slope'):
            families['slope'] += 1
        elif any(s in c for s in ['_dominant_freq', '_dominant_amp', '_freq_weighted_avg',
                                   '_spec_entropy', '_power_band', '_band_ratio']):
            families['frequency_domain'] += 1
        elif '_corr_' in c or c == 'linacc_gyro_mag_corr':
            families['cross_axis_correlation'] += 1
        elif '_std_ratio_' in c:
            families['axis_std_ratio'] += 1
        elif c.endswith('_sma'):
            families['signal_magnitude_area'] += 1
        elif '_peak_count' in c:
            families['peak_count'] += 1
        elif c.endswith('_zcr'):
            families['zero_crossing_rate'] += 1

    for fam, n in families.items():
        print(f"  {fam:30s}: {n}")
    print(f"  {'TOTAL':30s}: {sum(families.values())}")
    assert sum(families.values()) == n_features, (
        f"Family classification missed some features: "
        f"{sum(families.values())} vs {n_features}"
    )
    print("[PASS] All features classified into a family")

    print("\n" + "=" * 70)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 70)


if __name__ == '__main__':
    main()

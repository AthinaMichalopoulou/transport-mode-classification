"""Preprocess phyphox raw exports into per-session CSVs.

Mirrors the documented policy of notebook 02 in a reusable function so that
notebook 09 can preprocess external participants the same way internal
participants were preprocessed during model training.

Policy (matches notebook 02):
  - Resample all sensors onto a 50 Hz common grid via linear interpolation.
  - Trim 10 seconds from each end of every session.
  - Three-tier gap policy applied per sensor: short gaps (< 0.2 s) silently
    interpolated, medium gaps (0.2 to 0.5 s) interpolated and flagged, long
    gaps (> 0.5 s) interpolated but the samples are marked `bad_sample`.
    Windows containing any bad_sample are dropped at feature time.

If this function's output ever diverges from notebook 02's output on internal
data, the smoke test in notebook 09 will catch it and the user can adjust
the policy parameters here before running on externals.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


SAMPLING_RATE = 50.0           # Hz, common grid
EDGE_TRIM_SECONDS = 10.0       # trimmed from each end of every session
GAP_SHORT_S = 0.2              # gaps <= this are silent
GAP_MEDIUM_S = 0.5             # gaps in (short, medium] are flagged; > medium are bad

# Sensor file name patterns (case-insensitive substring match on filename)
SENSOR_PATTERNS = {
    'linear_acc':   [r'linear[\s_]*accel(?:erometer|eration)'],
    'gyroscope':    [r'gyroscope', r'\bgyro\b'],
    'accelerometer': [r'^accelerometer', r'^accel\.csv$'],
    'barometer':    [r'barometer', r'pressure'],
}

# How to rename sensor columns to the canonical schema used downstream
SENSOR_COLUMN_MAP = {
    'linear_acc': {
        'x': 'lin_ax', 'y': 'lin_ay', 'z': 'lin_az',
    },
    'gyroscope': {
        'x': 'gx', 'y': 'gy', 'z': 'gz',
    },
    'accelerometer': {
        'x': 'acc_x', 'y': 'acc_y', 'z': 'acc_z',
    },
    'barometer': {
        'pressure': 'pressure',
        'p': 'pressure',
        'x': 'pressure',  # phyphox sometimes stores pressure as 'X' column
    },
}


def find_sensor_files(session_dir: Path) -> Dict[str, Path]:
    """Find sensor CSVs in a session directory.

    Tolerant to iOS vs Android phyphox export naming. Returns a dict mapping
    canonical sensor names (linear_acc, gyroscope, accelerometer, barometer)
    to CSV paths. Missing sensors are silently absent from the dict.
    """
    files: Dict[str, Path] = {}
    csvs = list(session_dir.glob('*.csv'))
    for sensor_name, patterns in SENSOR_PATTERNS.items():
        for csv in csvs:
            name_lower = csv.name.lower()
            for pat in patterns:
                if re.search(pat, name_lower):
                    files[sensor_name] = csv
                    break
            if sensor_name in files:
                break
    return files


def load_sensor_csv(file_path: Path, sensor_name: str) -> pd.DataFrame:
    """Load a phyphox sensor CSV and standardize columns.

    Returns a DataFrame with a `t` column (seconds, from session start) plus
    the canonical channel columns for this sensor.

    Auto-detects comma vs tab vs semicolon delimiter because phyphox uses
    different formats depending on platform and locale (some Android exports
    use tabs with quoted column names; iOS typically uses commas). Strips
    quote characters from column names so substring matching downstream works.
    """
    # Auto-detect delimiter by trying common options and picking the one
    # that produces more than one column.
    df = None
    for sep in [',', '\t', ';']:
        try:
            test = pd.read_csv(file_path, sep=sep, nrows=2)
            if len(test.columns) > 1:
                df = pd.read_csv(file_path, sep=sep)
                break
        except Exception:
            continue
    if df is None:
        # Last resort: let pandas sniff the delimiter (slower, Python engine)
        df = pd.read_csv(file_path, sep=None, engine='python')

    # Strip quotes that some phyphox exports leave in column names
    df.columns = [str(c).strip().strip('"').strip("'") for c in df.columns]

    # Find the time column
    time_col = None
    for col in df.columns:
        if 'time' in col.lower() or col.strip().lower() == 't':
            time_col = col
            break
    if time_col is None:
        raise ValueError(f"No time column in {file_path.name}")
    df = df.rename(columns={time_col: 't'})

    # Rename the data columns. Phyphox uses headers like "Linear Acceleration x (m/s^2)"
    # or "X (m/s^2)" depending on platform and version.
    rename: Dict[str, str] = {}
    targets = SENSOR_COLUMN_MAP.get(sensor_name, {})

    for col in df.columns:
        if col == 't':
            continue
        col_lower = col.lower().strip()

        # Try axis matching (most common case)
        matched = False
        for axis_key, canonical in targets.items():
            if axis_key == 'pressure':
                # Pressure sensor: any non-time column maps to pressure
                if 'press' in col_lower or col_lower.startswith('p ') or col_lower == 'p':
                    rename[col] = canonical
                    matched = True
                    break
            else:
                # Axis match: look for ' x ', ' y ', ' z ' or column starting with X/Y/Z
                axis_letter = axis_key  # 'x', 'y', or 'z'
                if (f' {axis_letter} ' in col_lower
                        or f'_{axis_letter}' in col_lower
                        or col_lower.startswith(f'{axis_letter} ')
                        or col_lower.endswith(f' {axis_letter}')
                        or col_lower == axis_letter):
                    rename[col] = canonical
                    matched = True
                    break

        # Barometer fallback: a single non-time column becomes pressure
        if not matched and sensor_name == 'barometer':
            non_time_cols = [c for c in df.columns if c != 't']
            if len(non_time_cols) == 1:
                rename[col] = 'pressure'

    df = df.rename(columns=rename)

    # Keep only canonical columns + t
    canonical_set = set(SENSOR_COLUMN_MAP.get(sensor_name, {}).values())
    keep = ['t'] + [c for c in df.columns if c in canonical_set]
    df = df[keep].copy()

    # Drop rows with NaN time or any NaN data
    df = df.dropna()
    return df.reset_index(drop=True)


def preprocess_session(
    session_dir: Path,
    output_path: Path,
    target_fs: float = SAMPLING_RATE,
    trim_seconds: float = EDGE_TRIM_SECONDS,
    gap_short: float = GAP_SHORT_S,
    gap_medium: float = GAP_MEDIUM_S,
    require_sensors: Tuple[str, ...] = ('linear_acc', 'gyroscope'),
) -> dict:
    """Preprocess one phyphox session into a per-session CSV.

    Parameters
    ----------
    session_dir : Path
        Folder containing the sensor CSVs (Linear Accelerometer.csv, Gyroscope.csv, etc.)
    output_path : Path
        Where to write the combined per-session CSV.
    target_fs : float
        Sampling rate for the common grid (default 50 Hz).
    trim_seconds : float
        Seconds to trim from each edge (default 10 s).
    gap_short, gap_medium : float
        Three-tier gap policy thresholds in seconds.
    require_sensors : tuple of str
        Sensors that must be present for preprocessing to succeed.

    Returns
    -------
    dict
        Summary stats: n_samples, duration_s, n_bad, n_flagged, sensors_used.

    Raises
    ------
    FileNotFoundError
        If any required sensor is missing.
    ValueError
        If the session is too short after trimming.
    """
    sensor_files = find_sensor_files(session_dir)
    missing = [s for s in require_sensors if s not in sensor_files]
    if missing:
        raise FileNotFoundError(
            f"Missing required sensors {missing} in {session_dir.name}; "
            f"found {list(sensor_files.keys())}"
        )

    # Load each sensor
    sensors: Dict[str, pd.DataFrame] = {}
    for name, path in sensor_files.items():
        try:
            sensors[name] = load_sensor_csv(path, name)
        except Exception as e:
            # Drop sensors that fail to load (e.g. barometer with weird format),
            # but only if not required
            if name in require_sensors:
                raise
            continue

    # Determine common timeline (intersect available time ranges)
    t_min = max(s['t'].min() for s in sensors.values())
    t_max = min(s['t'].max() for s in sensors.values())
    t_min += trim_seconds
    t_max -= trim_seconds
    if t_max - t_min < 5.0:
        raise ValueError(
            f"Session {session_dir.name} too short after trimming "
            f"({t_max - t_min:.2f} s remaining)"
        )

    # Build 50 Hz grid
    dt = 1.0 / target_fs
    timeline = np.arange(t_min, t_max, dt)

    # Resample each sensor onto the grid via linear interpolation
    out = pd.DataFrame({'t': timeline})
    for sensor_name, df in sensors.items():
        t_src = df['t'].values
        for col in df.columns:
            if col == 't':
                continue
            out[col] = np.interp(timeline, t_src, df[col].values)

    # Three-tier gap policy.
    # Match notebook 02's behavior: gap detection uses ONE primary IMU sensor,
    # never the barometer. Barometer's native low sampling rate (around 1 Hz)
    # would otherwise mark every sample as bad. Preference order matches
    # notebook 02's REQUIRED_SENSORS iteration: accelerometer first, then
    # linear accelerometer, then gyroscope.
    flagged = np.zeros(len(timeline), dtype=bool)
    bad = np.zeros(len(timeline), dtype=bool)

    primary_preference = ['accelerometer', 'linear_acc', 'gyroscope']
    primary_sensor_name = next(
        (s for s in primary_preference if s in sensors), None
    )
    if primary_sensor_name is None:
        raise ValueError(
            f"No IMU sensor available for gap detection in {session_dir.name}"
        )

    primary_t = sensors[primary_sensor_name]['t'].values
    if len(primary_t) >= 2:
        gaps = np.diff(primary_t)
        for i, gap in enumerate(gaps):
            if gap <= gap_short:
                continue
            t_a, t_b = primary_t[i], primary_t[i + 1]
            mask = (timeline >= t_a) & (timeline <= t_b)
            if gap > gap_medium:
                bad[mask] = True
            else:
                flagged[mask] = True

    out['flagged_sample'] = flagged.astype(np.int8)
    out['bad_sample'] = bad.astype(np.int8)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    return {
        'n_samples':    int(len(out)),
        'duration_s':   float(t_max - t_min),
        'n_bad':        int(bad.sum()),
        'n_flagged':    int(flagged.sum()),
        'sensors_used': sorted(sensors.keys()),
        'fs':           float(target_fs),
    }


def discover_external_sessions(
    external_raw_dir: Path,
) -> list:
    """Find all session folders under data/raw/external/<participant>/.

    Returns a list of dicts with keys: participant, mode, day, session_id, path.
    Tolerant to either of these layouts:

      a) external/<participant>/<participant>_<mode>_<day>/<sensor csvs>
      b) external/<participant>/<mode>/<sensor csvs>          (then day defaults to 'day1')
    """
    sessions = []
    for participant_dir in sorted(external_raw_dir.iterdir()):
        if not participant_dir.is_dir():
            continue
        participant = participant_dir.name.lower()

        for sub in sorted(participant_dir.iterdir()):
            if not sub.is_dir():
                continue

            name = sub.name
            # Try pattern: <participant>_<mode>_<day>
            parts = name.lower().replace('-', '_').split('_')
            mode = None
            day = 'day1'
            if len(parts) >= 3 and parts[0] == participant:
                mode = parts[1]
                day = parts[2]
            elif len(parts) == 1:
                # Just a mode name like "walking"
                mode = parts[0]
                day = 'day1'
            elif len(parts) == 2:
                # Could be mode_day or participant_mode
                if parts[0] == participant:
                    mode = parts[1]
                else:
                    mode = parts[0]
                    day = parts[1] if 'day' in parts[1] else 'day1'

            if mode is None:
                continue
            session_id = f"{participant}_{mode}_{day}"
            sessions.append({
                'participant': participant,
                'mode':        mode,
                'day':         day,
                'session_id':  session_id,
                'path':        sub,
            })
    return sessions

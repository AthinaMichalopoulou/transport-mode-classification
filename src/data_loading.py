"""Session inventory and metadata parsing.

Public API:
    parse_session_id(stem) -> dict
        Extract participant, mode, day from a filename stem.
    build_session_index(interim_dir) -> list of dicts
        Discover all session CSVs in a directory and build an inventory.
"""

from __future__ import annotations

from pathlib import Path

from src.features import MODES


def parse_session_id(stem: str) -> dict:
    """Extract participant, mode, day from a session_id like 'babis_walking_day1'.

    Returns a dict with keys 'participant', 'mode', 'day'. Any field that
    cannot be parsed is set to None.
    """
    parts = stem.lower().split('_')
    if len(parts) < 3:
        return {'participant': None, 'mode': None, 'day': None}
    participant = parts[0]
    mode = None
    day = None
    for token in parts[1:]:
        if token in MODES:
            mode = token
        elif token.startswith('day'):
            day = token
    return {'participant': participant, 'mode': mode, 'day': day}


def build_session_index(interim_dir: Path) -> list:
    """Discover session CSVs in interim_dir and return a list of metadata dicts.

    Each dict has keys: session_id, path, participant, mode, day.
    """
    interim_dir = Path(interim_dir)
    session_files = sorted(interim_dir.glob('*.csv'))
    session_index = []
    for fp in session_files:
        meta = parse_session_id(fp.stem)
        session_index.append({'session_id': fp.stem, 'path': str(fp), **meta})
    return session_index

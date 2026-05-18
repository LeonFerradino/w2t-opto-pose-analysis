# -*- coding: utf-8 -*-
"""
pose_config.py — Single source of truth for the pose-kinematics analysis pipeline.

All paths, session metadata, keypoint registries, confidence thresholds, and
plot constants live here. No analysis logic.

Usage:
    from pose_config import (PROCESSED_PATH, MICE, build_session_registry,
                             SLEAP_KEYPOINTS, CONFIDENCE_THRESHOLDS, MOUSE_COLORS)
"""

from pathlib import Path
import os
import re as _re

from session_config import mouse_sessions as _SC_MOUSE_SESSIONS

# ── Paths ─────────────────────────────────────────────────────────────────────
PROCESSED_PATH = Path("f:/analysis/bpod/Documents/processed")
FIGURES_PATH   = Path("f:/analysis/bpod/Documents/figures/pose")
RESULTS_PATH   = Path("f:/analysis/bpod/Documents/results/pose")
BPOD_PATH      = Path("f:/analysis/bpod/Documents/bpod")

# Create output directories on import (safe no-op if they exist)
for _p in (FIGURES_PATH / "sync_qc",
           FIGURES_PATH / "trial_aligned",
           FIGURES_PATH / "learning",
           FIGURES_PATH / "opto",
           RESULTS_PATH):
    _p.mkdir(parents=True, exist_ok=True)


# ── Mouse registry ────────────────────────────────────────────────────────────
# IDs as used in the pickle filenames (underscore, not parenthesis)
MICE = [
    "SNA-145894_1",
    "MLA-026805_2",
    "MLA-026806_3",
    "MLA-026807_4",
]

# All calendar dates that could carry training sessions (11 days, no suffix)
TRAINING_DATES = [
    "20250905", "20250906", "20250907", "20250908", "20250909",
    "20250910", "20250911", "20250912", "20250913", "20250914", "20250915",
]

# All calendar dates that could carry opto sessions (_1 / _2 suffix)
OPTO_DATES = [
    "20250916", "20250917", "20250918", "20250919",
    "20250922", "20250923", "20250924", "20250925",
]

# Suffix → phase label
# _1 = blocked opto schedule, _2 = random opto schedule
OPTO_SUFFIX_MAP = {"1": "opto_blocked", "2": "opto_random"}

# ── Opto duration lookup ───────────────────────────────────────────────────────
# session_config.mouse_sessions is the authoritative source for which .mat
# files correspond to 0.5 s vs 2 s stimulation.  The suffix alone is NOT
# sufficient: e.g. SNA-145894 has opto_2s sessions run under the blocked
# protocol (suffix "1").  We extract the date from each listed path and build
# a {(mouse_short, date): opto_duration_s} lookup used by build_session_registry.

def _build_opto_duration_lookup() -> dict:
    _dur_map = {"opto_0.5s": 0.5, "opto_2s": 2.0}
    lookup: dict[tuple[str, str], float] = {}
    for mouse_short, areas in _SC_MOUSE_SESSIONS.items():
        for area_data in areas.values():
            for cond_key, paths in area_data.items():
                dur = _dur_map.get(cond_key)
                if dur is None:
                    continue            # skip 'W2T' entries
                for path in paths:
                    m = _re.search(r"_(\d{8})_", path)
                    if m:
                        # setdefault: first match wins (no date should conflict)
                        lookup.setdefault((mouse_short, m.group(1)), dur)
    return lookup

_OPTO_DURATION_LOOKUP: dict[tuple[str, str], float] = _build_opto_duration_lookup()


# ── Pose source index map ─────────────────────────────────────────────────────
# pose_estimations is a flat list of 7 dicts; map by (source, video_idx)
# SLEAP:    video 0 → idx 0 | video 1 → idx 1 | video 2 → idx 2
# DLC:      video 0 → idx 3 | video 1 → idx 4
# Facemap:  video 0 → idx 5 | video 1 → idx 6
POSE_SOURCE_INDEX = {
    ("sleap",   0): 0,
    ("sleap",   1): 1,
    ("sleap",   2): 2,
    ("dlc",     0): 3,
    ("dlc",     1): 4,
    ("facemap", 0): 5,
    ("facemap", 1): 6,
}


# ── Keypoint registries ───────────────────────────────────────────────────────
SLEAP_KEYPOINTS = [
    "nosetip", "noseback", "pupilL", "earbaseL", "eartipL",
    "lowerlip", "tm_joint", "frontpawL", "frontpawR", "shoulder",
]

DLC_KEYPOINTS = [
    "C1_end", "C1_mid", "C1_start",
    "C1_end_reverse", "C1_mid_reverse", "C1_start_reverse",
    "C2_end", "C2_mid", "C2_start",
    "C2_end_reverse", "C2_mid_reverse", "C2_start_reverse",
    "Opto_light_in", "Opto_light_out",
    "nose", "sensor_bottom", "sensor_top", "trial_light",
]

# DLC keypoints used for kinematic analysis (excludes sync/opto LED markers)
DLC_ANALYSIS_KEYPOINTS = [
    "C1_end", "C1_mid", "C1_start",
    "C1_end_reverse", "C1_mid_reverse", "C1_start_reverse",
    "C2_end", "C2_mid", "C2_start",
    "C2_end_reverse", "C2_mid_reverse", "C2_start_reverse",
    "nose", "sensor_bottom", "sensor_top",
]

# Per-source confidence/score key name in the keypoints dict
CONFIDENCE_KEY = {
    "sleap": "score",
    "dlc":   "confidence",
    "facemap": None,   # no confidence — pupil_size is a scalar
}

# Minimum acceptable confidence/score to keep a frame (NaN otherwise)
CONFIDENCE_THRESHOLDS = {
    "sleap":   0.6,
    "dlc":     0.9,    # higher: used for sync QC where precision matters
    "facemap": None,
}


# ── Trial-alignment parameters ────────────────────────────────────────────────
TRIAL_WINDOW_SEC    = (-0.5, 2.0)   # (pre, post) relative to Audio state onset
BASELINE_WINDOW_SEC = (-0.5, 0.0)   # window used for per-trial baseline subtraction
RESAMPLE_HZ         = 150.0         # camera frame rate; used as output grid

# Hit-aligned window (Section 4.2 kinematics figures)
# Epoch extracted around the HIT state onset (sensor-touch moment).
# Baseline = pre-cue window, well before the animal initiates movement.
HIT_WINDOW_SEC   = (-1.0, 1.0)   # (pre, post) relative to HIT state onset
HIT_BASELINE_SEC = (-1.0, -0.5)  # baseline subtraction window (pre-cue)

# Bpod state / event names
ALIGNMENT_STATE = "Audio"           # anchor for trial alignment
HIT_STATE       = "HIT"
MISS_STATE      = "Miss"
REWARD_STATE    = "Reward"
LICK_EVENTS     = ["Flex1Trig1", "Flex1Trig2"]

# TrialType codes (task_arguments['TrialTypes'])
TRIAL_TYPE_W2T  = 1
TRIAL_TYPE_OPTO = 2

# ── Publication keypoints (Section 4.2) ───────────────────────────────────────
# Four keypoints used in the main kinematics figures (Figures A–E).
# Each entry: source, video index, keypoint name(s), coordinate type.
# coord "y"     → y-position from extract_keypoint_timeseries
# coord "angle" → atan2(kp_end − kp_start) via extract_whisker_angle_timeseries
#
# Contralateral ("_reverse") keypoints track the non-executing whisker
# imaged from the opposite camera angle (confirmed by lab convention).
PUB_KEYPOINTS: dict[str, dict] = {
    "nose_y": dict(source="dlc", vid=0, kp="nose", coord="y"),
    "C2_angle": dict(
        source="dlc", vid=0,
        kp_start="C2_start", kp_end="C2_end", coord="angle",
    ),
    "C1r_angle": dict(
        source="dlc", vid=0,
        kp_start="C2_start_reverse", kp_end="C2_end_reverse", coord="angle",
    ),
    "fpaw_y": dict(
        source="sleap", vid=0, kp="frontpawL", coord="y",
    ),
}

PUB_KEYPOINT_LABELS: dict[str, str] = {
    "nose_y":    "Nose (y-pos)",
    "C2_angle":  "C2 whisker (°)",
    "C1r_angle": "Contra whisker (°)",
    "fpaw_y":    "Forepaw (y-pos)",
}


# ── Sync QC thresholds ────────────────────────────────────────────────────────
SYNC_DLC_KEYPOINT         = "sensor_top"   # DLC keypoint used for lick-sync QC
SYNC_MATCH_WINDOW_SEC     = 0.200          # max allowed offset for nearest-neighbour match
SYNC_FLAG_MEAN_OFFSET_SEC = 0.050          # flag session if |mean_offset| exceeds this
SYNC_FLAG_STD_SEC         = 0.030          # flag session if std_offset exceeds this
SYNC_FLAG_MATCH_RATE      = 0.70           # flag session if < 70% of events matched


# ── Brain area config — per mouse per date ────────────────────────────────────
# Key level 1 = mouse ID (pickle format, underscore)
# Key level 2 = date string YYYYMMDD
# Value        = inhibited brain area label, or EXCLUDE_AREA to skip from plots
#
# Source: inhibition_table_by_mouse_updated (Word file, 2025-09-xx dates)
# Note: MLA-026805_2 / 20250918 = A1 → marked EXCLUDE_AREA, not plotted.

EXCLUDE_AREA = "__EXCLUDE__"   # sentinel: sessions with this area are skipped

BRAIN_AREA_CONFIG: dict[str, dict[str, str]] = {
    "SNA-145894_1": {
        "20250916": "M1",
        "20250917": "S1",
        "20250918": "M2",
        "20250919": "mPFC",
        "20250922": "M1",
        "20250923": "S1",
        "20250924": "M2",
        "20250925": "mPFC",
    },
    "MLA-026805_2": {
        "20250916": "M1",
        "20250917": "M1",
        "20250918": EXCLUDE_AREA,   # A1 — excluded from plots per user instruction
    },
    "MLA-026806_3": {
        "20250916": "M1",
        "20250917": "S1",
        "20250918": "M2",
        "20250919": "mPFC",
        "20250922": "M1",
        "20250923": "S1",
        "20250924": "M2",
        "20250925": "mPFC",
    },
    "MLA-026807_4": {
        "20250916": "M1",
        "20250917": "M2",
        "20250918": "S1",
        "20250919": "mPFC",
        "20250922": "M1",
        "20250923": "S1",
        "20250924": "M2",
        "20250925": "mPFC",
    },
}

# Ordered list of brain areas to use as plot columns (excludes EXCLUDE_AREA)
BRAIN_AREAS_ORDERED = ["M1", "S1", "M2", "mPFC"]


def get_brain_area(mouse: str, date: str) -> str | None:
    """
    Return the inhibited brain area for a given mouse × date combination.

    Parameters
    ----------
    mouse : str
        Mouse ID in pickle format, e.g. "MLA-026805_2".
    date : str
        Date string YYYYMMDD, e.g. "20250916".

    Returns
    -------
    str or None
        Brain area label (e.g. "M1"), EXCLUDE_AREA sentinel if this session
        should be excluded, or None if the (mouse, date) pair is not in the
        config (training sessions, unknown dates).
    """
    return BRAIN_AREA_CONFIG.get(mouse, {}).get(date, None)


# ── Plot constants ─────────────────────────────────────────────────────────────
# Match the MOUSE_COLORS used in the bpod analysis scripts,
# but keyed to the underscore IDs used in pickle filenames.
MOUSE_COLORS = {
    "SNA-145894_1": "#d62728",
    "MLA-026805_2": "#1f77b4",
    "MLA-026806_3": "#ff7f0e",
    "MLA-026807_4": "#2ca02c",
}

MOUSE_MARKERS = {
    "SNA-145894_1": "o",
    "MLA-026805_2": "s",
    "MLA-026806_3": "^",
    "MLA-026807_4": "D",
}

HIT_COLOR   = "#E74C3C"
MISS_COLOR  = "#95A5A6"
OPTO_COLOR  = "#9B59B6"
W2T_COLOR   = "#2ECC71"

PHASE_COLORS = {
    "training":    "#3498DB",
    "opto_blocked": "#E67E22",
    "opto_random":  "#9B59B6",
}

FIGURE_SIZE_WIDE   = (14, 4)
FIGURE_SIZE_SQUARE = (6, 6)
FIGURE_SIZE_PANEL  = (12, 8)


# ── Excluded sessions ─────────────────────────────────────────────────────────
# Set of (mouse, date, suffix) tuples.
# suffix=None for training sessions; "1"/"2" for opto sessions.
#
# Reasons:
#   SNA-145894_1 opto_random (suffix="2") 20250917–20250925:
#       Sync collapse — DLC trial_light edges detected but 0 matches with Bpod
#       audio onsets (clock offset > 200 ms). All 7 random sessions unusable.
#   MLA-026805_2 20250918 suffix "1" and "2":
#       No trial_light signal (n_edges = 0) — LED not recorded.
#   MLA-026806_3 20250906 training (no suffix):
#       No trial_light signal (n_edges = 0) in training pickle.
#   MLA-026806_3 20250922 suffix "2":
#       No trial_light signal (n_edges = 0).
EXCLUDED_SESSIONS: set[tuple] = {
    # SNA-145894_1 — all opto_random sessions
    ("SNA-145894_1", "20250917", "2"),
    ("SNA-145894_1", "20250918", "2"),
    ("SNA-145894_1", "20250919", "2"),
    ("SNA-145894_1", "20250922", "2"),
    ("SNA-145894_1", "20250923", "2"),
    ("SNA-145894_1", "20250924", "2"),
    ("SNA-145894_1", "20250925", "2"),
    # MLA-026805_2 — no LED signal on 20250918
    ("MLA-026805_2", "20250918", "1"),
    ("MLA-026805_2", "20250918", "2"),
    # MLA-026806_3 — no LED signal
    ("MLA-026806_3", "20250906", None),
    ("MLA-026806_3", "20250922", "2"),
}


# ── All-features builder ──────────────────────────────────────────────────────

def build_all_features() -> dict:
    """
    Build the complete feature dictionary for pose analysis.

    Covers every body part from every camera for all three pose sources:
        SLEAP:   cameras 0–2, all 10 keypoints, coords x / y / speed
        DLC:     cameras 0–1, 15 analysis keypoints, coords x / y / speed
        Facemap: cameras 0–1, pupil size scalar

    Naming convention:
        SLEAP   → "s{vid}_{keypoint}_{coord}"   e.g. "s0_nosetip_speed"
        DLC     → "dlc{vid}_{keypoint}_{coord}" e.g. "dlc0_nose_x"
        Facemap → "fm{vid}_pupil"               e.g. "fm0_pupil"

    Returns
    -------
    dict
        feature_name → (source, video_idx, keypoint_name, coord)
        Compatible with compute_session_trial_kinematics() and all
        downstream analysis functions.

    Total features: 3×10×3 (SLEAP) + 2×15×3 (DLC) + 2 (Facemap) = 182
    """
    features = {}
    for vid in range(3):
        for kp in SLEAP_KEYPOINTS:
            for coord in ("x", "y", "speed"):
                features[f"s{vid}_{kp}_{coord}"] = ("sleap", vid, kp, coord)
    for vid in range(2):
        for kp in DLC_ANALYSIS_KEYPOINTS:
            for coord in ("x", "y", "speed"):
                features[f"dlc{vid}_{kp}_{coord}"] = ("dlc", vid, kp, coord)
    for vid in range(2):
        features[f"fm{vid}_pupil"] = ("facemap", vid, None, "pupil")
    return features


# ── Bpod raw file mapping ─────────────────────────────────────────────────────
# Mouse IDs in pickle format → folder names used by Bpod (parentheses notation)
MOUSE_BPOD_FOLDER = {
    "SNA-145894_1": "SNA-145894_(1)",
    "MLA-026805_2": "MLA-026805_(2)",
    "MLA-026806_3": "MLA-026806_(3)",
    "MLA-026807_4": "MLA-026807_(4)",
}

# Side (left/right) used in opto folder names per mouse
MOUSE_OPTO_SIDE = {
    "SNA-145894_1": "left",
    "MLA-026805_2": "right",
    "MLA-026806_3": "right",
    "MLA-026807_4": "left",
}


def find_bpod_mat_files(mouse: str, date: str, suffix: str | None) -> list:
    """
    Return sorted list of Bpod .mat file Paths for a given session.

    Parameters
    ----------
    mouse  : str   e.g. "MLA-026805_2"
    date   : str   YYYYMMDD
    suffix : str|None   None=training, "1"=opto_blocked, "2"=opto_random

    Returns
    -------
    list[Path]  May be empty if no files found.
    """
    folder_name = MOUSE_BPOD_FOLDER.get(mouse)
    if folder_name is None:
        return []

    if suffix is None:
        # Training sessions: W2T_right or W2T_left
        for side in ("right", "left"):
            for subfolder in ("Training", "training"):
                session_dir = BPOD_PATH / folder_name / subfolder / f"W2T_{side}" / "Session Data"
                if session_dir.exists():
                    files = sorted(session_dir.glob(f"*_{date}_*.mat"))
                    if files:
                        return files
        return []
    else:
        side = MOUSE_OPTO_SIDE.get(mouse, "right")
        if suffix == "1":
            folder = f"W2T_Opto_{side}_blocked"
        else:
            folder = f"W2T_Opto_{side}"
        session_dir = BPOD_PATH / folder_name / "Expert" / folder / "Session Data"
        if not session_dir.exists():
            return []
        return sorted(session_dir.glob(f"*_{date}_*.mat"))


# ── Session registry ──────────────────────────────────────────────────────────

def _make_pickle_path(mouse: str, date: str, suffix: str | None) -> Path:
    """Construct the expected pickle file path for a session."""
    if suffix is None:
        fname = f"{mouse}_{date}.pickle"
    else:
        fname = f"{mouse}_{date}_{suffix}.pickle"
    return PROCESSED_PATH / mouse / fname


def build_session_registry(exist_only: bool = True) -> list[dict]:
    """
    Build the full list of SessionRecord dicts for all mice × sessions.

    Each record is a plain dict with keys:
        mouse       str   e.g. "MLA-026805_2"
        date        str   YYYYMMDD
        suffix      str|None  None for training, "1" or "2" for opto
        phase       str   "training" | "opto_blocked" | "opto_random"
        day_index   int   1-based day number within the phase (per mouse)
        pickle_path Path  absolute path to the .pickle file

    Day index counts only sessions that actually exist on disk (within-phase,
    per mouse, chronologically sorted).

    Parameters
    ----------
    exist_only : bool
        If True (default), only include records where the pickle file exists.
        Set False to get the full candidate list regardless of existence.

    Returns
    -------
    list[dict]
        Sorted by (mouse, phase, day_index).
    """
    candidates = []

    for mouse in MICE:
        # "SNA-145894_1" → "SNA-145894" for session_config lookup
        mouse_short = _re.sub(r"_\d+$", "", mouse)

        # Training sessions (no suffix)
        for date in TRAINING_DATES:
            p = _make_pickle_path(mouse, date, None)
            candidates.append({
                "mouse":           mouse,
                "date":            date,
                "suffix":          None,
                "phase":           "training",
                "day_index":       -1,       # filled below
                "opto_duration_s": None,
                "pickle_path":     p,
            })

        # Opto sessions (_1 and _2)
        for date in OPTO_DATES:
            for sfx in ("1", "2"):
                p = _make_pickle_path(mouse, date, sfx)
                dur = _OPTO_DURATION_LOOKUP.get((mouse_short, date))
                candidates.append({
                    "mouse":           mouse,
                    "date":            date,
                    "suffix":          sfx,
                    "phase":           OPTO_SUFFIX_MAP[sfx],
                    "day_index":       -1,
                    "opto_duration_s": dur,  # 0.5, 2.0, or None if not listed
                    "pickle_path":     p,
                })

    # Filter to existing files if requested
    if exist_only:
        candidates = [c for c in candidates if c["pickle_path"].exists()]

    # Always remove known-bad sessions
    candidates = [
        c for c in candidates
        if (c["mouse"], c["date"], c["suffix"]) not in EXCLUDED_SESSIONS
    ]

    # Assign day_index within (mouse, phase) sorted by date
    from itertools import groupby
    key = lambda r: (r["mouse"], r["phase"])
    candidates.sort(key=key)
    for (mouse, phase), group in groupby(candidates, key=key):
        group = list(group)
        group.sort(key=lambda r: r["date"])
        for idx, rec in enumerate(group, start=1):
            rec["day_index"] = idx

    # Final sort: mouse, phase, day_index
    candidates.sort(key=lambda r: (r["mouse"], r["phase"], r["day_index"]))
    return candidates


def get_sessions_for_mouse(mouse: str, phase: str | None = None) -> list[dict]:
    """
    Return all existing sessions for one mouse, optionally filtered by phase.

    Parameters
    ----------
    mouse : str
        One of MICE.
    phase : str or None
        "training", "opto_blocked", "opto_random", or None for all.

    Returns
    -------
    list[dict]
        Ordered by day_index.
    """
    registry = build_session_registry()
    result = [r for r in registry if r["mouse"] == mouse]
    if phase is not None:
        result = [r for r in result if r["phase"] == phase]
    return result


def get_all_sessions(phase: str | None = None) -> list[dict]:
    """
    Return all existing sessions across all mice, optionally filtered by phase.

    Parameters
    ----------
    phase : str or None
        "training", "opto_blocked", "opto_random", or None for all.

    Returns
    -------
    list[dict]
    """
    registry = build_session_registry()
    if phase is not None:
        registry = [r for r in registry if r["phase"] == phase]
    return registry


def check_pupil_coverage(
    sessions: list | None = None,
    n_sample: int = 6,
    threshold: float = 0.5,
) -> bool:
    """
    Return True if Facemap pupil data is present in ≥ threshold fraction of
    sampled sessions, False otherwise.

    Loads up to n_sample pickles (evenly spread across sessions) and checks
    whether the 'facemap' key exists at the top level.  Caller should drop
    all fm0_pupil / fm1_pupil features from KEY_FEATURES lists when this
    returns False.

    Parameters
    ----------
    sessions  : list[dict] or None   Session registry; built if None.
    n_sample  : int                  Max sessions to inspect.
    threshold : float                Fraction threshold (default 0.5 → 50 %).

    Returns
    -------
    bool
    """
    import pickle
    if sessions is None:
        sessions = build_session_registry(exist_only=True)
    if not sessions:
        return False

    step = max(1, len(sessions) // n_sample)
    sampled = sessions[::step][:n_sample]

    n_with = 0
    for rec in sampled:
        try:
            with open(rec["pickle_path"], "rb") as fh:
                d = pickle.load(fh)
            # Facemap data is stored under key "facemap" (list of dicts per cam)
            if "facemap" in d and d["facemap"]:
                n_with += 1
            del d
        except Exception:
            pass

    coverage = n_with / len(sampled) if sampled else 0.0
    return coverage >= threshold

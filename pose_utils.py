# -*- coding: utf-8 -*-
"""
pose_utils.py — Shared low-level utilities for the pose-kinematics pipeline.

Memory contract
---------------
load_session_pickle() is the ONLY function that reads a .pickle from disk.
After extracting what you need, the caller must explicitly release the reference:

    d = load_session_pickle(path)
    result = extract_trial_table(d)
    del d
    gc.collect()

No function here loads a second file while one is open.

Usage:
    from pose_utils import (load_session_pickle, extract_trial_table,
                            get_pose_object, extract_keypoint_timeseries,
                            slice_timeseries_to_trial, compute_trial_speed,
                            build_keypoint_feature_matrix, fit_pca_on_session,
                            run_umap_on_features, compute_sync_alignment_error)
"""

import gc
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.stats import pearsonr

from pose_config import (
    POSE_SOURCE_INDEX, CONFIDENCE_KEY, CONFIDENCE_THRESHOLDS,
    ALIGNMENT_STATE, HIT_STATE, MISS_STATE, REWARD_STATE, LICK_EVENTS,
    TRIAL_TYPE_W2T, TRIAL_TYPE_OPTO,
    TRIAL_WINDOW_SEC, BASELINE_WINDOW_SEC, RESAMPLE_HZ,
    SYNC_MATCH_WINDOW_SEC,
    SYNC_FLAG_MEAN_OFFSET_SEC, SYNC_FLAG_STD_SEC, SYNC_FLAG_MATCH_RATE,
)


# ── Required top-level keys in every session pickle ───────────────────────────
_REQUIRED_KEYS = {
    "subject", "session", "trials_table",
    "task arguments", "pose_estimations",
}


# ═══════════════════════════════════════════════════════════════════════════════
# I/O
# ═══════════════════════════════════════════════════════════════════════════════

def load_session_pickle(pickle_path: Path) -> dict:
    """
    Load a single session pickle file and validate its structure.

    Parameters
    ----------
    pickle_path : Path
        Absolute path to the .pickle file (~400 MB).

    Returns
    -------
    dict
        Raw session dict. Caller must `del` it and call `gc.collect()` when done.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If any required top-level key is missing.
    """
    pickle_path = Path(pickle_path)
    if not pickle_path.exists():
        raise FileNotFoundError(f"Pickle not found: {pickle_path}")

    with open(pickle_path, "rb") as fh:
        d = pickle.load(fh)

    missing = _REQUIRED_KEYS - set(d.keys())
    if missing:
        raise ValueError(f"{pickle_path.name}: missing keys {missing}")

    return d


# ═══════════════════════════════════════════════════════════════════════════════
# Trial structure extraction
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_name(val):
    """
    Extract a plain string name from a possibly-nested DataFrame cell.

    The bpod pipeline stores state_type / event_type as DataFrames with
    columns 'state_name', 'event_name', or 'action_name'. Scalar strings
    are returned as-is.
    """
    if isinstance(val, pd.DataFrame):
        for col in ("state_name", "event_name", "action_name"):
            if col in val.columns:
                return val[col].iloc[0]
        return str(val.iloc[0, 0])
    return val


def extract_state_onset(trial_states_df: pd.DataFrame,
                        state_name: str) -> float | None:
    """
    Return the start_time of the first occurrence of state_name in one
    trial's nested states DataFrame.

    Parameters
    ----------
    trial_states_df : pd.DataFrame
        The nested states DataFrame for one trial (from trials_table['states']).
    state_name : str
        Target state name, e.g. "Audio", "HIT", "Miss".

    Returns
    -------
    float or None
        start_time in seconds (bpod clock), or None if the state is absent.
    """
    for _, row in trial_states_df.iterrows():
        if _extract_name(row.get("state_type", row.iloc[2])) == state_name:
            return float(row["start_time"])
    return None


def extract_lick_times(trial_events_df: pd.DataFrame) -> np.ndarray:
    """
    Return sorted lick event timestamps from one trial's nested events DataFrame.

    Lick events are Flex1Trig1 and Flex1Trig2.

    Parameters
    ----------
    trial_events_df : pd.DataFrame
        The nested events DataFrame for one trial (from trials_table['events']).

    Returns
    -------
    np.ndarray
        Shape (n_licks,). Empty array if no licks found.
    """
    licks = []
    for _, row in trial_events_df.iterrows():
        ename = _extract_name(row.get("event_type", row.iloc[1]))
        if ename in LICK_EVENTS:
            licks.append(float(row["timestamp"]))
    return np.sort(licks)


def load_trial_types_from_mat(mat_paths: list) -> np.ndarray:
    """
    Load and concatenate the per-trial TrialTypes array from one or more
    Bpod .mat files.  Returns np.ndarray of ints (1=W2T, 2=Opto), or
    empty array if files cannot be read.
    """
    from bpod_utils import load_mat
    all_types = []
    for p in mat_paths:
        try:
            data = load_mat(str(p))
            sd   = data.get("SessionData", {})
            tt   = sd.get("TrialTypes", None)
            if tt is not None:
                all_types.append(np.asarray(tt).flatten().astype(int))
        except Exception:
            pass
    return np.concatenate(all_types) if all_types else np.array([], dtype=int)


def extract_trial_table(session_dict: dict,
                        trial_types_array: np.ndarray | None = None) -> pd.DataFrame:
    """
    Build a flat per-trial DataFrame with outcome, state onsets, and lick info.

    Reads the nested structure inside trials_table and resolves state names
    using _extract_name().

    Columns
    -------
    trial_idx       int     0-based index
    trial_type      str     'W2T' or 'Opto'
    outcome         str     'HIT' or 'Miss'
    audio_onset     float   seconds (NaN if Audio state absent)
    reward_onset    float   seconds (NaN if no Reward)
    first_lick_time float   seconds after session start (NaN if no lick in window)
    lick_count      int     number of lick events in the trial
    reaction_time   float   first_lick_time - audio_onset (NaN if miss)

    Rows with no Audio state are included but have NaN audio_onset and are
    printed as a warning.

    Parameters
    ----------
    session_dict : dict
        Raw pickle dict from load_session_pickle().

    Returns
    -------
    pd.DataFrame
        One row per trial.
    """
    tt_df = session_dict["trials_table"]

    # Per-trial TrialType: prefer external array loaded from the Bpod .mat file
    # (SessionData.TrialTypes, 1=W2T, 2=Opto). Falls back to "W2T" for all
    # trials if no array is provided (training sessions, or when .mat unavailable).
    # trial_types_array must be aligned to the *unfiltered* trial rows in tt_df.
    has_external_tt = (trial_types_array is not None and len(trial_types_array) > 0)

    n_empty = 0
    skipped_indices = []
    rows = []
    raw_row_idx = 0   # index into trial_types_array (counts all rows, incl. skipped)
    for i, row in tt_df.iterrows():
        # states nested DF
        states_df = row.get("states", pd.DataFrame())
        if not isinstance(states_df, pd.DataFrame) or states_df.empty:
            n_empty += 1
            raw_row_idx += 1
            continue  # silently skip incomplete/aborted trials

        # Assign trial type from external array if available, else default W2T
        if has_external_tt and raw_row_idx < len(trial_types_array):
            tt_code = trial_types_array[raw_row_idx]
            trial_type = "Opto" if tt_code == 2 else "W2T"
        else:
            trial_type = "W2T"
        raw_row_idx += 1

        # Alignment onset: try "Audio" (new protocol) then "Cue" (older protocol)
        audio_onset = extract_state_onset(states_df, ALIGNMENT_STATE)
        if audio_onset is None:
            audio_onset = extract_state_onset(states_df, "Cue")

        reward_onset = extract_state_onset(states_df, REWARD_STATE)

        # Outcome: Miss state present → Miss; absent → HIT (works for both protocols)
        miss_onset = extract_state_onset(states_df, MISS_STATE)
        outcome = "Miss" if miss_onset is not None else "HIT"

        if audio_onset is None:
            warnings.warn(f"Trial {i}: no Audio/Cue state found")

        # Hit onset: prefer the explicit "HIT" state (blocked opto + training protocol).
        # Fall back to "RewardDelay" onset (random opto protocol, where touching the
        # sensor sends Bpod directly to RewardDelay with no intermediate HIT state).
        # Both represent the moment of sensor contact.
        hit_t = extract_state_onset(states_df, HIT_STATE)
        if hit_t is None:
            hit_t = extract_state_onset(states_df, "RewardDelay")

        # licks
        events_df = row.get("events", pd.DataFrame())
        if isinstance(events_df, pd.DataFrame) and not events_df.empty:
            lick_times = extract_lick_times(events_df)
        else:
            lick_times = np.array([])

        # Only use licks AFTER audio onset — the first touch in the trial
        # is the self-initiation touch (before the cue), not the response.
        if audio_onset is not None and len(lick_times) > 0:
            post_audio_licks = lick_times[lick_times > audio_onset]
        else:
            post_audio_licks = np.array([])
        first_lick = float(post_audio_licks[0]) if len(post_audio_licks) > 0 else np.nan
        rt = (first_lick - audio_onset
              if (audio_onset is not None and not np.isnan(first_lick))
              else np.nan)

        rows.append({
            "trial_idx":       i,
            "trial_type":      trial_type,
            "outcome":         outcome,
            "audio_onset":     audio_onset if audio_onset is not None else np.nan,
            "hit_onset":       hit_t if hit_t is not None else np.nan,
            "reward_onset":    reward_onset if reward_onset is not None else np.nan,
            "first_lick_time": first_lick,
            "lick_count":      len(lick_times),
            "reaction_time":   rt,
        })

    if n_empty:
        warnings.warn(f"{n_empty} trials skipped (empty states — likely aborted trials)")

    return pd.DataFrame(rows).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Pose data extraction
# ═══════════════════════════════════════════════════════════════════════════════

def get_pose_object(session_dict: dict,
                    source: str,
                    video_idx: int) -> dict | None:
    """
    Retrieve one pose estimation object by matching source name and video index.

    Each entry in pose_estimations has 'source' (e.g. "SLEAP", "DLC", "Facemap")
    and 'video' (int camera index) fields. Matching is case-insensitive on source.

    Parameters
    ----------
    session_dict : dict
    source : str
        "sleap", "dlc", or "facemap" (case-insensitive).
    video_idx : int
        Camera index within that source (0, 1, or 2 for SLEAP; 0 or 1 for others).

    Returns
    -------
    dict or None
        The pose estimation dict, or None if not found.
    """
    poses = session_dict.get("pose_estimations", [])
    source_lower = source.lower()
    for p in poses:
        if p.get("source", "").lower() == source_lower and p.get("video") == video_idx:
            return p
    warnings.warn(f"No pose object found for source={source}, video={video_idx} "
                  f"(session has {len(poses)} pose entries)")
    return None


def get_pose_timestamps(pose_obj: dict) -> np.ndarray:
    """
    Return the timestamp array for a pose estimation object.

    Parameters
    ----------
    pose_obj : dict

    Returns
    -------
    np.ndarray
        Shape (N_frames,). Seconds in the shared bpod/camera time base.
    """
    return np.asarray(pose_obj["timestamp"], dtype=float)


def extract_keypoint_timeseries(
    pose_obj: dict,
    keypoint_name: str,
    confidence_threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract x, y, and confidence score timeseries for one SLEAP or DLC keypoint.

    Frames where confidence < confidence_threshold are NaN-masked.

    Parameters
    ----------
    pose_obj : dict
        A SLEAP or DLC pose estimation dict.
    keypoint_name : str
        Key into the 'keypoints' dict of each frame.
    confidence_threshold : float or None
        Frames with score/confidence below this → x=NaN, y=NaN.
        None disables masking.

    Returns
    -------
    tuple of three np.ndarray, each shape (N_frames,)
        (x_positions, y_positions, confidence_scores)
    """
    frames = pose_obj["data"][0]
    n = len(frames)
    x    = np.full(n, np.nan)
    y    = np.full(n, np.nan)
    conf = np.full(n, np.nan)

    # Determine which confidence key to use
    source = pose_obj.get("source", "").lower()
    conf_key = CONFIDENCE_KEY.get(source, "score")

    for i, frame in enumerate(frames):
        kps = frame.get("keypoints", {})
        if keypoint_name not in kps:
            continue
        kp = kps[keypoint_name]
        xi = kp.get("x", np.nan)
        yi = kp.get("y", np.nan)
        ci = kp.get(conf_key, 1.0) if conf_key else 1.0
        conf[i] = ci
        if confidence_threshold is None or ci >= confidence_threshold:
            x[i] = xi
            y[i] = yi

    return x, y, conf


def extract_pupil_size_timeseries(
    pose_obj: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract pupil_size scalar timeseries from a Facemap pose object.

    Facemap frames have no 'keypoints' dict — only a 'pupil_size' float.

    Parameters
    ----------
    pose_obj : dict
        A Facemap pose estimation dict (source "Facemap").

    Returns
    -------
    tuple of two np.ndarray, each shape (N_frames,)
        (timestamps, pupil_sizes)
    """
    frames = pose_obj["data"][0]
    ts     = get_pose_timestamps(pose_obj)
    sizes  = np.array([float(f.get("pupil_size", np.nan)) for f in frames])
    # Ensure lengths match (truncate to shorter)
    n = min(len(ts), len(sizes))
    return ts[:n], sizes[:n]


def extract_whisker_angle_timeseries(
    pose_obj: dict,
    kp_start: str,
    kp_end: str,
    confidence_threshold: float | None = None,
) -> np.ndarray:
    """
    Compute whisker angle (degrees) from two DLC keypoints.

    Angle = atan2(y_end − y_start, x_end − x_start), vectorised over frames.
    Frames where either keypoint falls below confidence_threshold are NaN.

    Parameters
    ----------
    pose_obj : dict
        DLC pose estimation dict.
    kp_start : str
        Base keypoint (whisker root), e.g. "C2_start".
    kp_end : str
        Tip keypoint, e.g. "C2_end".
    confidence_threshold : float or None
        Applied to BOTH start and end independently.

    Returns
    -------
    np.ndarray
        Shape (N_frames,). Degrees in [−180, 180].
    """
    x_s, y_s, _ = extract_keypoint_timeseries(pose_obj, kp_start, confidence_threshold)
    x_e, y_e, _ = extract_keypoint_timeseries(pose_obj, kp_end,   confidence_threshold)

    angle = np.degrees(np.arctan2(y_e - y_s, x_e - x_s))
    # NaN wherever either endpoint was masked (low confidence)
    angle[np.isnan(x_s) | np.isnan(x_e)] = np.nan
    return angle


# ═══════════════════════════════════════════════════════════════════════════════
# Trial-aligned slicing
# ═══════════════════════════════════════════════════════════════════════════════

def slice_timeseries_to_trial(
    timestamps: np.ndarray,
    values: np.ndarray,
    event_time: float,
    pre_sec: float,
    post_sec: float,
    resample_hz: float = 150.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract a window around event_time and return on a uniform time grid.

    NaN gaps shorter than 0.1 s are linearly interpolated; longer gaps remain NaN.

    Parameters
    ----------
    timestamps : np.ndarray
        Shape (N,). Monotonically increasing. Same time base as event_time.
    values : np.ndarray
        Shape (N,). May contain NaN (low-confidence frames).
    event_time : float
        Anchor (e.g. Audio onset in bpod seconds).
    pre_sec : float
        Start of window relative to event (typically negative, e.g. -0.5).
    post_sec : float
        End of window (positive, e.g. 2.0).
    resample_hz : float
        Number of samples per second in the output grid.

    Returns
    -------
    tuple of np.ndarray
        (t_relative, values_resampled)
        t_relative : shape (M,), zero at event_time
        values_resampled : shape (M,)
    """
    t_start = event_time + pre_sec
    t_end   = event_time + post_sec

    # Uniform output grid
    n_out  = int(round((post_sec - pre_sec) * resample_hz)) + 1
    t_rel  = np.linspace(pre_sec, post_sec, n_out)
    t_abs  = t_rel + event_time

    # Restrict input to window (with a small margin)
    margin = 0.5
    mask = (timestamps >= t_start - margin) & (timestamps <= t_end + margin)
    ts_w = timestamps[mask]
    v_w  = values[mask]

    if len(ts_w) < 2:
        return t_rel, np.full(n_out, np.nan)

    # Fill short NaN gaps by linear interpolation
    valid   = ~np.isnan(v_w)
    if valid.sum() < 2:
        return t_rel, np.full(n_out, np.nan)

    # Build interpolator on valid samples only, fill NaN where gap > 0.1 s
    ts_valid = ts_w[valid]
    v_valid  = v_w[valid]

    try:
        interp_fn = interp1d(ts_valid, v_valid, kind="linear",
                             bounds_error=False, fill_value=np.nan)
        v_resampled = interp_fn(t_abs)
    except Exception:
        v_resampled = np.full(n_out, np.nan)

    # Re-NaN long gaps: find the nearest valid original sample for each output
    # point and blank those farther than 0.1 s from any valid frame
    MAX_GAP = 0.1  # seconds
    for j, ta in enumerate(t_abs):
        if not np.isnan(v_resampled[j]):
            nearest_dist = np.min(np.abs(ts_valid - ta))
            if nearest_dist > MAX_GAP:
                v_resampled[j] = np.nan

    return t_rel, v_resampled


def baseline_subtract(
    trial_traces: np.ndarray,
    t_relative: np.ndarray,
    baseline_window: tuple[float, float] = (-0.5, 0.0),
) -> np.ndarray:
    """
    Subtract per-trial baseline mean from each trial trace.

    Parameters
    ----------
    trial_traces : np.ndarray
        Shape (n_trials, n_timepoints).
    t_relative : np.ndarray
        Shape (n_timepoints,). Time axis (zero at event).
    baseline_window : tuple[float, float]
        (start, end) in seconds. Baseline = nanmean of samples in this range.

    Returns
    -------
    np.ndarray
        Shape (n_trials, n_timepoints). Per-trial baseline subtracted.
    """
    bl_mask = (t_relative >= baseline_window[0]) & (t_relative <= baseline_window[1])
    if not bl_mask.any():
        return trial_traces.copy()

    baseline = np.nanmean(trial_traces[:, bl_mask], axis=1, keepdims=True)
    return trial_traces - baseline


def compute_trial_speed(
    x: np.ndarray,
    y: np.ndarray,
    timestamps: np.ndarray,
) -> np.ndarray:
    """
    Compute instantaneous 2D speed (pixels/second) from x, y position arrays.

    Uses central differences; edge frames use forward/backward differences.
    NaN in adjacent frames → NaN speed.

    Parameters
    ----------
    x, y : np.ndarray
        Shape (N,). Pixel coordinates. NaN for occluded frames.
    timestamps : np.ndarray
        Shape (N,). Seconds.

    Returns
    -------
    np.ndarray
        Shape (N,). Speed in pixels/second.
    """
    n = len(x)
    speed = np.full(n, np.nan)

    dt = np.diff(timestamps)
    dx = np.diff(x)
    dy = np.diff(y)

    # Central differences for interior points
    for i in range(1, n - 1):
        if np.isnan(x[i-1]) or np.isnan(x[i+1]):
            continue
        dt2 = timestamps[i+1] - timestamps[i-1]
        if dt2 <= 0:
            continue
        dx2 = x[i+1] - x[i-1]
        dy2 = y[i+1] - y[i-1]
        speed[i] = np.sqrt(dx2**2 + dy2**2) / dt2

    # Edges
    if n >= 2:
        if not (np.isnan(x[0]) or np.isnan(x[1])) and dt[0] > 0:
            speed[0] = np.sqrt(dx[0]**2 + dy[0]**2) / dt[0]
        if not (np.isnan(x[-2]) or np.isnan(x[-1])) and dt[-1] > 0:
            speed[-1] = np.sqrt(dx[-1]**2 + dy[-1]**2) / dt[-1]

    return speed


# ═══════════════════════════════════════════════════════════════════════════════
# PCA / UMAP helpers
# ═══════════════════════════════════════════════════════════════════════════════

def build_keypoint_feature_matrix(
    session_dict: dict,
    source: str,
    video_idx: int,
    keypoints: list[str],
    confidence_threshold: float | None = None,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Assemble a (frames × features) matrix from multiple keypoints.

    Features are interleaved: [x_kp0, y_kp0, x_kp1, y_kp1, ...].
    Rows containing any NaN (from low-confidence frames) are excluded from
    the returned matrix; a boolean valid_mask tracks which original rows survived.

    Parameters
    ----------
    session_dict : dict
    source : str
    video_idx : int
    keypoints : list[str]
        Keypoint names to include.
    confidence_threshold : float or None
        Passed to extract_keypoint_timeseries.
    stride : int
        Subsample every N-th frame (default 1 = all frames).
        stride=10 reduces 150fps to 15fps — useful for UMAP.

    Returns
    -------
    tuple
        (feature_matrix, valid_mask)
        feature_matrix : shape (n_valid_frames, 2*n_keypoints)
        valid_mask     : shape (n_frames_after_stride,) boolean
    """
    pose_obj = get_pose_object(session_dict, source, video_idx)
    if pose_obj is None:
        return np.empty((0, 2*len(keypoints))), np.array([], dtype=bool)

    n_total_frames = len(pose_obj["data"][0])
    frame_indices  = np.arange(0, n_total_frames, stride)
    n_strided      = len(frame_indices)

    cols = []
    for kp in keypoints:
        x, y, _ = extract_keypoint_timeseries(pose_obj, kp, confidence_threshold)
        cols.append(x[frame_indices])
        cols.append(y[frame_indices])

    if not cols:
        return np.empty((0, 0)), np.zeros(n_strided, dtype=bool)

    mat = np.column_stack(cols)                  # (n_strided, 2*n_keypoints)
    valid_mask = ~np.any(np.isnan(mat), axis=1)  # rows with all valid features

    return mat[valid_mask], valid_mask


def fit_pca_on_session(
    feature_matrix: np.ndarray,
    n_components: int = 10,
):
    """
    Fit PCA on a feature matrix and return the model and projections.

    Parameters
    ----------
    feature_matrix : np.ndarray
        Shape (n_frames, n_features). Must have no NaN.
    n_components : int
        Number of principal components to keep.

    Returns
    -------
    tuple
        (pca_model, projections)
        projections : shape (n_frames, n_components)
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_matrix)

    n_comp = min(n_components, feature_matrix.shape[0], feature_matrix.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    projections = pca.fit_transform(X_scaled)

    # Attach scaler to pca for later use in project_trial_onto_pca
    pca._scaler = scaler

    return pca, projections


def project_trial_onto_pca(pca, trial_feature_matrix: np.ndarray) -> np.ndarray:
    """
    Project a trial's feature matrix onto a pre-fitted PCA model.

    Rows with any NaN are projected as a row of NaN.

    Parameters
    ----------
    pca : sklearn PCA (with ._scaler attached by fit_pca_on_session)
    trial_feature_matrix : np.ndarray
        Shape (n_timepoints, n_features).

    Returns
    -------
    np.ndarray
        Shape (n_timepoints, n_components).
    """
    n = trial_feature_matrix.shape[0]
    out = np.full((n, pca.n_components_), np.nan)
    valid = ~np.any(np.isnan(trial_feature_matrix), axis=1)
    if valid.any():
        X_scaled = pca._scaler.transform(trial_feature_matrix[valid])
        out[valid] = pca.transform(X_scaled)
    return out


def run_umap_on_features(
    feature_matrix: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_components: int = 2,
    random_state: int = 42,
) -> np.ndarray:
    """
    Run UMAP on a (no-NaN) feature matrix.

    Call this only on a pre-subsampled matrix to keep memory manageable.
    Typical usage: stride=10 across all training sessions → ~few thousand rows.

    Parameters
    ----------
    feature_matrix : np.ndarray
        Shape (n_frames, n_features). No NaN allowed.
    n_neighbors, min_dist, n_components, random_state : UMAP hyperparameters.

    Returns
    -------
    np.ndarray
        Shape (n_frames, n_components). UMAP embedding.

    Raises
    ------
    ImportError if umap-learn is not installed.
    """
    try:
        import umap
    except ImportError:
        raise ImportError("umap-learn is required: pip install umap-learn")

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state,
    )
    return reducer.fit_transform(feature_matrix)


# ═══════════════════════════════════════════════════════════════════════════════
# Synchronisation QC
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_threshold_crossings(
    timestamps: np.ndarray,
    values: np.ndarray,
    threshold_px: float = 5.0,
    min_gap_sec: float = 0.05,
) -> np.ndarray:
    """
    Detect rising-edge threshold crossings in a scalar timeseries.

    Uses a simple deviation from a sliding median (window ≈ 1 s at 150 fps).

    Parameters
    ----------
    timestamps : np.ndarray  shape (N,)
    values : np.ndarray       shape (N,) — may have NaN
    threshold_px : float      deviation above median to count as a crossing
    min_gap_sec : float       minimum time between consecutive crossings

    Returns
    -------
    np.ndarray
        Times of rising-edge crossings (seconds).
    """
    if len(values) < 10:
        return np.array([])

    # Rolling median baseline (window = 150 frames ≈ 1 s)
    win = min(150, len(values) // 4)
    baseline = pd.Series(values).rolling(win, center=True,
                                         min_periods=1).median().values
    deviation = np.abs(values - baseline)

    above = deviation > threshold_px
    # Rising edges
    edges = np.where(np.diff(above.astype(int)) == 1)[0]
    if len(edges) == 0:
        return np.array([])

    edge_times = timestamps[edges]

    # Debounce: keep only the first of any pair within min_gap_sec
    kept = [edge_times[0]]
    for t in edge_times[1:]:
        if t - kept[-1] >= min_gap_sec:
            kept.append(t)

    return np.array(kept)


def compute_sync_alignment_error(
    session_dict: dict,
    dlc_video_idx: int = 0,
    dlc_confidence_threshold: float = 0.7,
    transition_threshold_px: float = 5.0,
) -> dict:
    """
    Sanity-check the sync quality of an already-synchronised session.

    Background
    ----------
    The w2t-bkin pipeline synchronises bpod and video timestamps using the
    DLC ``trial_light`` keypoint: this LED is controlled by bpod and turns on
    exactly when the Audio state starts. The pipeline uses the rising edges of
    ``trial_light`` to align the two clocks, so the processed pickle files
    already share a common time base.

    This function verifies the quality of that alignment post-hoc by comparing
    the ``trial_light`` rising-edge times (video side) against the Audio state
    onset times stored in trials_table (bpod side).  After synchronisation the
    residuals should be within one camera frame (~7 ms at 150 fps).

    Strategy
    --------
    1. Detect ``trial_light`` rising edges in the DLC y-position trace.
    2. Read Audio onset times from trials_table (one per trial).
    3. Pair each edge with the nearest Audio onset (within ±200 ms).
    4. Compute residual distribution.  Flag sessions where residuals are
       unexpectedly large (> SYNC_FLAG_MEAN_OFFSET_SEC).

    Parameters
    ----------
    session_dict : dict
    dlc_video_idx : int
        Which DLC camera to use (0 or 1).
    dlc_confidence_threshold : float
        Frames below this confidence are excluded from edge detection.
    transition_threshold_px : float
        Pixel displacement threshold for edge detection.

    Returns
    -------
    dict with keys
        session, mouse,
        n_trial_light_edges, n_audio_onsets,
        n_matched_pairs, match_rate,
        residuals_sec (np.ndarray), mean_residual_sec, std_residual_sec,
        max_abs_residual_sec, flag_misaligned (bool)
    """
    mouse   = session_dict.get("subject", "unknown")
    session = session_dict.get("session", "unknown")

    # ── DLC trial_light rising edges ──────────────────────────────────────────
    # Detection strategy: trial_light is an LED controlled by bpod. When ON,
    # DLC detects it with high confidence; when OFF, confidence is low → NaN.
    # Rising edge = first valid (high-confidence) frame after a NaN gap.
    pose_obj = get_pose_object(session_dict, "dlc", dlc_video_idx)
    dlc_edges = np.array([])
    if pose_obj is not None:
        ts_dlc = get_pose_timestamps(pose_obj)
        _, _, conf_light = extract_keypoint_timeseries(
            pose_obj, "trial_light", confidence_threshold=None
        )
        # Boolean array: above threshold = LED on
        led_on = conf_light >= dlc_confidence_threshold
        # Rising edges: False→True transitions
        edges_idx = np.where(np.diff(led_on.astype(int)) == 1)[0] + 1
        if len(edges_idx) > 0:
            edge_times = ts_dlc[edges_idx]
            # Debounce: drop edges within 0.5 s of each other
            kept = [edge_times[0]]
            for t in edge_times[1:]:
                if t - kept[-1] >= 0.5:
                    kept.append(t)
            dlc_edges = np.array(kept)

    # ── Bpod Audio onsets from trials_table ───────────────────────────────────
    try:
        tt = extract_trial_table(session_dict)
        audio_onsets = tt["audio_onset"].dropna().values
    except Exception:
        audio_onsets = np.array([])

    # ── Match each DLC edge to the nearest Audio onset ────────────────────────
    residuals = []
    used_bpod = set()
    for t_edge in dlc_edges:
        if len(audio_onsets) == 0:
            break
        diffs = np.abs(audio_onsets - t_edge)
        idx   = int(np.argmin(diffs))
        if diffs[idx] <= SYNC_MATCH_WINDOW_SEC and idx not in used_bpod:
            residuals.append(t_edge - audio_onsets[idx])
            used_bpod.add(idx)

    residuals  = np.array(residuals)
    n_matched  = len(residuals)
    n_edges    = len(dlc_edges)
    n_audio    = len(audio_onsets)
    denom      = max(n_edges, n_audio, 1)
    match_rate = n_matched / denom

    mean_res = float(np.mean(residuals))        if n_matched > 0 else np.nan
    std_res  = float(np.std(residuals))         if n_matched > 0 else np.nan
    max_res  = float(np.max(np.abs(residuals))) if n_matched > 0 else np.nan

    flag = (
        n_edges < 5
        or match_rate < SYNC_FLAG_MATCH_RATE
        or (not np.isnan(mean_res) and abs(mean_res) > SYNC_FLAG_MEAN_OFFSET_SEC)
        or (not np.isnan(std_res)  and std_res        > SYNC_FLAG_STD_SEC)
    )

    return {
        "session":              session,
        "mouse":                mouse,
        "n_trial_light_edges":  n_edges,
        "n_audio_onsets":       n_audio,
        "n_matched_pairs":      n_matched,
        "match_rate":           match_rate,
        "residuals_sec":        residuals,
        "mean_residual_sec":    mean_res,
        "std_residual_sec":     std_res,
        "max_abs_residual_sec": max_res,
        "flag_misaligned":      flag,
    }

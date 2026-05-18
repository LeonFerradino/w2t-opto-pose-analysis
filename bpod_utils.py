# -*- coding: utf-8 -*-
"""
Shared utilities for all Bpod analysis scripts.

Import what you need:
    from bpod_utils import load_mat, is_hit, latency, extract_sort_key, \
                           extract_day_from_filename, extract_day_from_path, \
                           safe_getattr, sem, BASE_PATH, MIN_TRIALS
"""

import os
import re
import scipy.io
import numpy as np
from scipy.io.matlab.mio5_params import mat_struct

# ── Global constants ──────────────────────────────────────────────────────────
BASE_PATH  = 'F:/bpod/'
MIN_TRIALS = 40


def build_session_path(base, mouse, phase, task):
    """
    Return path to Session Data folder, handling both folder structures:
      nested: {base}/{mouse}/{phase}/{task}/Session Data   (original mice)
      flat:   {base}/{mouse}/{task}/Session Data           (new cohort)
    """
    nested = os.path.join(base, mouse, phase, task, 'Session Data')
    flat   = os.path.join(base, mouse, task,        'Session Data')
    return nested if os.path.exists(nested) else flat


# ── .mat loader ───────────────────────────────────────────────────────────────
def load_mat(filename):
    """Load a MATLAB .mat file and recursively convert all structs to dicts."""
    def _check_vars(d):
        for key in d:
            if isinstance(d[key], mat_struct):
                d[key] = _todict(d[key])
            elif isinstance(d[key], np.ndarray):
                d[key] = _toarray(d[key])
        return d

    def _todict(matobj):
        d = {}
        for strg in matobj._fieldnames:
            elem = getattr(matobj, strg)
            if isinstance(elem, mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _toarray(elem)
            else:
                d[strg] = elem
        return d

    def _toarray(ndarray):
        if ndarray.dtype != 'float64':
            elem_list = []
            for sub_elem in ndarray:
                if isinstance(sub_elem, mat_struct):
                    elem_list.append(_todict(sub_elem))
                elif isinstance(sub_elem, np.ndarray):
                    elem_list.append(_toarray(sub_elem))
                else:
                    elem_list.append(sub_elem)
            return np.array(elem_list, dtype='object')
        else:
            return ndarray

    return _check_vars(scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True))


# ── Trial helpers ─────────────────────────────────────────────────────────────
def safe_getattr(obj, attr):
    """Get attribute from dict or object, returning None if missing."""
    if isinstance(obj, dict):
        return obj.get(attr, None)
    else:
        return getattr(obj, attr, None)


def is_hit(trial):
    """Return True if the trial is a hit (Miss state absent or all-NaN)."""
    states = safe_getattr(trial, 'States')
    if states is None:
        return False
    miss = safe_getattr(states, 'Miss')
    if miss is None:
        return True
    elif isinstance(miss, np.ndarray):
        return np.isnan(miss).all()
    else:
        return False


def latency(trial, max_latency=2.0):
    """
    Return response latency (s) for a hit trial, or NaN.

    Checks states: Cue, Audio, W2T_Audio, W2T_Audio_opto (in that order).
    Returns NaN for misses or if no valid state is found.
    """
    if not is_hit(trial):
        return np.nan
    states = safe_getattr(trial, 'States')
    if states is None:
        return np.nan
    for state in ('Cue', 'Audio', 'W2T_Audio', 'W2T_Audio_opto'):
        times = safe_getattr(states, state)
        if (times is not None
                and isinstance(times, np.ndarray)
                and len(times) > 1
                and not np.isnan(times[:2]).any()):
            lat = times[1] - times[0]
            if 0 < lat <= max_latency:
                return lat
    return np.nan


# ── Filename helpers ──────────────────────────────────────────────────────────
def extract_sort_key(filename):
    """Return YYYYMMDDHHMMSS string for chronological sorting, or filename."""
    match = re.search(r'_(\d{8})_(\d{6})', filename)
    if match:
        return match.group(1) + match.group(2)
    return filename


def extract_day_from_filename(filename):
    """Return YYYYMMDD from a bare filename, or None."""
    match = re.search(r'_(\d{8})_', filename)
    return match.group(1) if match else None


def extract_day_from_path(filepath):
    """Return YYYYMMDD from a full filepath, or None."""
    return extract_day_from_filename(os.path.basename(filepath))


# ── Statistics ────────────────────────────────────────────────────────────────
def sem(data):
    """Standard error of the mean across axis=0, ignoring NaNs."""
    return np.nanstd(data, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(data), axis=0))


def fdr_bh(pvals):
    """Benjamini-Hochberg FDR correction. Returns array of adjusted p-values."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    if n == 0:
        return pvals
    order = np.argsort(pvals)
    pvals_sorted = pvals[order]
    adjusted = np.minimum(1.0, pvals_sorted * n / (np.arange(1, n + 1)))
    # enforce monotonicity from right
    for i in range(n - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])
    result = np.empty(n)
    result[order] = adjusted
    return result


def rank_biserial_r(a, b):
    """Rank-biserial correlation r for Wilcoxon signed-rank test (effect size).
    r = 1 - 2W / (n*(n+1)/2), where W is the sum of positive ranks."""
    from scipy.stats import wilcoxon
    diff = np.asarray(b) - np.asarray(a)
    diff = diff[diff != 0]
    n = len(diff)
    if n < 1:
        return np.nan
    stat, _ = wilcoxon(a, b)
    max_w = n * (n + 1) / 2
    return 1 - 2 * stat / max_w if max_w > 0 else np.nan


def bootstrap_ci(values, n_boot=5000, ci=95, seed=42):
    """Bootstrap percentile CI for the mean. Returns (lower, upper)."""
    rng = np.random.default_rng(seed)
    values = np.asarray([v for v in values if not np.isnan(v)])
    if len(values) == 0:
        return np.nan, np.nan
    boots = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    lo = np.percentile(boots, (100 - ci) / 2)
    hi = np.percentile(boots, 100 - (100 - ci) / 2)
    return lo, hi


def is_blocked_session(filepath):
    """Return True if the session comes from a _blocked task folder."""
    return '_blocked' in os.path.basename(filepath).lower()


def flag_outlier_sessions(values, mouse_id='', threshold_sd=2.5):
    """Flag session values that are >threshold_sd SDs from the mouse mean.
    Returns boolean array (True = outlier)."""
    values = np.asarray(values, dtype=float)
    valid = values[~np.isnan(values)]
    if len(valid) < 3:
        return np.zeros(len(values), dtype=bool)
    mu, sd = np.nanmean(valid), np.nanstd(valid, ddof=1)
    flags = np.abs(values - mu) > threshold_sd * sd
    if flags.any() and mouse_id:
        for i, f in enumerate(flags):
            if f:
                print(f"  [QC] {mouse_id} session {i}: value={values[i]:.2f} "
                      f"(mean={mu:.2f}, {abs(values[i]-mu)/sd:.1f} SD) — flagged")
    return flags

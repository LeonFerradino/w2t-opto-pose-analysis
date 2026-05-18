# -*- coding: utf-8 -*-
"""
pose_opto_summary_v3.py — Opto 4x4 heatmap with angle-wrap fix.

Identical to v2 (MLA/SNA separation, consistency indicators, artifact flags)
with one methodological fix:

Fix vs v2
---------
- Circular baseline subtraction for angle keypoints (C2_angle, C1r_angle):
  deviation = ((trace - circular_bl_mean + 180) % 360) - 180
  Removes +-350 deg jump artefacts from atan2 wrap-around crossings without
  the cumulative drift that np.unwrap introduces for oscillating whiskers.
- All other keypoints (nose_y, fpaw_y) use linear baseline_subtract as before.

STANDALONE — no modification to pose_utils.py, pose_opto_kinematics.py, pose_config.py.
"""

import gc
import sys
import warnings
import zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from pathlib import Path
from datetime import datetime
from scipy.stats import mannwhitneyu, chi2

sys.path.insert(0, str(Path(__file__).parent))

from pose_config import (
    BRAIN_AREAS_ORDERED, EXCLUDE_AREA,
    PUB_KEYPOINTS, PUB_KEYPOINT_LABELS,
    HIT_WINDOW_SEC, HIT_BASELINE_SEC, RESAMPLE_HZ,
    CONFIDENCE_THRESHOLDS,
    build_session_registry, find_bpod_mat_files,
    get_brain_area,
)
from pose_utils import (
    load_session_pickle, extract_trial_table, load_trial_types_from_mat,
    get_pose_object, get_pose_timestamps,
    extract_keypoint_timeseries,
    extract_whisker_angle_timeseries,
    slice_timeseries_to_trial, baseline_subtract,
)
from bpod_utils import fdr_bh


# ── Output directory ──────────────────────────────────────────────────────────
OUTDIR = Path("f:/analysis/bpod/Documents/output_opto_summary_v3")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ── Mouse groups ──────────────────────────────────────────────────────────────
MLA_MICE  = ["MLA-026805_2", "MLA-026806_3", "MLA-026807_4"]
SNA_MOUSE = "SNA-145894_1"

# ── Confidence threshold override ─────────────────────────────────────────────
# DLC lowered 0.9 → 0.8 (Jelte's default; retains more frames before interpolation).
# SLEAP unchanged at 0.6.
CONF_THR_OVERRIDE: dict = {
    "dlc":     0.8,
    "sleap":   0.6,
    "facemap": None,
}

# ── Additional exclusions not yet in pose_config.EXCLUDED_SESSIONS ────────────
EXTRA_EXCLUSIONS: set = {
    # SNA-145894_1 20250916_2: opto_random with sync collapse.
    ("SNA-145894_1", "20250916", "2"),
    # MLA-026807_4 20250922_2: 128 aborted trials, only ~8 W2T HITs.
    ("MLA-026807_4", "20250922", "2"),
}

# ── Analysis parameters ───────────────────────────────────────────────────────
POST_TOUCH_WINDOW     = (0.0, 1.0)   # (s) scalar window post-touch
MIN_TRIALS_PER_COND   = 5            # skip if either condition below this
MIN_MLA_MICE_FOR_CELL = 2            # NaN cell if fewer MLA mice contribute


# ════════════════════════════════════════════════════════════════════════════════
# Statistics helpers
# ════════════════════════════════════════════════════════════════════════════════

def mann_whitney_r(w2t_vals: np.ndarray, opto_vals: np.ndarray):
    """
    Unpaired rank-biserial r from Mann-Whitney U.
    r = 1 − 2U/(n1·n2).  r > 0 when Opto > W2T.
    Returns (np.nan, np.nan) when n < 3 in either group.
    """
    n1, n2 = len(w2t_vals), len(opto_vals)
    if n1 < 3 or n2 < 3:
        return np.nan, np.nan
    try:
        stat, p = mannwhitneyu(w2t_vals, opto_vals, alternative="two-sided")
        r = 1.0 - 2.0 * float(stat) / float(n1 * n2)
        return float(r), float(p)
    except Exception:
        return np.nan, np.nan


def fisher_combined_p(pvals):
    """Fisher's method: χ² = −2 Σ ln(pᵢ), df = 2k. Clips p to 1e-300."""
    valid = [max(float(p), 1e-300) for p in pvals
             if not np.isnan(p) and 0.0 < p <= 1.0]
    if not valid:
        return np.nan
    return float(chi2.sf(-2.0 * np.sum(np.log(valid)), df=2 * len(valid)))


def sig_stars(p: float) -> str:
    if np.isnan(p): return ""
    if p < 0.001:   return "***"
    if p < 0.01:    return "**"
    if p < 0.05:    return "*"
    return ""


def is_artifact_flag(sna_r: float, mla_r: float) -> bool:
    """
    True when SNA mirrors MLA effect:
      |SNA| ≥ 0.10  AND  same sign as MLA  AND  |SNA| ≥ 0.5 × |MLA|
    """
    if np.isnan(sna_r) or np.isnan(mla_r) or mla_r == 0.0:
        return False
    return (
        abs(sna_r) >= 0.10
        and np.sign(sna_r) == np.sign(mla_r)
        and abs(sna_r) >= 0.5 * abs(mla_r)
    )


def classify_cell(mla_r: float, mla_p_fdr: float, sna_r: float) -> str:
    """
    Classify a cell based on MLA significance and SNA control comparison.

    Returns one of:
      '🟢 Robust real'           MLA sig, SNA opposite or negligible (< 0.10)
      '🟡 Likely real'           MLA sig, SNA same direction but < 50% of MLA
      '🔴 Artifact-suspect'      MLA sig, SNA mirrors MLA (artifact_flag)
      '◻ Not significant'        MLA p_fdr ≥ 0.05
      '◻ No data'                MLA NaN
    """
    if np.isnan(mla_r) or np.isnan(mla_p_fdr):
        return "◻ No data"
    if mla_p_fdr >= 0.05:
        return "◻ Not significant"
    if is_artifact_flag(sna_r, mla_r):
        return "🔴 Artifact-suspect"
    # SNA negligible or opposite direction
    if np.isnan(sna_r) or abs(sna_r) < 0.10 or np.sign(sna_r) != np.sign(mla_r):
        return "🟢 Robust real"
    # SNA same direction but small (< 50% of MLA)
    return "🟡 Likely real"


# ════════════════════════════════════════════════════════════════════════════════
# Keypoint extraction  (refactored: extract timeseries once, slice twice)
# ════════════════════════════════════════════════════════════════════════════════

def get_keypoint_values(session_dict: dict, kp_name: str, spec: dict):
    """
    Extract the full keypoint timeseries from session pickle (using CONF_THR_OVERRIDE).

    Returns
    -------
    ts_pose : np.ndarray or None    — pose timestamps (session length)
    values  : np.ndarray or None    — signal values (NaN = low-confidence)
    raw_nan_rate : float            — fraction of NaN frames across whole session
    """
    source   = spec["source"]
    vid      = spec["vid"]
    coord    = spec["coord"]
    conf_thr = CONF_THR_OVERRIDE.get(source, CONFIDENCE_THRESHOLDS.get(source))

    pose_obj = get_pose_object(session_dict, source, vid)
    if pose_obj is None:
        return None, None, np.nan

    ts_pose = get_pose_timestamps(pose_obj)

    if coord == "angle":
        values = extract_whisker_angle_timeseries(
            pose_obj, spec["kp_start"], spec["kp_end"], conf_thr)
    else:
        x, y, _ = extract_keypoint_timeseries(pose_obj, spec["kp"], conf_thr)
        values = y if coord == "y" else x

    raw_nan_rate = float(np.isnan(values).mean()) if len(values) > 0 else np.nan
    return ts_pose, values, raw_nan_rate


def circular_baseline_subtract_deg(
    traces_2d: np.ndarray,
    t_rel: np.ndarray,
    baseline_sec: tuple,
) -> np.ndarray:
    """
    Circular baseline subtraction for angle traces (degrees).
    Computes circular mean of baseline window; subtracts modularly
    so result is in [-180, +180] (minimum-arc deviation from baseline).
    Fixes +-350 deg jumps from atan2 wrap-arounds without drift.
    """
    bl_mask = (t_rel >= baseline_sec[0]) & (t_rel <= baseline_sec[1])
    result  = traces_2d.copy()
    for i in range(len(result)):
        row    = result[i]
        bl_pts = row[bl_mask]
        valid  = bl_pts[~np.isnan(bl_pts)]
        if len(valid) < 2:
            continue
        bl_mean = np.rad2deg(
            np.arctan2(np.mean(np.sin(np.deg2rad(valid))),
                       np.mean(np.cos(np.deg2rad(valid))))
        )
        dev       = row - bl_mean
        dev       = ((dev + 180.0) % 360.0) - 180.0
        result[i] = dev
    return result


def compute_trial_scalars(
    ts_pose: np.ndarray,
    values: np.ndarray,
    trial_rows: pd.DataFrame,
    t_rel: np.ndarray,
    is_angle: bool = False,
) -> np.ndarray:
    """
    Given a pre-extracted timeseries, compute per-trial scalars.

    Slices each HIT trial into HIT_WINDOW_SEC, baseline-subtracts
    (circular for angle keypoints, linear for position keypoints),
    then returns mean signal in POST_TOUCH_WINDOW.

    Returns np.ndarray shape (n_trials,). NaN for failed extractions.
    """
    n_trial = len(trial_rows)
    n_t     = len(t_rel)
    traces  = np.full((n_trial, n_t), np.nan)

    for i, (_, tr_row) in enumerate(trial_rows.iterrows()):
        anchor = tr_row.get("hit_onset", np.nan)
        try:
            anchor = float(anchor)
        except (TypeError, ValueError):
            anchor = np.nan
        if np.isnan(anchor):
            continue
        _, trace = slice_timeseries_to_trial(
            ts_pose, values, anchor,
            HIT_WINDOW_SEC[0], HIT_WINDOW_SEC[1], RESAMPLE_HZ,
        )
        traces[i] = trace

    if is_angle:
        traces = circular_baseline_subtract_deg(traces, t_rel, HIT_BASELINE_SEC)
    else:
        traces = baseline_subtract(traces, t_rel, HIT_BASELINE_SEC)

    post_mask = (t_rel >= POST_TOUCH_WINDOW[0]) & (t_rel <= POST_TOUCH_WINDOW[1])
    if not post_mask.any():
        return np.full(n_trial, np.nan)
    return np.nanmean(traces[:, post_mask], axis=1)


# ════════════════════════════════════════════════════════════════════════════════
# Main analysis
# ════════════════════════════════════════════════════════════════════════════════

def run_analysis_v2():  # name kept for import compatibility
    """End-to-end: load → compute → plot double heatmap → save ZIP."""

    print("\n" + "=" * 65)
    print("OPTO SUMMARY v2 -- MLA (n<=3) + SNA wildtype control (n=1)")
    print("=" * 65)
    start_ts = datetime.now()
    print(f"Started: {start_ts.strftime('%Y-%m-%d %H:%M:%S')}")

    # ── Session registry ──────────────────────────────────────────────────────
    all_sessions = build_session_registry(exist_only=True)
    opto_sessions = [
        s for s in all_sessions
        if s["phase"] in ("opto_blocked", "opto_random")
        and (s["mouse"], s["date"], s["suffix"]) not in EXTRA_EXCLUSIONS
    ]
    print(f"\n{len(opto_sessions)} opto sessions after exclusions")

    t_rel = np.linspace(
        HIT_WINDOW_SEC[0], HIT_WINDOW_SEC[1],
        int(round((HIT_WINDOW_SEC[1] - HIT_WINDOW_SEC[0]) * RESAMPLE_HZ)) + 1,
    )
    keypoints = list(PUB_KEYPOINTS.keys())
    n_kp   = len(keypoints)
    n_area = len(BRAIN_AREAS_ORDERED)

    # ── Data accumulators ─────────────────────────────────────────────────────
    per_mla_data = {
        mouse: {
            kp: {area: {"w2t": [], "opto": []} for area in BRAIN_AREAS_ORDERED}
            for kp in keypoints
        }
        for mouse in MLA_MICE
    }
    per_sna_data = {
        kp: {area: {"w2t": [], "opto": []} for area in BRAIN_AREAS_ORDERED}
        for kp in keypoints
    }
    # NaN rate per (kp, area) across all contributing sessions
    nan_rate_accum = {
        kp: {area: [] for area in BRAIN_AREAS_ORDERED} for kp in keypoints
    }

    used_sessions    = []
    skipped_sessions = []

    # ── Session loop ──────────────────────────────────────────────────────────
    for rec in sorted(opto_sessions,
                      key=lambda s: (s["mouse"], s["date"], s["suffix"])):
        mouse  = rec["mouse"]
        date   = rec["date"]
        suffix = rec["suffix"]
        area   = get_brain_area(mouse, date)

        if area is None or area == EXCLUDE_AREA or area not in BRAIN_AREAS_ORDERED:
            skipped_sessions.append({
                "mouse": mouse, "date": date, "suffix": suffix,
                "reason": f"area={area!r} (excluded or unknown)",
            })
            continue

        dur = rec.get("opto_duration_s")
        if dur is None:
            skipped_sessions.append({
                "mouse": mouse, "date": date, "suffix": suffix,
                "reason": "opto_duration_s=None (protocol not in session_config)",
            })
            continue

        group = "MLA" if mouse in MLA_MICE else "SNA"
        tag = f"  [{group}] {mouse} {date}_{suffix} ({area}, {dur}s)"
        print(f"{tag} ...", end=" ", flush=True)

        d = None
        try:
            d = load_session_pickle(rec["pickle_path"])
            mat_files       = find_bpod_mat_files(mouse, date, suffix)
            trial_types_arr = load_trial_types_from_mat(mat_files) if mat_files else None

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tt = extract_trial_table(d, trial_types_array=trial_types_arr)

            w2t_mask  = ((tt["trial_type"] == "W2T") & (tt["outcome"] == "HIT")
                         & tt["hit_onset"].notna())
            opto_mask = ((tt["trial_type"] == "Opto") & (tt["outcome"] == "HIT")
                         & tt["hit_onset"].notna())
            n_w2t  = int(w2t_mask.sum())
            n_opto = int(opto_mask.sum())
            print(f"W2T={n_w2t} Opto={n_opto}", end=" ", flush=True)

            if n_w2t < MIN_TRIALS_PER_COND or n_opto < MIN_TRIALS_PER_COND:
                skipped_sessions.append({
                    "mouse": mouse, "date": date, "suffix": suffix,
                    "reason": (f"<{MIN_TRIALS_PER_COND} trials "
                               f"(W2T={n_w2t}, Opto={n_opto})"),
                })
                print("SKIP")
                del d; gc.collect()
                continue

            w2t_rows  = tt[w2t_mask].reset_index(drop=True)
            opto_rows = tt[opto_mask].reset_index(drop=True)

            session_ok = False
            for kp_name, spec in PUB_KEYPOINTS.items():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ts_pose, values, nan_rate = get_keypoint_values(d, kp_name, spec)

                if ts_pose is None:
                    continue

                # Accumulate NaN rate for interpolation audit
                if not np.isnan(nan_rate):
                    nan_rate_accum[kp_name][area].append(nan_rate)

                is_ang = (PUB_KEYPOINTS[kp_name].get("coord") == "angle")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    w2t_scalars  = compute_trial_scalars(ts_pose, values, w2t_rows,  t_rel, is_angle=is_ang)
                    opto_scalars = compute_trial_scalars(ts_pose, values, opto_rows, t_rel, is_angle=is_ang)

                w2t_valid  = w2t_scalars[~np.isnan(w2t_scalars)]
                opto_valid = opto_scalars[~np.isnan(opto_scalars)]

                if (len(w2t_valid) >= MIN_TRIALS_PER_COND
                        and len(opto_valid) >= MIN_TRIALS_PER_COND):
                    if mouse in MLA_MICE:
                        per_mla_data[mouse][kp_name][area]["w2t"].extend(w2t_valid.tolist())
                        per_mla_data[mouse][kp_name][area]["opto"].extend(opto_valid.tolist())
                    else:  # SNA
                        per_sna_data[kp_name][area]["w2t"].extend(w2t_valid.tolist())
                        per_sna_data[kp_name][area]["opto"].extend(opto_valid.tolist())
                    session_ok = True

            del d; gc.collect()

            if session_ok:
                used_sessions.append({
                    "mouse": mouse, "date": date, "suffix": suffix,
                    "area": area, "group": group,
                    "opto_duration": dur, "phase": rec["phase"],
                    "n_w2t": n_w2t, "n_opto": n_opto,
                })
                print("ok")
            else:
                skipped_sessions.append({
                    "mouse": mouse, "date": date, "suffix": suffix,
                    "reason": "no keypoint yielded ≥5 valid scalars per condition",
                })
                print("SKIP (no valid scalars)")

        except Exception as exc:
            skipped_sessions.append({
                "mouse": mouse, "date": date, "suffix": suffix,
                "reason": f"Exception: {exc}",
            })
            print(f"ERROR: {exc}")
            if d is not None:
                try: del d; gc.collect()
                except Exception: pass

    print(f"\n{len(used_sessions)} sessions used, {len(skipped_sessions)} skipped")

    # ── MLA cell statistics ───────────────────────────────────────────────────
    # Arrays: [n_kp, n_area]
    mla_r      = np.full((n_kp, n_area), np.nan)
    mla_p      = np.full((n_kp, n_area), np.nan)   # Fisher combined (pre-FDR)
    mla_p_fdr  = np.full((n_kp, n_area), np.nan)
    mla_n      = np.zeros((n_kp, n_area), dtype=int)
    # mla_consist: [count_same_sign, n_contributing] per cell
    mla_consist = np.zeros((n_kp, n_area, 2), dtype=int)
    # Per-mouse r for consistency reporting
    mla_per_mouse_r = {
        (ki, ai): {} for ki in range(n_kp) for ai in range(n_area)
    }

    for ki, kp_name in enumerate(keypoints):
        for ai, area in enumerate(BRAIN_AREAS_ORDERED):
            mouse_rs, mouse_ps = [], []
            for mouse in MLA_MICE:
                w2t_pool  = np.array(per_mla_data[mouse][kp_name][area]["w2t"])
                opto_pool = np.array(per_mla_data[mouse][kp_name][area]["opto"])
                if (len(w2t_pool) < MIN_TRIALS_PER_COND
                        or len(opto_pool) < MIN_TRIALS_PER_COND):
                    continue
                r, p = mann_whitney_r(w2t_pool, opto_pool)
                if not np.isnan(r):
                    mouse_rs.append(r)
                    mouse_ps.append(p)
                    mla_per_mouse_r[(ki, ai)][mouse] = r

            n_mice = len(mouse_rs)
            mla_n[ki, ai] = n_mice
            if n_mice >= MIN_MLA_MICE_FOR_CELL:
                cell_mean = float(np.mean(mouse_rs))
                mla_r[ki, ai] = cell_mean
                mla_p[ki, ai] = fisher_combined_p(mouse_ps)
                count_same = sum(1 for r in mouse_rs
                                 if np.sign(r) == np.sign(cell_mean))
                mla_consist[ki, ai] = [count_same, n_mice]

    # FDR over all valid MLA cells
    valid_mla = ~np.isnan(mla_p)
    if valid_mla.sum() > 0:
        mla_p_fdr[valid_mla] = fdr_bh(mla_p[valid_mla])

    # ── SNA cell statistics ───────────────────────────────────────────────────
    sna_r     = np.full((n_kp, n_area), np.nan)
    sna_p     = np.full((n_kp, n_area), np.nan)   # raw Mann-Whitney
    sna_p_fdr = np.full((n_kp, n_area), np.nan)   # FDR-corrected (indicative)

    for ki, kp_name in enumerate(keypoints):
        for ai, area in enumerate(BRAIN_AREAS_ORDERED):
            w2t_pool  = np.array(per_sna_data[kp_name][area]["w2t"])
            opto_pool = np.array(per_sna_data[kp_name][area]["opto"])
            if (len(w2t_pool) >= MIN_TRIALS_PER_COND
                    and len(opto_pool) >= MIN_TRIALS_PER_COND):
                r, p = mann_whitney_r(w2t_pool, opto_pool)
                sna_r[ki, ai] = r
                sna_p[ki, ai] = p

    valid_sna = ~np.isnan(sna_p)
    if valid_sna.sum() > 0:
        sna_p_fdr[valid_sna] = fdr_bh(sna_p[valid_sna])

    # ── Artifact flags and cell classification ────────────────────────────────
    artifact_flags = np.zeros((n_kp, n_area), dtype=bool)
    cell_class     = np.full((n_kp, n_area), "", dtype=object)
    for ki in range(n_kp):
        for ai in range(n_area):
            artifact_flags[ki, ai] = is_artifact_flag(sna_r[ki, ai], mla_r[ki, ai])
            cell_class[ki, ai]     = classify_cell(
                mla_r[ki, ai], mla_p_fdr[ki, ai], sna_r[ki, ai])

    n_robust   = int(np.sum(cell_class == "🟢 Robust real"))
    n_likely   = int(np.sum(cell_class == "🟡 Likely real"))
    n_artifact = int(np.sum(artifact_flags))
    n_not_sig  = int(np.sum(cell_class == "◻ Not significant"))

    print(f"\nMLA cells: {n_robust} Robust | {n_likely} Likely | "
          f"{n_artifact} Artifact-suspect | {n_not_sig} n.s.")

    # ═══════════════════════════════════════════════════════════════════════════
    # Figure — stacked double heatmap
    # ═══════════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(10, 13))
    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        width_ratios=[20, 1],
        hspace=0.55, wspace=0.08,
    )
    ax_mla  = fig.add_subplot(gs[0, 0])
    ax_sna  = fig.add_subplot(gs[1, 0])
    ax_cbar = fig.add_subplot(gs[:, 1])

    vmin, vmax = -0.5, 0.5
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad(color="#cccccc")   # NaN cells shown as medium gray

    kp_labels = [PUB_KEYPOINT_LABELS[k] for k in keypoints]

    # ── Top panel: MLA ────────────────────────────────────────────────────────
    ax_mla.imshow(np.ma.array(mla_r, mask=np.isnan(mla_r)),
                  cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    ax_mla.set_xticks(range(n_area))
    ax_mla.set_xticklabels(BRAIN_AREAS_ORDERED, fontsize=11, fontweight="bold")
    ax_mla.set_yticks(range(n_kp))
    ax_mla.set_yticklabels(kp_labels, fontsize=10)
    ax_mla.set_ylabel("Keypoint", fontsize=11, labelpad=6)
    ax_mla.set_title(
        "Experimental — opsin-expressing mice (MLA, n ≤ 3)\n"
        "rank-biserial r; r < 0: suppression  |  r > 0: elevation\n"
        "stars in () = sign-inconsistent across mice",
        fontsize=9.5, pad=8,
    )

    for xi in range(n_area + 1):
        ax_mla.axvline(xi - 0.5, color="white", linewidth=1.2)
    for yi in range(n_kp + 1):
        ax_mla.axhline(yi - 0.5, color="white", linewidth=1.2)

    for ki in range(n_kp):
        for ai in range(n_area):
            r_val = mla_r[ki, ai]
            p_fdr = mla_p_fdr[ki, ai]
            n_m   = int(mla_n[ki, ai])
            c_same, c_total = int(mla_consist[ki, ai, 0]), int(mla_consist[ki, ai, 1])

            if np.isnan(r_val):
                ax_mla.text(ai, ki, "—\n(no data)", ha="center", va="center",
                            fontsize=8, color="gray", style="italic")
                continue

            txt_col = "white" if abs(r_val) >= 0.28 else "black"
            ss = sig_stars(p_fdr)
            stars_consistent = (c_same >= 2)
            stars_str  = ss if stars_consistent else (f"({ss})" if ss else "")
            consist_str = f"{c_same}/{c_total}" if c_total > 0 else ""
            line1 = f"{r_val:+.2f}{stars_str}"
            line2 = f"{consist_str}  n={n_m}"
            ax_mla.text(ai, ki, f"{line1}\n{line2}",
                        ha="center", va="center",
                        fontsize=9.5, color=txt_col, fontweight="bold",
                        linespacing=1.5)

    # ── Bottom panel: SNA control ─────────────────────────────────────────────
    ax_sna.imshow(np.ma.array(sna_r, mask=np.isnan(sna_r)),
                  cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    ax_sna.set_xticks(range(n_area))
    ax_sna.set_xticklabels(BRAIN_AREAS_ORDERED, fontsize=11, fontweight="bold")
    ax_sna.set_yticks(range(n_kp))
    ax_sna.set_yticklabels(kp_labels, fontsize=10)
    ax_sna.set_ylabel("Keypoint", fontsize=11, labelpad=6)
    ax_sna.set_xlabel("Inhibited brain area", fontsize=11, labelpad=6)
    ax_sna.set_title(
        "Light-only control — wildtype SNA-145894_1 (n = 1)\n"
        "FDR-corrected stars indicative only; (!) = artifact flag",
        fontsize=9.5, pad=8,
    )

    for xi in range(n_area + 1):
        ax_sna.axvline(xi - 0.5, color="white", linewidth=1.2)
    for yi in range(n_kp + 1):
        ax_sna.axhline(yi - 0.5, color="white", linewidth=1.2)

    for ki in range(n_kp):
        for ai in range(n_area):
            r_val = sna_r[ki, ai]
            p_fdr = sna_p_fdr[ki, ai]
            art   = bool(artifact_flags[ki, ai])

            if np.isnan(r_val):
                ax_sna.text(ai, ki, "—\n(no data)", ha="center", va="center",
                            fontsize=8, color="gray", style="italic")
                continue

            txt_col  = "white" if abs(r_val) >= 0.28 else "black"
            ss       = sig_stars(p_fdr)
            art_mark = " (!)" if art else ""
            p_str    = f"p={p_fdr:.3f}" if not np.isnan(p_fdr) else "p=—"
            line1    = f"{r_val:+.2f}{ss}"
            line2    = f"{p_str}{art_mark}"
            ax_sna.text(ai, ki, f"{line1}\n{line2}",
                        ha="center", va="center",
                        fontsize=9.5, color=txt_col,
                        linespacing=1.5)

    # ── Shared colorbar ───────────────────────────────────────────────────────
    sm = ScalarMappable(cmap=plt.cm.RdBu_r, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax_cbar)
    cbar.set_label("rank-biserial r\n(W2T vs Opto)", fontsize=9.5, labelpad=8)
    cbar.ax.tick_params(labelsize=8.5)
    cbar.set_ticks([-0.5, -0.25, 0, 0.25, 0.5])

    png_path = OUTDIR / "opto_summary_v3_heatmap.png"
    svg_path = OUTDIR / "opto_summary_v3_heatmap.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {svg_path}")

    # ═══════════════════════════════════════════════════════════════════════════
    # CSV
    # ═══════════════════════════════════════════════════════════════════════════
    csv_rows = []
    for ki, kp_name in enumerate(keypoints):
        for ai, area in enumerate(BRAIN_AREAS_ORDERED):
            c_same, c_total = int(mla_consist[ki, ai, 0]), int(mla_consist[ki, ai, 1])
            consist_str = f"{c_same}/{c_total}" if c_total > 0 else ""
            nr_list = nan_rate_accum[kp_name][area]
            frames_nan_pct = float(np.mean(nr_list)) * 100 if nr_list else np.nan
            csv_rows.append({
                "cell":              f"{kp_name} x {area}",
                "mla_effect_pooled": mla_r[ki, ai],
                "mla_p_fisher":      mla_p[ki, ai],
                "mla_p_fdr":         mla_p_fdr[ki, ai],
                "mla_n_mice":        int(mla_n[ki, ai]),
                "mla_consistency":   consist_str,
                "sna_effect":        sna_r[ki, ai],
                "sna_p":             sna_p[ki, ai],
                "sna_p_fdr":         sna_p_fdr[ki, ai],
                "artifact_flag":     bool(artifact_flags[ki, ai]),
                "classification":    cell_class[ki, ai],
                "frames_nan_pct":    frames_nan_pct,
            })
    csv_df   = pd.DataFrame(csv_rows)
    csv_path = OUTDIR / "opto_summary_v3_data.csv"
    csv_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Markdown report
    # ═══════════════════════════════════════════════════════════════════════════
    now_str = start_ts.strftime("%Y-%m-%d %H:%M:%S")

    # ── Effect size tables ────────────────────────────────────────────────────
    hdr = " | ".join(f"{a:^12}" for a in BRAIN_AREAS_ORDERED)
    sep = " | ".join(f"{'-'*12}" for _ in BRAIN_AREAS_ORDERED)

    mla_tbl_lines = [
        f"| {'Keypoint':<22} | {hdr} |",
        f"|{'-'*24}|{sep}|",
    ]
    sna_tbl_lines = [
        f"| {'Keypoint':<22} | {hdr} |",
        f"|{'-'*24}|{sep}|",
    ]
    for ki, kp_name in enumerate(keypoints):
        mla_cells, sna_cells = [], []
        for ai in range(n_area):
            r_m = mla_r[ki, ai]; p_m = mla_p_fdr[ki, ai]
            c_same, c_total = int(mla_consist[ki, ai, 0]), int(mla_consist[ki, ai, 1])
            if np.isnan(r_m):
                mla_cells.append(f"{'—':^12}")
            else:
                ss = sig_stars(p_m)
                ok = (c_same >= 2)
                ss_d = ss if ok else (f"({ss})" if ss else "")
                mla_cells.append(f"{r_m:+.3f}{ss_d} ({c_same}/{c_total})")

            r_s = sna_r[ki, ai]; p_s = sna_p_fdr[ki, ai]
            art = bool(artifact_flags[ki, ai])
            if np.isnan(r_s):
                sna_cells.append(f"{'—':^12}")
            else:
                ss = sig_stars(p_s)
                art_m = " (!)" if art else ""
                sna_cells.append(f"{r_s:+.3f}{ss}{art_m}")

        mla_tbl_lines.append(
            f"| {PUB_KEYPOINT_LABELS[kp_name]:<22} | "
            + " | ".join(f"{c:^12}" for c in mla_cells) + " |"
        )
        sna_tbl_lines.append(
            f"| {PUB_KEYPOINT_LABELS[kp_name]:<22} | "
            + " | ".join(f"{c:^12}" for c in sna_cells) + " |"
        )
    mla_table = "\n".join(mla_tbl_lines)
    sna_table = "\n".join(sna_tbl_lines)

    # ── Per-mouse consistency ─────────────────────────────────────────────────
    consist_lines = []
    for ki, kp_name in enumerate(keypoints):
        kp_label = PUB_KEYPOINT_LABELS[kp_name]
        for ai, area in enumerate(BRAIN_AREAS_ORDERED):
            if np.isnan(mla_r[ki, ai]):
                continue
            pm = mla_per_mouse_r.get((ki, ai), {})
            parts = []
            for mouse in MLA_MICE:
                if mouse in pm:
                    short = mouse.split("_")[0][-8:]
                    parts.append(f"{short}: r={pm[mouse]:+.3f}")
            c_same, c_total = int(mla_consist[ki, ai, 0]), int(mla_consist[ki, ai, 1])
            flag = "  ← INCONSISTENT" if c_same < 2 else ""
            consist_lines.append(
                f"- **{kp_label} × {area}** [{c_same}/{c_total}]: "
                + ", ".join(parts) + flag
            )
    consist_text = "\n".join(consist_lines) if consist_lines else "(no data)"

    # ── Bio-highlight summary ─────────────────────────────────────────────────
    bio_lines = []
    for ki, kp_name in enumerate(keypoints):
        kp_label = PUB_KEYPOINT_LABELS[kp_name]
        row_cls  = [cell_class[ki, ai] for ai in range(n_area)]
        robust   = [BRAIN_AREAS_ORDERED[ai] for ai, c in enumerate(row_cls) if "🟢" in c]
        likely   = [BRAIN_AREAS_ORDERED[ai] for ai, c in enumerate(row_cls) if "🟡" in c]
        artifact = [BRAIN_AREAS_ORDERED[ai] for ai, c in enumerate(row_cls) if "🔴" in c]
        not_sig  = [BRAIN_AREAS_ORDERED[ai] for ai, c in enumerate(row_cls)
                    if "Not significant" in c]
        parts = []
        if robust:   parts.append(f"🟢 Robust real: {', '.join(robust)}")
        if likely:   parts.append(f"🟡 Likely real: {', '.join(likely)}")
        if artifact: parts.append(f"🔴 Artifact-suspect: {', '.join(artifact)}")
        if not_sig:  parts.append(f"◻ n.s.: {', '.join(not_sig)}")
        if parts:
            bio_lines.append(f"- **{kp_label}**: " + "; ".join(parts))
    bio_text = "\n".join(bio_lines) if bio_lines else "(no significant MLA effects)"

    # ── SNA opposite-direction notes ──────────────────────────────────────────
    opposite_lines = []
    for ki, kp_name in enumerate(keypoints):
        for ai, area in enumerate(BRAIN_AREAS_ORDERED):
            mla_v = mla_r[ki, ai]; sna_v = sna_r[ki, ai]
            if np.isnan(mla_v) or np.isnan(sna_v):
                continue
            if np.sign(sna_v) != np.sign(mla_v) and abs(sna_v) >= 0.05:
                opposite_lines.append(
                    f"  - **{PUB_KEYPOINT_LABELS[kp_name]} × {area}**: "
                    f"MLA r={mla_v:+.3f}, SNA r={sna_v:+.3f} (opposite direction)"
                )
    opposite_text = ("\n".join(opposite_lines)
                     if opposite_lines else "  (no notable opposite-direction effects)")

    # ── NaN rate summary for interpolation audit ──────────────────────────────
    nan_summary_lines = []
    for kp_name in keypoints:
        all_rates = []
        for area in BRAIN_AREAS_ORDERED:
            all_rates.extend(nan_rate_accum[kp_name][area])
        mean_nan = float(np.mean(all_rates)) * 100 if all_rates else np.nan
        source = PUB_KEYPOINTS[kp_name]["source"].upper()
        nan_summary_lines.append(
            f"  - {PUB_KEYPOINT_LABELS[kp_name]} ({source}): "
            f"{mean_nan:.1f}% raw NaN frames (with 0.8 threshold)"
        )
    nan_text = "\n".join(nan_summary_lines)

    # ── Sessions tables ───────────────────────────────────────────────────────
    used_hdr  = ("| Mouse | Date | Sfx | Area | Group | Opto dur"
                 " | Phase | n_W2T | n_Opto |")
    used_sep  = "|-------|------|-----|------|-------|----------|-------|-------|--------|"
    used_rows = [used_hdr, used_sep]
    for s in used_sessions:
        used_rows.append(
            f"| {s['mouse']} | {s['date']} | {s['suffix']} | {s['area']}"
            f" | {s['group']} | {s['opto_duration']}s | {s['phase']}"
            f" | {s['n_w2t']} | {s['n_opto']} |"
        )
    used_table = "\n".join(used_rows)

    skip_rows = ["| Mouse | Date | Sfx | Reason |", "|-------|------|-----|--------|"]
    for s in skipped_sessions:
        skip_rows.append(
            f"| {s['mouse']} | {s['date']} | {s['suffix']} | {s['reason']} |"
        )
    skip_table = "\n".join(skip_rows)

    # ── Assemble report ───────────────────────────────────────────────────────
    report = f"""# Opto Summary Heatmap v2 — Analysis Report

**Date:** {now_str}
**Script:** pose_opto_summary_v3.py
**Output:** {OUTDIR}

---

## Section: Interpolation Audit

### Status Quo (unchanged from pose_utils.py)

Interpolation is **already implemented** inside `pose_utils.slice_timeseries_to_trial`
(line ~477–541 of pose_utils.py). The function:

1. Extracts the raw irregular-timestamp signal within the HIT window ± 0.5 s margin.
2. Builds a **linear interpolator** on valid (non-NaN) frames only.
3. Resamples to the uniform {RESAMPLE_HZ:.0f} Hz output grid.
4. **Re-NaNs** any output point farther than `MAX_GAP = 0.1 s` from the nearest valid
   input frame → effectively fills NaN gaps ≤ **15 frames at {RESAMPLE_HZ:.0f} Hz**.

### Assessment

- `MAX_GAP = 0.1 s = 15 frames` is slightly above the 7-frame guideline from Jelte's
  pipeline. It is conservative enough: whisker DLC often has brief occlusion gaps of
  5–10 frames during crossings. Leaving it at 15 frames avoids over-NaN-ing those gaps.
- `max_gap` is **not configurable** as a function argument in the current code.
- No change was made to `slice_timeseries_to_trial` (constraint: pose_utils.py read-only).

### v2 Change: DLC Confidence Threshold 0.9 → 0.8

`CONFIDENCE_THRESHOLDS['dlc'] = 0.9` (pose_config.py, read-only) is overridden locally
to `CONF_THR_OVERRIDE['dlc'] = 0.8` in this script. This matches Jelte's pipeline
default and retains more DLC frames as valid before interpolation is needed.

Effect: more frames are valid, reducing NaN gaps that `slice_timeseries_to_trial` must
interpolate over. SLEAP threshold unchanged at 0.6.

### Raw NaN Rate per Keypoint (with 0.8 DLC threshold)

{nan_text}

*(These are session-mean NaN rates in the raw keypoint timeseries across all contributing
sessions. They represent the upper bound on frames that required interpolation.)*

---

## Section: Effect Size Tables

*(MLA: FDR-corrected; consistency (same/total) in parentheses; stars in () = sign-inconsistent)*
*(SNA: FDR-corrected indicative; (!) = artifact flag)*

### TOP Panel — MLA Mice (opsin-expressing, n ≤ 3)

{mla_table}

### BOTTOM Panel — SNA Control (wildtype SNA-145894_1, n = 1)

{sna_table}

---

## Section: Robust Effects Classification

| Keypoint x Area | MLA r | SNA r | Classification |
|-----------------|-------|-------|----------------|
"""

    for ki, kp_name in enumerate(keypoints):
        for ai, area in enumerate(BRAIN_AREAS_ORDERED):
            mla_v = mla_r[ki, ai]; sna_v = sna_r[ki, ai]
            cls   = cell_class[ki, ai]
            mla_s = f"{mla_v:+.3f}" if not np.isnan(mla_v) else "—"
            sna_s = f"{sna_v:+.3f}" if not np.isnan(sna_v) else "—"
            report += (f"| {PUB_KEYPOINT_LABELS[kp_name]} x {area}"
                       f" | {mla_s} | {sna_s} | {cls} |\n")

    report += f"""
**Summary: {n_robust} 🟢 Robust real | {n_likely} 🟡 Likely real | {n_artifact} 🔴 Artifact-suspect | {n_not_sig} ◻ n.s.**

### SNA Opposite-Direction Effects (supports robust interpretation)

{opposite_text}

---

## Section: Per-Mouse Consistency (MLA)

*(format: [same-sign/total mice]; ← INCONSISTENT marks cells where < 2 mice agree)*

{consist_text}

---

## Section: Bio-Highlight Summary

{bio_text}

---

## Sessions Used ({len(used_sessions)})

{used_table}

---

## Sessions Skipped ({len(skipped_sessions)})

{skip_table}

---

## Known Limitations

- MLA-026805_2 has **only M1 sessions** → S1/M2/mPFC MLA cells have n=2 only.
- SNA-145894_1 has **only opto_blocked** sessions (opto_random excluded: sync collapse).
  Both blocked and random pooled for MLA mice.
- Fisher's method (MLA cells) does not check sign consistency — stars can appear even
  with opposite-direction mice. Parenthesised stars "(***)" flag this case.
- SNA n=1: FDR-corrected stars are indicative only; single-mouse statistics cannot
  support biological inference without replication.
- DLC threshold 0.9→0.8 change increases data yield vs v1 but may include some frames
  of marginally lower tracking quality.
"""

    # ── v2 vs v3 classification comparison ───────────────────────────────────
    v2_csv = Path("f:/analysis/bpod/Documents/output_opto_summary_v2/opto_summary_v2_data.csv")
    compare_block = "\n---\n\n## v2 vs v3 Classification Comparison (Angle-Wrap Fix)\n\n"
    if v2_csv.exists():
        try:
            v2_df  = pd.read_csv(v2_csv)
            v3_rows = csv_rows   # already computed above
            compare_lines = ["| Cell | v2 class | v3 class | MLA r v2 | MLA r v3 | Changed? |",
                             "|------|----------|----------|----------|----------|----------|"]
            changed = 0
            for ki, kp_name in enumerate(keypoints):
                for ai, area in enumerate(BRAIN_AREAS_ORDERED):
                    cell_label = f"{kp_name} x {area}"
                    v2_row = v2_df[v2_df["cell"] == cell_label]
                    v3_row = next((r for r in v3_rows if r["cell"] == cell_label), None)
                    if v2_row.empty or v3_row is None:
                        continue
                    v2_cls = v2_row.iloc[0]["classification"]
                    v3_cls = v3_row["classification"]
                    v2_r   = v2_row.iloc[0]["mla_effect_pooled"]
                    v3_r   = v3_row["mla_effect_pooled"]
                    ch     = "YES <<" if v2_cls != v3_cls else "no"
                    if v2_cls != v3_cls:
                        changed += 1
                    compare_lines.append(
                        f"| {cell_label} | {v2_cls} | {v3_cls} "
                        f"| {v2_r:.3f} | {v3_r:.3f} | {ch} |"
                    )
            compare_block += "\n".join(compare_lines)
            compare_block += f"\n\n**Total classification changes: {changed}/16**\n"
            if changed == 0:
                compare_block += "\nAll classifications identical — angle-wrap fix did not change the opto story.\n"
            else:
                compare_block += "\n**Changed cells are marked << above.**\n"
        except Exception as exc:
            compare_block += f"\n*(Comparison failed: {exc})*\n"
    else:
        compare_block += "\n*(v2 CSV not found — skipping comparison)*\n"

    report += compare_block
    report_path = OUTDIR / "opto_summary_v3_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Saved: {report_path}")

    # ── ZIP ───────────────────────────────────────────────────────────────────
    zip_path = Path("f:/analysis/bpod/Documents/output_opto_summary_v3.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in [png_path, svg_path, csv_path, report_path]:
            if f.exists():
                zf.write(f, f.name)

    zip_mb = zip_path.stat().st_size / 1e6
    print(f"\nZIP: {zip_path}  ({zip_mb:.2f} MB)")
    print("\n" + "=" * 65)
    print(f"DONE -- {n_robust} Robust | {n_likely} Likely | {n_artifact} Artifact-suspect")
    print(f"ZIP:  {zip_path}")
    print("=" * 65)


if __name__ == "__main__":
    run_analysis_v2()

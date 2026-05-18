# -*- coding: utf-8 -*-
"""
pose_trial_aligned.py — Trial-aligned kinematics: HIT vs Miss comparison.

For every session, pose trajectories are aligned to the Audio state onset
and split by trial outcome (HIT / Miss) and trial type (W2T / Opto).
The primary analysis uses training-phase W2T trials to establish what
correct vs incorrect trials look like in terms of body kinematics.

Figures produced
----------------
trial_aligned/hit_vs_miss_{feature}.pdf   — mean ± CI traces, FDR significance
trial_aligned/heatmap_{mouse}_{feature}.pdf — all-trial raster sorted by RT
trial_aligned/trajectory_2d_{mouse}.pdf    — nose x/y in post-audio window

Usage
-----
    python pose_trial_aligned.py
"""

import gc
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import mannwhitneyu
from collections import defaultdict

from pose_config import (
    MICE, FIGURES_PATH, RESULTS_PATH, MOUSE_COLORS,
    SLEAP_KEYPOINTS, DLC_KEYPOINTS, CONFIDENCE_THRESHOLDS,
    TRIAL_WINDOW_SEC, BASELINE_WINDOW_SEC, RESAMPLE_HZ,
    HIT_COLOR, MISS_COLOR, FIGURE_SIZE_WIDE, FIGURE_SIZE_PANEL,
    build_session_registry, build_all_features,
)
from pose_utils import (
    load_session_pickle, extract_trial_table,
    get_pose_object, get_pose_timestamps,
    extract_keypoint_timeseries, extract_pupil_size_timeseries,
    slice_timeseries_to_trial, baseline_subtract, compute_trial_speed,
)
from bpod_utils import bootstrap_ci, fdr_bh

OUTDIR = FIGURES_PATH / "trial_aligned"
OUTDIR.mkdir(parents=True, exist_ok=True)
RESDIR = RESULTS_PATH
RESDIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Keypoints to extract per analysis
# ─────────────────────────────────────────────────────────────────────────────

# Dict: feature_name → (source, video_idx, keypoint_name, coord)
#   coord is 'x', 'y', 'speed', or 'pupil'
# All body parts from all cameras across SLEAP, DLC, and Facemap (182 features).
DEFAULT_FEATURES = build_all_features()


# ─────────────────────────────────────────────────────────────────────────────
# Per-session extraction
# ─────────────────────────────────────────────────────────────────────────────

def _build_time_grid(window_sec: tuple, resample_hz: float) -> np.ndarray:
    pre, post = window_sec
    n = int(round((post - pre) * resample_hz)) + 1
    return np.linspace(pre, post, n)


def compute_session_trial_kinematics(
    session_dict: dict,
    trial_table: pd.DataFrame,
    features: dict | None = None,
    window_sec: tuple = TRIAL_WINDOW_SEC,
    resample_hz: float = RESAMPLE_HZ,
) -> dict:
    """
    Extract trial-aligned kinematic traces for one session.

    Parameters
    ----------
    session_dict : dict
        Raw pickle dict. NOT deleted here — caller is responsible.
    trial_table : pd.DataFrame
        Output of extract_trial_table().
    features : dict or None
        Feature spec dict (see DEFAULT_FEATURES). None → use DEFAULT_FEATURES.
    window_sec : tuple
    resample_hz : float

    Returns
    -------
    dict with keys
        't_relative'  : np.ndarray  (n_timepoints,)
        'outcomes'    : np.ndarray  (n_trials,) str
        'trial_types' : np.ndarray  (n_trials,) str
        'reaction_times' : np.ndarray (n_trials,)
        per feature_name : np.ndarray (n_trials, n_timepoints)
    """
    if features is None:
        features = DEFAULT_FEATURES

    t_rel = _build_time_grid(window_sec, resample_hz)
    n_trials = len(trial_table)
    n_t      = len(t_rel)

    result = {
        "t_relative":     t_rel,
        "outcomes":       trial_table["outcome"].values,
        "trial_types":    trial_table["trial_type"].values,
        "reaction_times": trial_table["reaction_time"].values,
    }
    for feat in features:
        result[feat] = np.full((n_trials, n_t), np.nan)

    for feat, (source, vid, kp_name, coord) in features.items():
        conf_thr = CONFIDENCE_THRESHOLDS.get(source)

        if coord == "pupil":
            # Facemap pupil_size
            pose_obj = get_pose_object(session_dict, source, vid)
            if pose_obj is None:
                continue
            ts_pose, values = extract_pupil_size_timeseries(pose_obj)
        elif coord == "speed":
            pose_obj = get_pose_object(session_dict, source, vid)
            if pose_obj is None:
                continue
            ts_pose = get_pose_timestamps(pose_obj)
            x, y, _ = extract_keypoint_timeseries(pose_obj, kp_name, conf_thr)
            values   = compute_trial_speed(x, y, ts_pose)
        else:  # x or y
            pose_obj = get_pose_object(session_dict, source, vid)
            if pose_obj is None:
                continue
            ts_pose = get_pose_timestamps(pose_obj)
            x, y, _ = extract_keypoint_timeseries(pose_obj, kp_name, conf_thr)
            values   = x if coord == "x" else y

        # Slice each trial (use positional index, not DataFrame index)
        for pos, (_, trial_row) in enumerate(trial_table.iterrows()):
            audio_on = trial_row["audio_onset"]
            if np.isnan(audio_on):
                continue
            _, trace = slice_timeseries_to_trial(
                ts_pose, values, audio_on,
                window_sec[0], window_sec[1], resample_hz,
            )
            result[feat][pos] = trace

    # Baseline-subtract all features except reaction_time
    for feat in features:
        mat = result[feat]
        if mat.shape[0] > 0:
            result[feat] = baseline_subtract(mat, t_rel, BASELINE_WINDOW_SEC)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Cross-session aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_kinematics_across_sessions(
    sessions: list[dict],
    features: dict | None = None,
    phase_filter: str | None = None,
    trial_type_filter: str | None = None,
) -> dict:
    """
    Memory-safe loop: load → extract → append → del → gc.collect().

    Returns
    -------
    dict
        {mouse_id: {feature: np.ndarray(n_trials_total, n_t),
                    'outcomes': ..., 'trial_types': ...,
                    'reaction_times': ..., 't_relative': ...}}
    """
    if features is None:
        features = DEFAULT_FEATURES
    if phase_filter is not None:
        sessions = [s for s in sessions if s["phase"] == phase_filter]

    # Pre-build time grid from a dummy call
    t_rel = _build_time_grid(TRIAL_WINDOW_SEC, RESAMPLE_HZ)

    aggregated = {mouse: defaultdict(list) for mouse in MICE}

    for idx, rec in enumerate(sessions):
        mouse = rec["mouse"]
        tag   = f"[{idx+1}/{len(sessions)}] {mouse} {rec['date']}"
        print(f"  Loading {tag} ...", end=" ", flush=True)

        try:
            d  = load_session_pickle(rec["pickle_path"])
            tt = extract_trial_table(d)

            if trial_type_filter is not None:
                tt = tt[tt["trial_type"] == trial_type_filter].reset_index(drop=True)
            if tt.empty:
                print("(no trials after filter) skip")
                del d; gc.collect()
                continue

            sess_result = compute_session_trial_kinematics(d, tt, features)
            del d; gc.collect()

            aggregated[mouse]["outcomes"].append(sess_result["outcomes"])
            aggregated[mouse]["trial_types"].append(sess_result["trial_types"])
            aggregated[mouse]["reaction_times"].append(sess_result["reaction_times"])
            for feat in features:
                aggregated[mouse][feat].append(sess_result[feat])

            print(f"n_trials={len(tt)}")

        except Exception as e:
            warnings.warn(f"{tag}: {e}")
            try:
                del d
            except NameError:
                pass
            gc.collect()

    # Concatenate lists → arrays
    result = {}
    for mouse in MICE:
        md = aggregated[mouse]
        if not md:
            continue
        mouse_dict = {"t_relative": t_rel}
        for key in ("outcomes", "trial_types"):
            if md[key]:
                mouse_dict[key] = np.concatenate(md[key])
        if md["reaction_times"]:
            mouse_dict["reaction_times"] = np.concatenate(md["reaction_times"])
        for feat in features:
            if md[feat]:
                mouse_dict[feat] = np.vstack(md[feat])
        result[mouse] = mouse_dict

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mean_ci(traces: np.ndarray, n_boot: int = 5000) -> tuple:
    """Return (mean, lo, hi) per timepoint, using bootstrap_ci across trials."""
    n_t = traces.shape[1]
    mean = np.nanmean(traces, axis=0)
    lo   = np.full(n_t, np.nan)
    hi   = np.full(n_t, np.nan)
    for t in range(n_t):
        col = traces[:, t]
        col = col[~np.isnan(col)]
        if len(col) >= 2:
            lo[t], hi[t] = bootstrap_ci(col, n_boot=n_boot)
    return mean, lo, hi


def plot_hit_vs_miss_traces(
    aggregated: dict,
    feature_name: str,
    output_path: Path,
    n_boot: int = 5000,
) -> None:
    """
    Mean ± bootstrap CI traces for HIT and Miss trials.
    One panel per mouse + grand-average panel.
    FDR-corrected Mann-Whitney significance band below x-axis.
    """
    mice_with_data = [m for m in MICE if m in aggregated
                      and feature_name in aggregated[m]]
    if not mice_with_data:
        print(f"  No data for {feature_name}")
        return

    t_rel  = aggregated[mice_with_data[0]]["t_relative"]
    n_cols = len(mice_with_data) + 1
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4), sharey=False)
    if n_cols == 1:
        axes = [axes]

    grand_hit, grand_miss = [], []

    for ax, mouse in zip(axes[:-1], mice_with_data):
        md       = aggregated[mouse]
        traces   = md[feature_name]
        outcomes = md["outcomes"]

        hit_traces  = traces[outcomes == "HIT"]
        miss_traces = traces[outcomes == "Miss"]

        if hit_traces.shape[0] > 0:
            mh, lh, hh = _mean_ci(hit_traces, n_boot)
            ax.plot(t_rel, mh, color=HIT_COLOR, linewidth=2, label=f"HIT (n={len(hit_traces)})")
            ax.fill_between(t_rel, lh, hh, color=HIT_COLOR, alpha=0.2)
            grand_hit.append(hit_traces)

        if miss_traces.shape[0] > 0:
            mm, lm, hm = _mean_ci(miss_traces, n_boot)
            ax.plot(t_rel, mm, color=MISS_COLOR, linewidth=2, label=f"Miss (n={len(miss_traces)})")
            ax.fill_between(t_rel, lm, hm, color=MISS_COLOR, alpha=0.2)
            grand_miss.append(miss_traces)

        # FDR-corrected significance
        if hit_traces.shape[0] >= 3 and miss_traces.shape[0] >= 3:
            pvals = []
            for t in range(len(t_rel)):
                h = hit_traces[:, t][~np.isnan(hit_traces[:, t])]
                m = miss_traces[:, t][~np.isnan(miss_traces[:, t])]
                if len(h) >= 3 and len(m) >= 3:
                    _, p = mannwhitneyu(h, m, alternative="two-sided")
                else:
                    p = 1.0
                pvals.append(p)
            pvals_fdr = fdr_bh(np.array(pvals))
            sig_band  = pvals_fdr < 0.05
            yl = ax.get_ylim()
            y_sig = yl[0] - 0.05 * (yl[1] - yl[0])
            ax.scatter(t_rel[sig_band], [y_sig] * sig_band.sum(),
                       color="black", s=4, marker="|")

        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Time from Audio onset (s)")
        ax.set_ylabel(feature_name.replace("_", " "))
        ax.set_title(mouse.split("_")[0], fontsize=9)
        ax.legend(frameon=False, fontsize=7)
        ax.grid(True, alpha=0.3)

    # Grand-average panel
    ax = axes[-1]
    if grand_hit:
        all_hit = np.vstack(grand_hit)
        mh, lh, hh = _mean_ci(all_hit, n_boot)
        ax.plot(t_rel, mh, color=HIT_COLOR, linewidth=2.5,
                label=f"HIT (n={len(all_hit)})")
        ax.fill_between(t_rel, lh, hh, color=HIT_COLOR, alpha=0.2)
    if grand_miss:
        all_miss = np.vstack(grand_miss)
        mm, lm, hm = _mean_ci(all_miss, n_boot)
        ax.plot(t_rel, mm, color=MISS_COLOR, linewidth=2.5,
                label=f"Miss (n={len(all_miss)})")
        ax.fill_between(t_rel, lm, hm, color=MISS_COLOR, alpha=0.2)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Time from Audio onset (s)")
    ax.set_title("Grand Average", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Trial-aligned: {feature_name.replace('_', ' ')} — HIT vs Miss",
                 fontsize=11)
    plt.tight_layout()
    output_path = Path(output_path)
    for ext in ("pdf", "png"):
        fig.savefig(output_path.with_suffix(f".{ext}"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path.stem}")


def plot_trial_heatmap(
    aggregated: dict,
    feature_name: str,
    mouse: str,
    output_path: Path,
) -> None:
    """
    All-trial heatmap for one mouse, sorted by reaction time.
    Outcome strip (HIT=red, Miss=gray) on the right margin.
    """
    if mouse not in aggregated or feature_name not in aggregated[mouse]:
        return

    md       = aggregated[mouse]
    traces   = md[feature_name]
    outcomes = md["outcomes"]
    rts      = md.get("reaction_times", np.full(len(outcomes), np.nan))
    t_rel    = md["t_relative"]

    if traces.shape[0] == 0:
        return

    # Sort by reaction time (NaN at bottom)
    sort_key = np.where(np.isnan(rts), 999, rts)
    order    = np.argsort(sort_key)
    traces_sorted   = traces[order]
    outcomes_sorted = outcomes[order]

    vmax = np.nanpercentile(np.abs(traces_sorted), 95)
    vmin = -vmax

    fig, (ax_heat, ax_strip) = plt.subplots(
        1, 2, figsize=(9, max(3, traces_sorted.shape[0] * 0.05)),
        gridspec_kw={"width_ratios": [20, 1]},
    )

    im = ax_heat.imshow(
        traces_sorted, aspect="auto", cmap="RdBu_r",
        vmin=vmin, vmax=vmax, interpolation="nearest",
        extent=[t_rel[0], t_rel[-1], 0, traces_sorted.shape[0]],
    )
    ax_heat.axvline(0, color="white", linewidth=0.8, linestyle="--")
    ax_heat.set_xlabel("Time from Audio onset (s)")
    ax_heat.set_ylabel("Trial (sorted by reaction time)")
    ax_heat.set_title(f"{mouse.split('_')[0]} — {feature_name.replace('_', ' ')}")
    plt.colorbar(im, ax=ax_heat, label="Baseline-subtracted value", shrink=0.8)

    # Outcome strip
    outcome_colors = np.array(
        [[int(c.lstrip("#")[i:i+2], 16)/255 for i in (0, 2, 4)]
         for c in [HIT_COLOR if o == "HIT" else MISS_COLOR
                   for o in outcomes_sorted]]
    )
    ax_strip.imshow(
        outcome_colors.reshape(-1, 1, 3), aspect="auto",
        extent=[0, 1, 0, len(outcomes_sorted)],
    )
    ax_strip.set_xticks([])
    ax_strip.set_yticks([])
    ax_strip.set_title("Out", fontsize=7)

    plt.tight_layout()
    output_path = Path(output_path)
    for ext in ("pdf", "png"):
        fig.savefig(output_path.with_suffix(f".{ext}"), dpi=150)
    plt.close(fig)


def plot_2d_trajectory(
    aggregated: dict,
    mouse: str,
    output_path: Path,
    x_feature: str = "nose_x",
    y_feature:  str = "nose_y",
    time_window: tuple = (0.0, 0.5),
) -> None:
    """
    Mean 2D nose trajectory in the post-audio window, HIT vs Miss.
    Color encodes time within window.
    """
    if mouse not in aggregated:
        return
    if x_feature not in aggregated[mouse] or y_feature not in aggregated[mouse]:
        return

    md       = aggregated[mouse]
    t_rel    = md["t_relative"]
    outcomes = md["outcomes"]
    x_traces = md[x_feature]
    y_traces = md[y_feature]

    t_mask = (t_rel >= time_window[0]) & (t_rel <= time_window[1])
    t_win  = t_rel[t_mask]
    n_win  = t_mask.sum()
    if n_win == 0:
        return

    fig, ax = plt.subplots(figsize=(5, 5))
    cmap = plt.cm.viridis
    norm = plt.Normalize(t_win[0], t_win[-1])

    for outcome, color_base in (("HIT", HIT_COLOR), ("Miss", MISS_COLOR)):
        sel = outcomes == outcome
        if not sel.any():
            continue
        mean_x = np.nanmean(x_traces[sel][:, t_mask], axis=0)
        mean_y = np.nanmean(y_traces[sel][:, t_mask], axis=0)

        for i in range(n_win - 1):
            ax.plot([mean_x[i], mean_x[i+1]], [mean_y[i], mean_y[i+1]],
                    color=cmap(norm(t_win[i])),
                    linewidth=3 if outcome == "HIT" else 1.5,
                    alpha=0.9, solid_capstyle="round")

        ax.scatter(mean_x[0], mean_y[0], s=60, color=cmap(norm(t_win[0])),
                   edgecolors="black", linewidths=0.5, zorder=5)
        ax.text(mean_x[-1], mean_y[-1], outcome,
                fontsize=8, color=color_base, va="bottom")

    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                 label="Time from Audio onset (s)", shrink=0.7)
    ax.set_xlabel("Nose x (px, baseline-subtracted)")
    ax.set_ylabel("Nose y (px, baseline-subtracted)")
    ax.set_title(f"{mouse.split('_')[0]} — 2D nose trajectory ({time_window[0]}–{time_window[1]} s)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = Path(output_path)
    for ext in ("pdf", "png"):
        fig.savefig(output_path.with_suffix(f".{ext}"), dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_hit_vs_miss_analysis(features: dict | None = None) -> None:
    """
    End-to-end trial-aligned HIT vs Miss analysis.

    Loads training-phase W2T trials only (no opto contamination).
    Produces trace plots, heatmaps, and 2D trajectory plots.
    """
    if features is None:
        features = DEFAULT_FEATURES

    print("\n" + "=" * 65)
    print("POSE — Trial-aligned HIT vs Miss Analysis (training phase)")
    print("=" * 65)

    sessions = build_session_registry(exist_only=True)

    print("\nAggregating kinematics (training × W2T trials)...")
    aggregated = aggregate_kinematics_across_sessions(
        sessions,
        features=features,
        phase_filter="training",
        trial_type_filter="W2T",
    )

    print("\nPlotting HIT vs Miss traces...")
    for feat in features:
        plot_hit_vs_miss_traces(
            aggregated, feat,
            output_path=OUTDIR / f"hit_vs_miss_{feat}",
        )

    print("\nPlotting trial heatmaps...")
    for mouse in MICE:
        if mouse not in aggregated:
            continue
        for feat in ("nose_speed", "pupil_size"):
            if feat not in aggregated.get(mouse, {}):
                continue
            plot_trial_heatmap(
                aggregated, feat, mouse,
                output_path=OUTDIR / f"heatmap_{mouse}_{feat}",
            )

    print("\nPlotting 2D nose trajectories...")
    for mouse in MICE:
        if mouse not in aggregated:
            continue
        plot_2d_trajectory(
            aggregated, mouse,
            output_path=OUTDIR / f"trajectory_2d_{mouse}",
        )

    # Save summary CSV: mean metrics per mouse × outcome
    rows = []
    for mouse, md in aggregated.items():
        for outcome in ("HIT", "Miss"):
            sel = md.get("outcomes", np.array([])) == outcome
            n   = sel.sum()
            if n == 0:
                continue
            rts = md.get("reaction_times", np.array([]))[sel]
            rows.append({
                "mouse":            mouse,
                "outcome":          outcome,
                "n_trials":         int(n),
                "mean_reaction_time": float(np.nanmean(rts)),
                "median_reaction_time": float(np.nanmedian(rts)),
            })
    df_summary = pd.DataFrame(rows)
    csv_out = RESDIR / "trial_aligned_summary.csv"
    df_summary.to_csv(csv_out, index=False)
    print(f"\nSummary saved: {csv_out}")
    print("=" * 65)


if __name__ == "__main__":
    run_hit_vs_miss_analysis()

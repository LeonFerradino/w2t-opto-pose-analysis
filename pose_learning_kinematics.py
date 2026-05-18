# -*- coding: utf-8 -*-
"""
pose_learning_kinematics.py — Stereotypy development across 11 training days.

Tracks how body kinematics become more consistent and stereotyped as mice
learn the auditory detection task. Key metrics per session:
  - Stereotypy index: mean pairwise Pearson r of HIT trial speed traces
  - PC1 variance explained: fraction of movement variance in the first PC
  - Median reaction time

Figures produced
----------------
learning/stereotypy_{feature}.pdf   — per-mouse + group mean learning curve
learning/reaction_time.pdf          — reaction time across training
learning/pca_variance.pdf           — PC1 variance explained by day
learning/umap_training.pdf          — UMAP of all frames colored by day

Usage
-----
    python pose_learning_kinematics.py
"""

import gc
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import combinations
import statsmodels.formula.api as smf

from pose_config import (
    MICE, FIGURES_PATH, RESULTS_PATH, MOUSE_COLORS, MOUSE_MARKERS,
    SLEAP_KEYPOINTS, CONFIDENCE_THRESHOLDS,
    TRIAL_WINDOW_SEC, BASELINE_WINDOW_SEC, RESAMPLE_HZ,
    HIT_WINDOW_SEC, HIT_BASELINE_SEC,
    PUB_KEYPOINTS, PUB_KEYPOINT_LABELS,
    FIGURE_SIZE_WIDE, PHASE_COLORS,
    build_session_registry, build_all_features,
)
from pose_utils import (
    load_session_pickle, extract_trial_table,
    get_pose_object, get_pose_timestamps,
    extract_keypoint_timeseries, extract_pupil_size_timeseries,
    extract_whisker_angle_timeseries,
    slice_timeseries_to_trial, baseline_subtract, compute_trial_speed,
    build_keypoint_feature_matrix, fit_pca_on_session, run_umap_on_features,
)
from bpod_utils import bootstrap_ci

OUTDIR = FIGURES_PATH / "learning"
OUTDIR.mkdir(parents=True, exist_ok=True)
RESDIR = RESULTS_PATH
RESDIR.mkdir(parents=True, exist_ok=True)

# Features to compute stereotypy for: speed traces + pupil from all cameras.
# Derived from build_all_features() so naming stays consistent across scripts.
STEREO_FEATURES = {
    name: spec
    for name, spec in build_all_features().items()
    if spec[3] in ("speed", "pupil")
}

# Key features for publication overview figures (subset of STEREO_FEATURES)
# Order matters: whiskers first (task-relevant), then head/body, then pupil
KEY_FEATURES_LEARNING = [f for f in [
    "dlc0_C2_mid_speed",
    "dlc0_C1_mid_speed",
    "s0_nosetip_speed",
    "s0_tm_joint_speed",
    "s0_frontpawL_speed",
    "fm0_pupil",
] if f in STEREO_FEATURES]

FEATURE_LABELS = {
    "s0_nosetip_speed":    "Nose",
    "s0_noseback_speed":   "Noseback",
    "s0_pupilL_speed":     "Pupil L",
    "s0_earbaseL_speed":   "Ear base",
    "s0_eartipL_speed":    "Ear tip",
    "s0_lowerlip_speed":   "Lower lip",
    "s0_tm_joint_speed":   "TM joint",
    "s0_frontpawL_speed":  "Forepaw L",
    "s0_frontpawR_speed":  "Forepaw R",
    "s0_shoulder_speed":   "Shoulder",
    "dlc0_C1_mid_speed":   "C1 whisker",
    "dlc0_C1_end_speed":   "C1 tip",
    "dlc0_C1_start_speed": "C1 base",
    "dlc0_C2_mid_speed":   "C2 whisker",
    "dlc0_C2_end_speed":   "C2 tip",
    "dlc0_C2_start_speed": "C2 base",
    "dlc0_nose_speed":     "Nose (DLC)",
    "fm0_pupil":           "Pupil",
    "fm1_pupil":           "Pupil (cam1)",
}


HIT_COLOR  = "#1565C0"
MISS_COLOR = "#C62828"


def style_ax(ax):
    """Remove top and right spines for publication-quality axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_stereotypy_index(
    trial_traces: np.ndarray,
    t_relative: np.ndarray,
    response_window: tuple = (0.0, 1.0),
) -> float:
    """
    Mean pairwise Pearson r of trial traces within the response window.

    High value → animal moves the same way on every trial.

    Parameters
    ----------
    trial_traces : np.ndarray  (n_trials, n_timepoints)
    t_relative : np.ndarray    (n_timepoints,)
    response_window : tuple    (start, end) in seconds

    Returns
    -------
    float   Mean r, or np.nan if < 3 trials.
    """
    mask = (t_relative >= response_window[0]) & (t_relative <= response_window[1])
    if not mask.any():
        return np.nan

    traces_win = trial_traces[:, mask]
    # Drop trials where > 50% of the window is NaN
    row_nan = np.mean(np.isnan(traces_win), axis=1)
    traces_win = traces_win[row_nan < 0.5]

    n = traces_win.shape[0]
    if n < 3:
        return np.nan

    # Replace remaining NaN with column mean for correlation
    col_means = np.nanmean(traces_win, axis=0)
    for col in range(traces_win.shape[1]):
        nan_idx = np.isnan(traces_win[:, col])
        traces_win[nan_idx, col] = col_means[col]

    rs = []
    for i, j in combinations(range(n), 2):
        r = np.corrcoef(traces_win[i], traces_win[j])[0, 1]
        if not np.isnan(r):
            rs.append(r)

    return float(np.mean(rs)) if rs else np.nan


def compute_session_pca_variance(
    session_dict: dict,
    trial_table: pd.DataFrame,
    source: str = "sleap",
    video_idx: int = 0,
    keypoints: list | None = None,
    trial_type_filter: str = "W2T",
    outcome_filter: str = "HIT",
    n_pca_components: int = 10,
) -> dict:
    """
    Fit PCA on trial-aligned feature matrices from filtered trials.

    Returns PC1 variance explained and scree curve.

    Parameters
    ----------
    session_dict : dict
    trial_table : pd.DataFrame
    source : str
    video_idx : int
    keypoints : list or None   If None, uses SLEAP_KEYPOINTS
    trial_type_filter, outcome_filter : str
    n_pca_components : int

    Returns
    -------
    dict  keys: pc1_variance_explained, explained_variance_ratio,
                cumulative_variance_ratio, n_trials_used
    """
    if keypoints is None:
        keypoints = SLEAP_KEYPOINTS

    conf_thr = CONFIDENCE_THRESHOLDS.get(source)
    pose_obj = get_pose_object(session_dict, source, video_idx)
    if pose_obj is None:
        return {"pc1_variance_explained": np.nan,
                "explained_variance_ratio": np.array([]),
                "cumulative_variance_ratio": np.array([]),
                "n_trials_used": 0}

    ts_pose = get_pose_timestamps(pose_obj)
    t_rel   = np.linspace(TRIAL_WINDOW_SEC[0], TRIAL_WINDOW_SEC[1],
                          int(round((TRIAL_WINDOW_SEC[1] - TRIAL_WINDOW_SEC[0])
                                    * RESAMPLE_HZ)) + 1)

    # Filter trials
    sel = trial_table
    if trial_type_filter:
        sel = sel[sel["trial_type"] == trial_type_filter]
    if outcome_filter:
        sel = sel[sel["outcome"] == outcome_filter]
    if sel.empty:
        return {"pc1_variance_explained": np.nan,
                "explained_variance_ratio": np.array([]),
                "cumulative_variance_ratio": np.array([]),
                "n_trials_used": 0}

    # Build per-trial × keypoints feature matrix (time axis flattened)
    # Each trial → 1 row = concatenated [x_kp0_t0..tN, y_kp0_t0..tN, x_kp1..., ...]
    trial_rows = []
    for _, tr in sel.iterrows():
        audio_on = tr["audio_onset"]
        if np.isnan(audio_on):
            continue
        feat_row = []
        for kp in keypoints:
            x, y, _ = extract_keypoint_timeseries(pose_obj, kp, conf_thr)
            _, x_al = slice_timeseries_to_trial(ts_pose, x, audio_on,
                                                TRIAL_WINDOW_SEC[0],
                                                TRIAL_WINDOW_SEC[1], RESAMPLE_HZ)
            _, y_al = slice_timeseries_to_trial(ts_pose, y, audio_on,
                                                TRIAL_WINDOW_SEC[0],
                                                TRIAL_WINDOW_SEC[1], RESAMPLE_HZ)
            feat_row.extend(x_al.tolist())
            feat_row.extend(y_al.tolist())
        trial_rows.append(feat_row)

    if len(trial_rows) < 3:
        return {"pc1_variance_explained": np.nan,
                "explained_variance_ratio": np.array([]),
                "cumulative_variance_ratio": np.array([]),
                "n_trials_used": len(trial_rows)}

    X = np.array(trial_rows)
    # Drop columns with too many NaN
    nan_frac = np.mean(np.isnan(X), axis=0)
    X = X[:, nan_frac < 0.3]
    # Drop rows with any remaining NaN
    valid_rows = ~np.any(np.isnan(X), axis=1)
    X = X[valid_rows]

    if X.shape[0] < 3 or X.shape[1] < 2:
        return {"pc1_variance_explained": np.nan,
                "explained_variance_ratio": np.array([]),
                "cumulative_variance_ratio": np.array([]),
                "n_trials_used": X.shape[0]}

    n_comp = min(n_pca_components, X.shape[0] - 1, X.shape[1])
    pca, _ = fit_pca_on_session(X, n_components=n_comp)
    evr    = pca.explained_variance_ratio_

    return {
        "pc1_variance_explained":  float(evr[0]) if len(evr) > 0 else np.nan,
        "explained_variance_ratio": evr,
        "cumulative_variance_ratio": np.cumsum(evr),
        "n_trials_used": X.shape[0],
    }


def _extract_speed_traces(session_dict, trial_table, feat_spec) -> tuple:
    """
    Extract HIT trial speed traces for stereotypy computation.
    Returns (traces array, t_relative array).
    """
    source, vid, kp_name, coord = feat_spec
    conf_thr = CONFIDENCE_THRESHOLDS.get(source)
    pose_obj = get_pose_object(session_dict, source, vid)
    if pose_obj is None:
        return np.empty((0, 0)), np.array([])

    t_rel = np.linspace(TRIAL_WINDOW_SEC[0], TRIAL_WINDOW_SEC[1],
                        int(round((TRIAL_WINDOW_SEC[1] - TRIAL_WINDOW_SEC[0])
                                  * RESAMPLE_HZ)) + 1)
    ts_pose = get_pose_timestamps(pose_obj)

    if coord == "pupil":
        _, values = extract_pupil_size_timeseries(pose_obj)
    elif coord == "speed":
        x, y, _ = extract_keypoint_timeseries(pose_obj, kp_name, conf_thr)
        values   = compute_trial_speed(x, y, ts_pose)
    else:
        x, y, _ = extract_keypoint_timeseries(pose_obj, kp_name, conf_thr)
        values   = x if coord == "x" else y

    hit_trials = trial_table[(trial_table["trial_type"] == "W2T") &
                              (trial_table["outcome"] == "HIT")]
    traces = []
    for _, tr in hit_trials.iterrows():
        if np.isnan(tr["audio_onset"]):
            continue
        _, trace = slice_timeseries_to_trial(
            ts_pose, values, tr["audio_onset"],
            TRIAL_WINDOW_SEC[0], TRIAL_WINDOW_SEC[1], RESAMPLE_HZ,
        )
        traces.append(trace)

    if not traces:
        return np.empty((0, len(t_rel))), t_rel

    mat = np.vstack(traces)
    mat = baseline_subtract(mat, t_rel, BASELINE_WINDOW_SEC)
    return mat, t_rel


# ─────────────────────────────────────────────────────────────────────────────
# Cross-session aggregation
# ─────────────────────────────────────────────────────────────────────────────

def compute_learning_curves(
    sessions: list | None = None,
) -> tuple:
    """
    Memory-safe loop over training sessions.

    For each session: load → extract metrics → del pickle → gc.collect().

    Returns
    -------
    tuple (df, trace_store, t_rel)
        df : pd.DataFrame  columns: mouse, date, day_index, phase,
            stereotypy_{feature} for each feature in STEREO_FEATURES,
            mean_reaction_time_sec, median_reaction_time_sec,
            pc1_variance_explained, n_hit_trials, n_miss_trials
        trace_store : dict  {feature: {"early": [mean_trace, ...],
                                       "late":  [mean_trace, ...]}}
            "early" = days 1-3, "late" = days 9-11
        t_rel : np.ndarray  common time axis for traces
    """
    if sessions is None:
        sessions = build_session_registry(exist_only=True)
    sessions = [s for s in sessions if s["phase"] == "training"]

    t_rel = np.linspace(TRIAL_WINDOW_SEC[0], TRIAL_WINDOW_SEC[1],
                        int(round((TRIAL_WINDOW_SEC[1] - TRIAL_WINDOW_SEC[0])
                                  * RESAMPLE_HZ)) + 1)
    trace_store = {f: {"early": [], "late": []} for f in KEY_FEATURES_LEARNING}

    rows = []
    for idx, rec in enumerate(sessions):
        mouse     = rec["mouse"]
        day_index = rec["day_index"]
        tag       = f"[{idx+1}/{len(sessions)}] {mouse} day {day_index}"
        print(f"  {tag} ...", end=" ", flush=True)

        try:
            d  = load_session_pickle(rec["pickle_path"])
            tt = extract_trial_table(d)

            n_hit  = (tt["outcome"] == "HIT").sum()
            n_miss = (tt["outcome"] == "Miss").sum()

            # Reaction time
            hit_rts = tt.loc[tt["outcome"] == "HIT", "reaction_time"].dropna().values
            mean_rt   = float(np.nanmean(hit_rts))   if len(hit_rts) > 0 else np.nan
            median_rt = float(np.nanmedian(hit_rts)) if len(hit_rts) > 0 else np.nan

            # PCA variance
            pca_res = compute_session_pca_variance(d, tt)
            pc1_var = pca_res["pc1_variance_explained"]

            # Stereotypy per feature + collect mean traces for key features
            stereo_vals = {}
            for feat_name, feat_spec in STEREO_FEATURES.items():
                traces, t_rel_feat = _extract_speed_traces(d, tt, feat_spec)
                if traces.shape[0] >= 3:
                    stereo_vals[feat_name] = compute_stereotypy_index(traces, t_rel_feat)
                    if feat_name in KEY_FEATURES_LEARNING:
                        mean_tr = np.nanmean(traces, axis=0)
                        if day_index <= 3:
                            trace_store[feat_name]["early"].append(mean_tr)
                        elif day_index >= 9:
                            trace_store[feat_name]["late"].append(mean_tr)
                else:
                    stereo_vals[feat_name] = np.nan

            del d; gc.collect()

            row = {
                "mouse":                    mouse,
                "date":                     rec["date"],
                "day_index":                day_index,
                "phase":                    rec["phase"],
                "n_hit_trials":             int(n_hit),
                "n_miss_trials":            int(n_miss),
                "mean_reaction_time_sec":   mean_rt,
                "median_reaction_time_sec": median_rt,
                "pc1_variance_explained":   pc1_var,
            }
            for feat_name, val in stereo_vals.items():
                row[f"stereotypy_{feat_name}"] = val

            rows.append(row)
            print(f"HIT={n_hit}  RT={median_rt:.3f}s  "
                  f"stereo_nose={stereo_vals.get('s0_nosetip_speed', np.nan):.2f}")

        except Exception as e:
            warnings.warn(f"{tag}: {e}")
            try:
                del d
            except NameError:
                pass
            gc.collect()
            rows.append({"mouse": mouse, "date": rec["date"],
                          "day_index": day_index, "phase": rec["phase"]})

    return pd.DataFrame(rows), trace_store, t_rel


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────

def fit_lme_learning_trend(
    learning_df: pd.DataFrame,
    outcome_col: str,
) -> dict:
    """
    Fit: outcome_col ~ day_index + (1 | mouse).

    Random intercepts only (n=4 mice too small for random slopes).

    Parameters
    ----------
    learning_df : pd.DataFrame
    outcome_col : str

    Returns
    -------
    dict  keys: slope, slope_pval, slope_ci_95, intercept, model_summary
    """
    sub = learning_df[["mouse", "day_index", outcome_col]].dropna()
    if sub["mouse"].nunique() < 2 or len(sub) < 5:
        return {"slope": np.nan, "slope_pval": np.nan,
                "slope_ci_95": (np.nan, np.nan), "intercept": np.nan,
                "model_summary": "insufficient data"}
    try:
        model  = smf.mixedlm(f"{outcome_col} ~ day_index", sub, groups=sub["mouse"])
        result = model.fit(reml=True, method="lbfgs")
        ci     = result.conf_int()
        return {
            "slope":       float(result.fe_params["day_index"]),
            "slope_pval":  float(result.pvalues["day_index"]),
            "slope_ci_95": (float(ci.loc["day_index", 0]),
                            float(ci.loc["day_index", 1])),
            "intercept":   float(result.fe_params["Intercept"]),
            "model_summary": result.summary().as_text(),
        }
    except Exception as e:
        return {"slope": np.nan, "slope_pval": np.nan,
                "slope_ci_95": (np.nan, np.nan), "intercept": np.nan,
                "model_summary": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_learning_curves(
    learning_df: pd.DataFrame,
    outcome_col: str,
    y_label: str,
    output_path: Path,
    n_boot: int = 5000,
) -> None:
    """
    Per-mouse thin lines (colored) + bootstrap CI ribbon for cross-mouse mean.
    Annotate with LME slope and p-value.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)

    all_vals_by_day = {}
    for mouse in MICE:
        sub = learning_df[learning_df["mouse"] == mouse][
            ["day_index", outcome_col]].dropna()
        if sub.empty:
            continue
        ax.plot(sub["day_index"], sub[outcome_col],
                color=MOUSE_COLORS.get(mouse, "gray"),
                linewidth=1.2, alpha=0.55, marker="o", markersize=4,
                label=mouse.split("_")[0])
        for _, row in sub.iterrows():
            d = int(row["day_index"])
            all_vals_by_day.setdefault(d, []).append(row[outcome_col])

    days = sorted(all_vals_by_day)
    if days:
        means = [np.nanmean(all_vals_by_day[d]) for d in days]
        lo_arr = [bootstrap_ci(all_vals_by_day[d], n_boot=n_boot)[0] for d in days]
        hi_arr = [bootstrap_ci(all_vals_by_day[d], n_boot=n_boot)[1] for d in days]
        ax.plot(days, means, color="black", linewidth=2.5,
                marker="o", markersize=5, label="Group mean", zorder=5)
        ax.fill_between(days, lo_arr, hi_arr, color="black",
                        alpha=0.15, label="95% Bootstrap CI")

    # LME annotation
    lme = fit_lme_learning_trend(learning_df, outcome_col)
    slope = lme.get("slope", np.nan)
    pval  = lme.get("slope_pval", np.nan)
    if not np.isnan(slope):
        sig = ("***" if pval < 0.001 else "**" if pval < 0.01
               else "*" if pval < 0.05 else "ns")
        ax.text(0.02, 0.97,
                f"LME slope = {slope:+.4f}  p = {pval:.4f}  {sig}",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round", fc="white", alpha=0.7))

    ax.set_xlabel("Training day")
    ax.set_ylabel(y_label)
    ax.set_title(f"Learning Curve — {y_label}")
    ax.legend(frameon=False, fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = Path(output_path)
    for ext in ("pdf", "png"):
        fig.savefig(output_path.with_suffix(f".{ext}"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path.stem}")


def plot_pca_variance_by_day(
    learning_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    PC1 variance explained vs training day (group mean + per-mouse lines).
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    col = "pc1_variance_explained"

    all_vals_by_day = {}
    for mouse in MICE:
        sub = learning_df[learning_df["mouse"] == mouse][
            ["day_index", col]].dropna()
        if sub.empty:
            continue
        ax.plot(sub["day_index"], sub[col],
                color=MOUSE_COLORS.get(mouse, "gray"),
                linewidth=1.0, alpha=0.5, marker="o", markersize=3,
                label=mouse.split("_")[0])
        for _, row in sub.iterrows():
            all_vals_by_day.setdefault(int(row["day_index"]), []).append(row[col])

    days = sorted(all_vals_by_day)
    if days:
        means = [np.nanmean(all_vals_by_day[d]) for d in days]
        ax.plot(days, means, color="black", linewidth=2.5,
                marker="o", markersize=5, label="Group mean", zorder=5)

    ax.set_xlabel("Training day")
    ax.set_ylabel("PC1 variance explained")
    ax.set_title("PCA — Fraction of Movement Variance in PC1 across Learning")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = Path(output_path)
    for ext in ("pdf", "png"):
        fig.savefig(output_path.with_suffix(f".{ext}"), dpi=150)
    plt.close(fig)


def plot_stereotypy_overview(
    learning_df: pd.DataFrame,
    key_features: list,
    output_path: Path,
) -> None:
    """
    Publication-ready multi-panel stereotypy overview (Hwang / Peters style).

    One panel per key body part. Each panel shows stereotypy index vs training
    day: per-mouse thin colored lines + group mean ± SEM black line.
    LME p-value annotated in red (significant) or gray (n.s.) at panel top.

    Parameters
    ----------
    learning_df : pd.DataFrame  output of compute_learning_curves()
    key_features : list         feature names (must be stereotypy_{feat} columns)
    output_path : Path
    """
    valid = [f for f in key_features
             if f"stereotypy_{f}" in learning_df.columns]
    if not valid:
        print("  plot_stereotypy_overview: no valid features, skipping")
        return

    n = len(valid)
    fig, axes = plt.subplots(1, n, figsize=(2.8 * n, 3.6), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, feat in zip(axes, valid):
        col = f"stereotypy_{feat}"

        all_vals_by_day: dict = {}
        for mouse in MICE:
            sub = learning_df[learning_df["mouse"] == mouse][
                ["day_index", col]].dropna()
            if sub.empty:
                continue
            ax.plot(sub["day_index"].values, sub[col].values,
                    color=MOUSE_COLORS.get(mouse, "gray"),
                    linewidth=1.0, alpha=0.55,
                    marker="o", markersize=3.5, zorder=2)
            for _, row in sub.iterrows():
                all_vals_by_day.setdefault(int(row["day_index"]),
                                           []).append(row[col])

        days = sorted(all_vals_by_day)
        if days:
            means = np.array([np.nanmean(all_vals_by_day[d]) for d in days])
            sems  = np.array([
                np.nanstd(all_vals_by_day[d]) / max(np.sqrt(len(all_vals_by_day[d])), 1)
                for d in days
            ])
            ax.plot(days, means, color="black", linewidth=2.2,
                    marker="o", markersize=4.5, zorder=5)
            ax.fill_between(days, means - sems, means + sems,
                            color="black", alpha=0.12, zorder=4)

        # LME p-value annotation (Hwang style: red at top for significant)
        lme   = fit_lme_learning_trend(learning_df, col)
        pval  = lme.get("slope_pval", np.nan)
        if not np.isnan(pval):
            sig   = ("***" if pval < 0.001 else "**" if pval < 0.01
                     else "*" if pval < 0.05 else "n.s.")
            color = "crimson" if pval < 0.05 else "dimgray"
            ax.text(0.97, 0.97, sig,
                    transform=ax.transAxes, fontsize=9, va="top", ha="right",
                    color=color, fontweight="bold")
            ax.text(0.97, 0.87, f"p={pval:.3f}",
                    transform=ax.transAxes, fontsize=7, va="top", ha="right",
                    color=color)

        feat_label = FEATURE_LABELS.get(feat, feat.replace("_speed", "").replace("_", " "))
        ax.text(0.05, 0.97, feat_label, transform=ax.transAxes,
                fontsize=8, va="top", fontweight="bold", color="black")

        style_ax(ax)
        ax.set_xlabel("Training day", fontsize=9)
        ax.set_xticks([1, 4, 7, 11])
        if ax is axes[0]:
            ax.set_ylabel("Stereotypy index\n(mean pairwise r)", fontsize=9)

    fig.suptitle("Movement Stereotypy across Learning  (n = 4 mice, mean ± SEM)",
                 fontsize=9, y=1.03)
    plt.tight_layout()
    output_path = Path(output_path)
    for ext in ("pdf", "png"):
        fig.savefig(output_path.with_suffix(f".{ext}"), dpi=200,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.stem}")


def plot_early_late_traces(
    trace_store: dict,
    t_rel: np.ndarray,
    key_features: list,
    output_path: Path,
) -> None:
    """
    Early (days 1-3) vs Late (days 9-11) mean kinematic trace comparison.

    Layout: N rows (one per key feature) × 2 columns (early / late).
    Each panel: thin gray session-mean lines + thick colored group mean ± SEM.
    Inspired by Peters et al. 2017 Fig. 3c and Staab et al. 2025 Fig. 1.

    Parameters
    ----------
    trace_store : dict   {feature: {"early": [arr, ...], "late": [arr, ...]}}
    t_rel : np.ndarray   common time axis (from compute_learning_curves)
    key_features : list
    output_path : Path
    """
    valid = [f for f in key_features if f in trace_store]
    if not valid:
        return

    n = len(valid)
    EARLY_COLOR = "#555555"
    LATE_COLOR  = "#1565C0"   # strong blue for late

    fig, axes = plt.subplots(n, 2, figsize=(7, 2.4 * n),
                              sharex=True, sharey="row")
    if n == 1:
        axes = axes[None, :]

    phase_info = [
        ("early", EARLY_COLOR, "Days 1–3"),
        ("late",  LATE_COLOR,  "Days 9–11"),
    ]

    for ri, feat in enumerate(valid):
        for ci, (phase, color, label) in enumerate(phase_info):
            ax = axes[ri, ci]
            session_means = trace_store.get(feat, {}).get(phase, [])

            if session_means:
                mat = np.vstack(session_means)
                for tr in mat:
                    ax.plot(t_rel, tr, color=color, alpha=0.25,
                            linewidth=0.8, zorder=2)
                mean = np.nanmean(mat, axis=0)
                sem  = np.nanstd(mat, axis=0) / max(np.sqrt(mat.shape[0]), 1)
                ax.plot(t_rel, mean, color=color, linewidth=2.2, zorder=5)
                ax.fill_between(t_rel, mean - sem, mean + sem,
                                color=color, alpha=0.2, zorder=4)
                ax.text(0.97, 0.97, f"n={mat.shape[0]} sess",
                        transform=ax.transAxes, fontsize=6,
                        va="top", ha="right", color="gray")
            else:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=8, color="gray")

            ax.axvline(0, color="gray", linewidth=0.8, linestyle="--",
                       alpha=0.6, zorder=1)
            style_ax(ax)

            if ri == 0:
                ax.set_title(label, fontsize=9, color=color, fontweight="bold",
                             pad=4)
            if ri == n - 1:
                ax.set_xlabel("Time from cue onset (s)", fontsize=8)

        feat_label = FEATURE_LABELS.get(feat, feat.replace("_speed", "").replace("_", " "))
        axes[ri, 0].set_ylabel(feat_label, fontsize=8, labelpad=4)

    fig.suptitle("Kinematics: Early vs Late Training",
                 fontsize=10, y=1.01)
    plt.tight_layout()
    output_path = Path(output_path)
    for ext in ("pdf", "png"):
        fig.savefig(output_path.with_suffix(f".{ext}"), dpi=200,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.stem}")


def plot_umap_manifold(
    umap_embedding: np.ndarray,
    day_labels: np.ndarray,
    mouse: str,
    output_path: Path,
) -> None:
    """
    UMAP 2D embedding of kinematic frames colored by training day.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    days_unique = np.unique(day_labels[~np.isnan(day_labels.astype(float))])
    cmap  = plt.cm.plasma
    norm  = plt.Normalize(days_unique.min(), days_unique.max())

    for day in days_unique:
        sel = day_labels == day
        ax.scatter(umap_embedding[sel, 0], umap_embedding[sel, 1],
                   c=[cmap(norm(day))] * sel.sum(),
                   s=2, alpha=0.4, linewidths=0)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Training day", shrink=0.7)
    ax.set_title(f"UMAP — {mouse.split('_')[0]} — kinematics manifold by day")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    plt.tight_layout()
    output_path = Path(output_path)
    for ext in ("pdf", "png"):
        fig.savefig(output_path.with_suffix(f".{ext}"), dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Per-body-part learning figures (new)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_traces_hit_miss(
    session_dict: dict,
    trial_table: pd.DataFrame,
    feat_spec: tuple,
    t_rel: np.ndarray,
) -> tuple:
    """
    Extract HIT and MISS trial speed/pupil traces from a single session.

    Returns
    -------
    hit_mat  : np.ndarray  (n_hit_trials, n_timepoints), baseline-subtracted
    miss_mat : np.ndarray  (n_miss_trials, n_timepoints), baseline-subtracted
    """
    source, vid, kp_name, coord = feat_spec
    conf_thr = CONFIDENCE_THRESHOLDS.get(source)
    pose_obj = get_pose_object(session_dict, source, vid)
    empty    = np.empty((0, len(t_rel)))
    if pose_obj is None:
        return empty, empty

    ts_pose = get_pose_timestamps(pose_obj)

    if coord == "pupil":
        _, values = extract_pupil_size_timeseries(pose_obj)
    elif coord == "speed":
        x, y, _ = extract_keypoint_timeseries(pose_obj, kp_name, conf_thr)
        values   = compute_trial_speed(x, y, ts_pose)
    else:
        x, y, _ = extract_keypoint_timeseries(pose_obj, kp_name, conf_thr)
        values   = x if coord == "x" else y

    hit_traces, miss_traces = [], []
    for outcome, container in [("HIT", hit_traces), ("Miss", miss_traces)]:
        subset = trial_table[
            (trial_table["trial_type"] == "W2T") &
            (trial_table["outcome"]    == outcome)
        ]
        for _, tr in subset.iterrows():
            if np.isnan(tr["audio_onset"]):
                continue
            _, trace = slice_timeseries_to_trial(
                ts_pose, values, tr["audio_onset"],
                TRIAL_WINDOW_SEC[0], TRIAL_WINDOW_SEC[1], RESAMPLE_HZ,
            )
            container.append(trace)

    def _stack(traces):
        if not traces:
            return empty
        mat = np.vstack(traces)
        return baseline_subtract(mat, t_rel, BASELINE_WINDOW_SEC)

    return _stack(hit_traces), _stack(miss_traces)


def _response_mean(traces: np.ndarray, t_rel: np.ndarray,
                   window: tuple = (0.0, 1.0)) -> float:
    """Mean signal amplitude across trials within the response window."""
    if traces.shape[0] == 0:
        return np.nan
    mask = (t_rel >= window[0]) & (t_rel <= window[1])
    if not mask.any():
        return np.nan
    return float(np.nanmean(traces[:, mask]))


def plot_bodypart_learning_figure(
    feat_name: str,
    feat_spec: tuple,
    sessions: list,
    outdir: Path,
) -> None:
    """
    One figure per body part (Peters / Hwang style).

    Layout: 1 row × 3 panels
      [A] Early (days 1–3)  — HIT (blue) vs MISS (red) trial-aligned traces
      [B] Late (days 9–11)  — same
      [C] Response magnitude across training days (HIT + MISS, per-mouse
          colored lines + group mean ± SEM, LME p-value in red)

    Thin gray = per-session means (individual variability).
    Thick colored lines = group mean ± SEM across sessions.
    """
    t_rel = np.linspace(
        TRIAL_WINDOW_SEC[0], TRIAL_WINDOW_SEC[1],
        int(round((TRIAL_WINDOW_SEC[1] - TRIAL_WINDOW_SEC[0]) * RESAMPLE_HZ)) + 1,
    )

    # ── Data collection ───────────────────────────────────────────────────────
    early_hit_means:  list[np.ndarray] = []
    early_miss_means: list[np.ndarray] = []
    late_hit_means:   list[np.ndarray] = []
    late_miss_means:  list[np.ndarray] = []
    metric_rows: list[dict] = []

    for rec in sessions:
        if rec["phase"] != "training":
            continue
        day = rec["day_index"]
        tag = f"{rec['mouse']} day {day}"
        try:
            d       = load_session_pickle(rec["pickle_path"])
            tt      = extract_trial_table(d)
            hit_mat, miss_mat = _extract_traces_hit_miss(d, tt, feat_spec, t_rel)
            del d; gc.collect()

            hit_val  = _response_mean(hit_mat,  t_rel)
            miss_val = _response_mean(miss_mat, t_rel)

            if hit_mat.shape[0] >= 2:
                hm = np.nanmean(hit_mat, axis=0)
                if day <= 3:
                    early_hit_means.append(hm)
                elif day >= 9:
                    late_hit_means.append(hm)

            if miss_mat.shape[0] >= 2:
                mm = np.nanmean(miss_mat, axis=0)
                if day <= 3:
                    early_miss_means.append(mm)
                elif day >= 9:
                    late_miss_means.append(mm)

            metric_rows.append({
                "mouse":     rec["mouse"],
                "day_index": day,
                "hit_val":   hit_val,
                "miss_val":  miss_val,
            })

        except Exception as e:
            warnings.warn(f"bodypart {feat_name} {tag}: {e}")
            try:
                del d
            except NameError:
                pass
            gc.collect()

    metric_df = pd.DataFrame(metric_rows)

    # ── Figure ────────────────────────────────────────────────────────────────
    feat_label = FEATURE_LABELS.get(feat_name, feat_name.replace("_speed", "").replace("_", " "))
    fig, axes  = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(feat_label, fontsize=12, fontweight="bold", y=1.01)

    # Helper: draw trace panel (panels A & B)
    def _draw_trace_panel(ax, hit_traces, miss_traces, title):
        # thin individual session-mean lines
        for tr in hit_traces:
            ax.plot(t_rel, tr, color=HIT_COLOR,  alpha=0.18, lw=0.9, zorder=2)
        for tr in miss_traces:
            ax.plot(t_rel, tr, color=MISS_COLOR, alpha=0.18, lw=0.9, zorder=2)

        # thick group mean ± SEM
        for traces, color, label in [
            (hit_traces,  HIT_COLOR,  "HIT"),
            (miss_traces, MISS_COLOR, "MISS"),
        ]:
            if len(traces) < 1:
                continue
            mat  = np.vstack(traces)
            mean = np.nanmean(mat, axis=0)
            sem  = np.nanstd(mat, axis=0) / max(np.sqrt(mat.shape[0]), 1)
            ax.plot(t_rel, mean, color=color, lw=2.2,
                    label=f"{label} (n={mat.shape[0]})", zorder=5)
            ax.fill_between(t_rel, mean - sem, mean + sem,
                            color=color, alpha=0.18, zorder=4)

        ax.axvline(0, color="gray", lw=0.8, ls="--", alpha=0.6, zorder=1)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xlabel("Time from cue onset (s)", fontsize=8)
        ax.set_xticks([-0.5, 0, 0.5, 1.0, 1.5])
        style_ax(ax)

    _draw_trace_panel(axes[0], early_hit_means, early_miss_means, "Early  (days 1–3)")
    _draw_trace_panel(axes[1], late_hit_means,  late_miss_means,  "Late  (days 9–11)")

    axes[0].set_ylabel("Speed (a.u., baseline-subtracted)", fontsize=8)
    axes[0].legend(frameon=False, fontsize=7, loc="upper right")
    axes[1].legend(frameon=False, fontsize=7, loc="upper right")

    # Panel C: metric across days
    ax = axes[2]
    all_hit_by_day:  dict[int, list] = {}
    all_miss_by_day: dict[int, list] = {}

    for mouse in MICE:
        sub = metric_df[metric_df["mouse"] == mouse]
        c   = MOUSE_COLORS.get(mouse, "gray")

        # per-mouse thin lines (HIT solid, MISS dashed)
        hit_sub  = sub.dropna(subset=["hit_val"])
        miss_sub = sub.dropna(subset=["miss_val"])
        if not hit_sub.empty:
            ax.plot(hit_sub["day_index"], hit_sub["hit_val"],
                    color=c, lw=0.9, alpha=0.45, marker="o", ms=3, ls="-", zorder=2)
        if not miss_sub.empty:
            ax.plot(miss_sub["day_index"], miss_sub["miss_val"],
                    color=c, lw=0.9, alpha=0.45, marker="o", ms=3, ls="--", zorder=2)

        for _, row in sub.iterrows():
            d = int(row["day_index"])
            if not np.isnan(row.get("hit_val", np.nan)):
                all_hit_by_day.setdefault(d, []).append(row["hit_val"])
            if not np.isnan(row.get("miss_val", np.nan)):
                all_miss_by_day.setdefault(d, []).append(row["miss_val"])

    # group mean ± SEM lines
    for vals_by_day, color, label, ls in [
        (all_hit_by_day,  HIT_COLOR,  "HIT",  "-"),
        (all_miss_by_day, MISS_COLOR, "MISS", "--"),
    ]:
        days = sorted(vals_by_day)
        if not days:
            continue
        means = np.array([np.nanmean(vals_by_day[d]) for d in days])
        sems  = np.array([
            np.nanstd(vals_by_day[d]) / max(np.sqrt(len(vals_by_day[d])), 1)
            for d in days
        ])
        ax.plot(days, means, color=color, lw=2.2, marker="o", ms=4.5,
                ls=ls, label=label, zorder=5)
        ax.fill_between(days, means - sems, means + sems,
                        color=color, alpha=0.15, zorder=4)

    # LME p-value for HIT trend (Hwang style: red if significant)
    if "hit_val" in metric_df.columns and metric_df["hit_val"].notna().sum() >= 5:
        lme_df = (metric_df[["mouse", "day_index", "hit_val"]]
                  .rename(columns={"hit_val": "__m__"}))
        lme  = fit_lme_learning_trend(lme_df, "__m__")
        pval = lme.get("slope_pval", np.nan)
        if not np.isnan(pval):
            sig   = ("***" if pval < 0.001 else "**" if pval < 0.01
                     else "*" if pval < 0.05 else "n.s.")
            color = "crimson" if pval < 0.05 else "dimgray"
            ax.text(0.97, 0.97, f"HIT: p = {pval:.3f}  {sig}",
                    transform=ax.transAxes, fontsize=8, va="top", ha="right",
                    color=color, fontweight="bold")

    ax.set_xlabel("Training day", fontsize=8)
    ax.set_ylabel("Mean speed in 0–1 s window (a.u.)", fontsize=8)
    ax.set_title("Response across training", fontsize=9, fontweight="bold")
    ax.set_xticks([1, 4, 7, 11])
    ax.legend(frameon=False, fontsize=7)
    style_ax(ax)

    plt.tight_layout()
    out = outdir / f"bodypart_{feat_name}"
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: bodypart_{feat_name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_learning_analysis() -> None:
    """
    End-to-end learning kinematics analysis.

    Pass 1: compute scalar metrics per session (stereotypy, RT, PCA variance)
    Pass 2: collect subsampled feature matrices for UMAP
    """
    print("\n" + "=" * 65)
    print("POSE — Learning Kinematics (training days 1–11)")
    print("=" * 65)

    sessions = [s for s in build_session_registry() if s["phase"] == "training"]

    # ── Pupil coverage check ──────────────────────────────────────────────────
    # Drop fm0_pupil / fm1_pupil from KEY_FEATURES if Facemap data is absent
    # or sparse (< 50 % of sessions).  Prevents empty subplots.
    from pose_config import check_pupil_coverage
    global KEY_FEATURES_LEARNING
    if not check_pupil_coverage(sessions):
        print("  [INFO] Facemap pupil data sparse (<50 % of sessions) — "
              "dropping pupil from figures.")
        KEY_FEATURES_LEARNING = [f for f in KEY_FEATURES_LEARNING
                                  if "pupil" not in f]

    # ── Pass 1: scalar metrics + key trace collection ─────────────────────────
    print("\nPass 1: computing session metrics...")
    df, trace_store, t_rel = compute_learning_curves(sessions)
    csv_out = RESDIR / "learning_curves.csv"
    df.to_csv(csv_out, index=False)
    print(f"  Saved: {csv_out}")

    # Print LME results for key features
    print("\n--- LME: stereotypy ~ day_index + (1|mouse) ---")
    for feat in KEY_FEATURES_LEARNING:
        col = f"stereotypy_{feat}"
        if col not in df.columns:
            continue
        lme = fit_lme_learning_trend(df, col)
        sig = ("***" if lme["slope_pval"] < 0.001
               else "**"  if lme["slope_pval"] < 0.01
               else "*"   if lme["slope_pval"] < 0.05 else "ns")
        label = FEATURE_LABELS.get(feat, feat)
        print(f"  {label:<22} slope={lme['slope']:+.4f}  "
              f"p={lme['slope_pval']:.4f}  {sig}")

    # Also print RT and PCA
    for col, label in [
        ("median_reaction_time_sec", "Median reaction time"),
        ("pc1_variance_explained",   "PC1 variance explained"),
    ]:
        if col not in df.columns:
            continue
        lme = fit_lme_learning_trend(df, col)
        sig = ("***" if lme["slope_pval"] < 0.001
               else "**" if lme["slope_pval"] < 0.01
               else "*"  if lme["slope_pval"] < 0.05 else "ns")
        print(f"  {label:<22} slope={lme['slope']:+.4f}  "
              f"p={lme['slope_pval']:.4f}  {sig}")

    # ── Publication overview figures ──────────────────────────────────────────
    print("\nPlotting publication overview figures...")
    plot_stereotypy_overview(df, KEY_FEATURES_LEARNING,
                              output_path=OUTDIR / "stereotypy_overview")
    plot_early_late_traces(trace_store, t_rel, KEY_FEATURES_LEARNING,
                            output_path=OUTDIR / "early_late_traces")

    # ── Supplementary: individual feature plots ───────────────────────────────
    print("\nPlotting supplementary learning curves (all features)...")
    for feat in STEREO_FEATURES:
        col = f"stereotypy_{feat}"
        if col not in df.columns:
            continue
        label = FEATURE_LABELS.get(feat, feat.replace("_", " "))
        plot_learning_curves(df, col, f"Stereotypy — {label}",
                             output_path=OUTDIR / f"detail_{col}")

    for col, label in [
        ("median_reaction_time_sec", "Median Reaction Time (s)"),
        ("pc1_variance_explained",   "PC1 Variance Explained"),
    ]:
        if col not in df.columns:
            continue
        plot_learning_curves(df, col, label, output_path=OUTDIR / col)

    plot_pca_variance_by_day(df, output_path=OUTDIR / "pca_variance_by_day")

    # ── Per-body-part figures: HIT vs MISS traces + across-training metric ───
    print("\nPlotting per-body-part learning figures...")
    all_features = build_all_features()
    for feat_name in KEY_FEATURES_LEARNING:
        if feat_name not in all_features:
            continue
        try:
            plot_bodypart_learning_figure(
                feat_name, all_features[feat_name], sessions, OUTDIR
            )
        except Exception as e:
            warnings.warn(f"bodypart figure {feat_name}: {e}")

    # ── Pass 2: UMAP (per mouse, subsampled) ──────────────────────────────────
    print("\nPass 2: collecting subsampled features for UMAP (stride=10)...")
    for mouse in MICE:
        mouse_sessions = [s for s in sessions if s["mouse"] == mouse]
        if not mouse_sessions:
            continue

        feat_matrices = []
        day_labels    = []

        for rec in mouse_sessions:
            try:
                d = load_session_pickle(rec["pickle_path"])
                mat, mask = build_keypoint_feature_matrix(
                    d, "sleap", 0, SLEAP_KEYPOINTS,
                    confidence_threshold=CONFIDENCE_THRESHOLDS["sleap"],
                    stride=10,
                )
                del d; gc.collect()

                if mat.shape[0] > 0:
                    feat_matrices.append(mat)
                    day_labels.extend([rec["day_index"]] * mat.shape[0])
                    print(f"  {mouse} day {rec['day_index']}: "
                          f"{mat.shape[0]} valid frames")
            except Exception as e:
                warnings.warn(f"{mouse} {rec['date']}: UMAP pass failed: {e}")
                try:
                    del d
                except NameError:
                    pass
                gc.collect()

        if not feat_matrices:
            continue

        X_all      = np.vstack(feat_matrices)
        day_arr    = np.array(day_labels)

        try:
            print(f"  {mouse}: running UMAP on {X_all.shape[0]} frames...")
            embedding = run_umap_on_features(X_all, n_neighbors=15,
                                              min_dist=0.1, n_components=2)
            plot_umap_manifold(embedding, day_arr, mouse,
                               output_path=OUTDIR / f"umap_{mouse}")
            print(f"  {mouse}: UMAP saved")
        except Exception as e:
            warnings.warn(f"{mouse}: UMAP failed: {e}")

    print("\n" + "=" * 65)
    print("Learning kinematics analysis complete.")
    print(f"Results: {RESDIR}")
    print(f"Figures: {OUTDIR}")
    print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# Section 4.2 — hit-aligned kinematics (Figures A – D)
# ─────────────────────────────────────────────────────────────────────────────

# ── LearnScore helper ─────────────────────────────────────────────────────────
_RT_NAIVE, _RT_EXPERT   = 1.7757, 0.50
_SEM_NAIVE, _SEM_EXPERT = 0.0866, 0.0698


def _compute_session_learn_score(trial_table: pd.DataFrame) -> float:
    """
    Session-level LearnScore using the same formula as learn_index_mean+SEM.py.

    LS = 0.6 * hit_rate + 0.35 * norm(mean_RT) + 0.05 * norm(sem_RT)

    Returns np.nan when fewer than 3 W2T HIT trials exist.
    """
    w2t = trial_table[trial_table["trial_type"] == "W2T"]
    if w2t.empty:
        return np.nan
    hit_rate = float((w2t["outcome"] == "HIT").mean())
    hit_rts  = w2t.loc[w2t["outcome"] == "HIT", "reaction_time"].dropna().values
    if len(hit_rts) < 3:
        return np.nan
    mean_rt = float(np.nanmean(hit_rts))
    sem_rt  = float(np.nanstd(hit_rts, ddof=1) / np.sqrt(len(hit_rts)))
    norm_hr  = hit_rate
    norm_rt  = float(np.clip((_RT_NAIVE  - mean_rt) / (_RT_NAIVE  - _RT_EXPERT),  0, 1))
    norm_sem = float(np.clip((_SEM_NAIVE - sem_rt)  / (_SEM_NAIVE - _SEM_EXPERT), 0, 1))
    return 0.6 * norm_hr + 0.35 * norm_rt + 0.05 * norm_sem


# ── Core extraction ───────────────────────────────────────────────────────────

def extract_hit_aligned_traces_session(
    session_dict: dict,
    trial_table:  pd.DataFrame,
    window_sec:   tuple | None = None,
    resample_hz:  float | None = None,
) -> dict:
    """
    Extract hit-aligned traces for the 4 publication keypoints.

    Alignment anchor: HIT state onset (sensor-touch moment).
    Only W2T HIT trials with a valid hit_onset are included.
    Baseline subtraction uses HIT_BASELINE_SEC relative to hit onset.

    Parameters
    ----------
    session_dict : dict   Raw pickle (caller must del + gc.collect after use).
    trial_table  : pd.DataFrame   Output of extract_trial_table().
    window_sec, resample_hz : override defaults from pose_config.

    Returns
    -------
    dict
        't_relative'   : np.ndarray (n_t,)
        'n_hit_trials' : int
        For each key in PUB_KEYPOINTS:
            np.ndarray (n_hit_trials, n_t) — baseline-subtracted traces.
            All-NaN row when keypoint data unavailable for that trial.
    """
    if window_sec  is None:  window_sec  = HIT_WINDOW_SEC
    if resample_hz is None:  resample_hz = RESAMPLE_HZ

    t_rel = np.linspace(window_sec[0], window_sec[1],
                        int(round((window_sec[1] - window_sec[0]) * resample_hz)) + 1)
    n_t = len(t_rel)

    hit_mask   = (
        (trial_table["trial_type"] == "W2T") &
        (trial_table["outcome"]    == "HIT")  &
        (trial_table["hit_onset"].notna())
    )
    hit_trials = trial_table[hit_mask].reset_index(drop=True)
    n_hits     = len(hit_trials)

    result = {"t_relative": t_rel, "n_hit_trials": n_hits}

    for kp_name, spec in PUB_KEYPOINTS.items():
        traces   = np.full((n_hits, n_t), np.nan)
        source   = spec["source"]
        vid      = spec["vid"]
        conf_thr = CONFIDENCE_THRESHOLDS.get(source)
        coord    = spec["coord"]

        pose_obj = get_pose_object(session_dict, source, vid)
        if pose_obj is None:
            result[kp_name] = traces
            continue

        ts_pose = get_pose_timestamps(pose_obj)

        if coord == "angle":
            values = extract_whisker_angle_timeseries(
                pose_obj, spec["kp_start"], spec["kp_end"], conf_thr)
        else:  # "y" or "x"
            x, y, _ = extract_keypoint_timeseries(pose_obj, spec["kp"], conf_thr)
            values   = y if coord == "y" else x

        for i, (_, tr_row) in enumerate(hit_trials.iterrows()):
            _, trace = slice_timeseries_to_trial(
                ts_pose, values, float(tr_row["hit_onset"]),
                window_sec[0], window_sec[1], resample_hz,
            )
            traces[i] = trace

        traces = baseline_subtract(traces, t_rel, HIT_BASELINE_SEC)
        result[kp_name] = traces

    return result


# ── Per-timepoint bootstrap CI helper ────────────────────────────────────────

def _bootstrap_ci_traces(traces: np.ndarray, n_boot: int = 1000) -> tuple:
    """
    Per-timepoint 95% bootstrap CI for a (n_trials, n_t) matrix.
    Returns (lo, hi) each shape (n_t,), NaN where fewer than 2 valid samples.
    """
    n_t = traces.shape[1]
    lo  = np.full(n_t, np.nan)
    hi  = np.full(n_t, np.nan)
    for ti in range(n_t):
        col = traces[:, ti][~np.isnan(traces[:, ti])]
        if len(col) >= 2:
            lo[ti], hi[ti] = bootstrap_ci(col, n_boot=n_boot)
    return lo, hi


# ── Learning-curves v2: hit-aligned, position / angle metrics ────────────────

def compute_learning_curves_v2(sessions: list | None = None) -> tuple:
    """
    Memory-safe loop over training sessions.

    For each session:
        load pickle → extract hit-aligned traces for PUB_KEYPOINTS →
        compute stereotypy index (mean pairwise Pearson r) →
        compute session LearnScore → del pickle → gc.collect()

    The stereotypy index uses the full HIT_WINDOW_SEC window, matching the
    response window the animal produces during the task.

    Returns
    -------
    tuple (df, t_rel)
        df     : pd.DataFrame  columns: mouse, date, day_index, learn_score,
                               si_{kp} for each pub keypoint, n_hit_trials
        t_rel  : np.ndarray   common time axis
    """
    if sessions is None:
        sessions = build_session_registry(exist_only=True)
    sessions = [s for s in sessions if s["phase"] == "training"]

    t_rel = np.linspace(HIT_WINDOW_SEC[0], HIT_WINDOW_SEC[1],
                        int(round((HIT_WINDOW_SEC[1] - HIT_WINDOW_SEC[0])
                                  * RESAMPLE_HZ)) + 1)

    rows = []
    for idx, rec in enumerate(sessions):
        mouse = rec["mouse"]
        day   = rec["day_index"]
        tag   = f"[{idx+1}/{len(sessions)}] {mouse} day {day}"
        print(f"  {tag} ...", end=" ", flush=True)

        try:
            d  = load_session_pickle(rec["pickle_path"])
            tt = extract_trial_table(d)

            ls         = _compute_session_learn_score(tt)
            sess       = extract_hit_aligned_traces_session(d, tt)
            del d; gc.collect()

            n_hits = sess["n_hit_trials"]
            si_vals = {}
            for kp in PUB_KEYPOINTS:
                si_vals[kp] = compute_stereotypy_index(sess[kp], t_rel)

            row = {
                "mouse":        mouse,
                "date":         rec["date"],
                "day_index":    day,
                "phase":        rec["phase"],
                "n_hit_trials": n_hits,
                "learn_score":  ls,
            }
            for kp, si in si_vals.items():
                row[f"si_{kp}"] = si

            rows.append(row)
            si_nose = si_vals.get("nose_y", np.nan)
            print(f"n_hits={n_hits}  LS={ls:.3f}  si_nose={si_nose:.3f}")

        except Exception as e:
            warnings.warn(f"{tag}: {e}")
            try:
                del d
            except NameError:
                pass
            gc.collect()
            rows.append({"mouse": mouse, "date": rec["date"],
                         "day_index": day, "phase": rec["phase"]})

    return pd.DataFrame(rows), t_rel


# ── Expert-day finder ────────────────────────────────────────────────────────

def find_expert_day_per_mouse(learning_df: pd.DataFrame) -> dict:
    """
    Return {mouse: day_index} for the day with the highest session LearnScore.
    Used to define the expert session for Figure A and Figure C.
    """
    result = {}
    for mouse in MICE:
        sub = learning_df[learning_df["mouse"] == mouse].dropna(subset=["learn_score"])
        if sub.empty:
            continue
        best = sub.loc[sub["learn_score"].idxmax()]
        result[mouse] = int(best["day_index"])
        print(f"  Expert day {mouse}: day {result[mouse]} "
              f"(LS={best['learn_score']:.3f})")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Figure A — Hit-aligned trajectories: novice vs expert (per mouse)
# ─────────────────────────────────────────────────────────────────────────────

def plot_figure_A_per_mouse(
    sessions_registry: list,
    expert_days:       dict,
    outdir:            Path,
    n_boot:            int = 1000,
) -> None:
    """
    Figure A: hit-aligned individual-trial overlays, novice (day 1) vs expert.

    One PNG + PDF per mouse (4 total).
    Layout: 4 columns (pub keypoints) × 2 rows (novice / expert).
    Each panel: thin grey individual trials + bold median + 95% bootstrap CI.
    Biological story: diffuse trials in novice → tight cluster in expert.
    """
    training_sessions = {
        (rec["mouse"], rec["day_index"]): rec
        for rec in sessions_registry
        if rec["phase"] == "training"
    }

    for mouse in MICE:
        expert_day = expert_days.get(mouse)
        nov_rec    = training_sessions.get((mouse, 1))
        exp_rec    = training_sessions.get((mouse, expert_day)) if expert_day else None

        if nov_rec is None:
            print(f"  Figure A: no day-1 session for {mouse}, skipping")
            continue

        kp_list = list(PUB_KEYPOINTS.keys())
        n_kp    = len(kp_list)
        fig, axes = plt.subplots(2, n_kp,
                                  figsize=(3.5 * n_kp, 6.5),
                                  sharex=True, sharey=False)

        phase_recs = [
            (nov_rec, "Novice (Day 1)",
             "#999999"),
            (exp_rec, f"Expert (Day {expert_day})",
             MOUSE_COLORS.get(mouse, "steelblue")),
        ]

        for row_i, (rec, row_label, color) in enumerate(phase_recs):
            if rec is None:
                for col_i in range(n_kp):
                    axes[row_i, col_i].text(
                        0.5, 0.5, "no session",
                        transform=axes[row_i, col_i].transAxes,
                        ha="center", va="center", fontsize=8, color="gray")
                continue

            try:
                d    = load_session_pickle(rec["pickle_path"])
                tt   = extract_trial_table(d)
                sess = extract_hit_aligned_traces_session(d, tt)
                del d; gc.collect()
            except Exception as e:
                warnings.warn(f"Figure A {mouse} row {row_i}: {e}")
                try:
                    del d
                except NameError:
                    pass
                gc.collect()
                continue

            t_rel = sess["t_relative"]

            for col_i, kp in enumerate(kp_list):
                ax     = axes[row_i, col_i]
                traces = sess[kp]

                if traces.shape[0] == 0 or np.all(np.isnan(traces)):
                    ax.text(0.5, 0.5, "no data",
                            transform=ax.transAxes, ha="center", va="center",
                            fontsize=8, color="gray")
                    style_ax(ax)
                    continue

                # Individual trials (thin grey)
                for tr in traces:
                    if not np.all(np.isnan(tr)):
                        ax.plot(t_rel, tr, color="#cccccc", lw=0.5,
                                alpha=0.6, zorder=2)

                # Median + 95 % CI
                median      = np.nanmedian(traces, axis=0)
                lo_arr, hi_arr = _bootstrap_ci_traces(traces, n_boot=n_boot)
                ax.plot(t_rel, median, color=color, lw=2.2, zorder=5)
                ax.fill_between(t_rel, lo_arr, hi_arr,
                                color=color, alpha=0.25, zorder=4)

                ax.axvline(0, color="gray", lw=0.8, ls="--", alpha=0.6, zorder=1)
                style_ax(ax)

                # Labels
                if row_i == 0:
                    ax.set_title(PUB_KEYPOINT_LABELS.get(kp, kp),
                                 fontsize=9, fontweight="bold", pad=4)
                if col_i == 0:
                    ax.set_ylabel(row_label, fontsize=9, color=color,
                                  fontweight="bold", labelpad=4)
                if row_i == 1:
                    ax.set_xlabel("Time from hit (s)", fontsize=8)

                ax.text(0.97, 0.97, f"n={traces.shape[0]}",
                        transform=ax.transAxes, fontsize=6,
                        ha="right", va="top", color="gray")

        mouse_label = mouse.split("_(")[0]
        fig.suptitle(f"{mouse_label} — Hit-aligned kinematics: Novice vs Expert",
                     fontsize=10, fontweight="bold", y=1.01)
        plt.tight_layout()
        out = outdir / f"kin_figA_{mouse}"
        for ext in ("png", "pdf"):
            fig.savefig(out.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: kin_figA_{mouse}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure B — Stereotypy index across training (all mice + group mean)
# ─────────────────────────────────────────────────────────────────────────────

def plot_figure_B_stereotypy(
    learning_df: pd.DataFrame,
    outdir:      Path,
    n_boot:      int = 1000,
) -> None:
    """
    Figure B: stereotypy index (mean pairwise Pearson r) across 11 training days.

    Per-mouse coloured lines + thick black group mean ± 95% bootstrap CI.
    LME slope annotation (outcome ~ day + 1|mouse) in panel corner.
    One panel per publication keypoint.
    """
    kp_list    = list(PUB_KEYPOINTS.keys())
    valid_kps  = [kp for kp in kp_list if f"si_{kp}" in learning_df.columns]
    if not valid_kps:
        print("  Figure B: no si_ columns found, skipping")
        return

    n   = len(valid_kps)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 4.0), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, kp in zip(axes, valid_kps):
        col = f"si_{kp}"
        all_vals_by_day: dict[int, list] = {}

        for mouse in MICE:
            sub = learning_df[learning_df["mouse"] == mouse][
                ["day_index", col]].dropna()
            if sub.empty:
                continue
            color  = MOUSE_COLORS.get(mouse, "gray")
            marker = MOUSE_MARKERS.get(mouse, "o")
            ax.plot(sub["day_index"], sub[col],
                    color=color, lw=1.2, alpha=0.6,
                    marker=marker, ms=5, zorder=2)
            for _, row in sub.iterrows():
                all_vals_by_day.setdefault(int(row["day_index"]),
                                           []).append(row[col])

        days = sorted(all_vals_by_day)
        if days:
            means  = np.array([np.nanmean(all_vals_by_day[d]) for d in days])
            lo_arr = [bootstrap_ci(all_vals_by_day[d], n_boot=n_boot)[0]
                      for d in days]
            hi_arr = [bootstrap_ci(all_vals_by_day[d], n_boot=n_boot)[1]
                      for d in days]
            ax.plot(days, means, color="black", lw=2.5,
                    marker="o", ms=5, zorder=5, label="Group mean")
            ax.fill_between(days, lo_arr, hi_arr,
                            color="black", alpha=0.12, zorder=4,
                            label="95% CI")

        # LME slope annotation
        lme  = fit_lme_learning_trend(learning_df, col)
        pval = lme.get("slope_pval", np.nan)
        if not np.isnan(pval):
            sig   = ("***" if pval < 0.001 else "**" if pval < 0.01
                     else "*" if pval < 0.05 else "n.s.")
            clr   = "crimson" if pval < 0.05 else "dimgray"
            ax.text(0.97, 0.97, sig, transform=ax.transAxes, fontsize=9,
                    va="top", ha="right", color=clr, fontweight="bold")
            ax.text(0.97, 0.87, f"p={pval:.3f}", transform=ax.transAxes,
                    fontsize=7, va="top", ha="right", color=clr)

        ax.set_xlabel("Training day", fontsize=9)
        ax.set_xticks([1, 4, 7, 11])
        if ax is axes[0]:
            ax.set_ylabel("Stereotypy index\n(mean pairwise r)", fontsize=9)
        ax.text(0.05, 0.97, PUB_KEYPOINT_LABELS.get(kp, kp),
                transform=ax.transAxes, fontsize=8,
                va="top", fontweight="bold")
        style_ax(ax)

    fig.suptitle("Movement stereotypy across learning  (n = 4 mice,  mean ± 95 % CI)",
                 fontsize=9, y=1.02)
    plt.tight_layout()
    out = outdir / "kin_figB_stereotypy"
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: kin_figB_stereotypy")


# ─────────────────────────────────────────────────────────────────────────────
# Figure C — Inter-mouse comparison: individual signatures
# ─────────────────────────────────────────────────────────────────────────────

def plot_figure_C_intermouse(
    sessions_registry: list,
    expert_days:       dict,
    outdir:            Path,
    n_boot:            int = 1000,
) -> None:
    """
    Figure C: expert-day median trajectories of all 4 mice on the same axes.

    One panel per publication keypoint.
    Inset 4×4 Pearson r matrix shows cross-mouse trajectory similarity.
    Biological story: each mouse has its own stereotyped signature; the
    cross-mouse r is low, confirming individuality.
    """
    training_sessions = {
        (rec["mouse"], rec["day_index"]): rec
        for rec in sessions_registry
        if rec["phase"] == "training"
    }

    # Load expert-day traces for each mouse
    mouse_medians: dict[str, dict] = {}   # {mouse: {kp: median_trace}}
    t_ref = None

    for mouse in MICE:
        expert_day = expert_days.get(mouse)
        if expert_day is None:
            continue
        rec = training_sessions.get((mouse, expert_day))
        if rec is None:
            continue

        try:
            d    = load_session_pickle(rec["pickle_path"])
            tt   = extract_trial_table(d)
            sess = extract_hit_aligned_traces_session(d, tt)
            del d; gc.collect()
        except Exception as e:
            warnings.warn(f"Figure C {mouse}: {e}")
            try:
                del d
            except NameError:
                pass
            gc.collect()
            continue

        t_ref = sess["t_relative"]
        kp_medians = {}
        for kp in PUB_KEYPOINTS:
            traces = sess[kp]
            if traces.shape[0] >= 3 and not np.all(np.isnan(traces)):
                kp_medians[kp] = np.nanmedian(traces, axis=0)
        if kp_medians:
            mouse_medians[mouse] = kp_medians
        print(f"  Figure C: loaded {mouse} expert day {expert_day} "
              f"({sess['n_hit_trials']} hits)")

    if not mouse_medians or t_ref is None:
        print("  Figure C: no data available, skipping")
        return

    kp_list = list(PUB_KEYPOINTS.keys())
    n_kp    = len(kp_list)
    mice_ok = [m for m in MICE if m in mouse_medians]

    fig, axes = plt.subplots(1, n_kp,
                              figsize=(3.8 * n_kp, 4.5),
                              sharey=False)
    if n_kp == 1:
        axes = [axes]

    for ax, kp in zip(axes, kp_list):
        # Overlay median trace per mouse
        for mouse in mice_ok:
            med = mouse_medians[mouse].get(kp)
            if med is None:
                continue
            color  = MOUSE_COLORS.get(mouse, "gray")
            marker = MOUSE_MARKERS.get(mouse, "o")
            label  = mouse.split("_(")[0].split("_")[0]
            ax.plot(t_ref, med, color=color, lw=2.2, zorder=3, label=label)

        ax.axvline(0, color="gray", lw=0.8, ls="--", alpha=0.6, zorder=1)
        ax.set_title(PUB_KEYPOINT_LABELS.get(kp, kp),
                     fontsize=9, fontweight="bold", pad=4)
        ax.set_xlabel("Time from hit (s)", fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel("Baseline-subtracted signal", fontsize=8)
            ax.legend(frameon=False, fontsize=7, loc="upper left")
        style_ax(ax)

        # ── 4×4 Pearson r inset ──────────────────────────────────────────────
        mice_r = [m for m in mice_ok
                  if kp in mouse_medians.get(m, {})]
        if len(mice_r) >= 2:
            n_m   = len(mice_r)
            r_mat = np.full((n_m, n_m), np.nan)
            for ii, mi in enumerate(mice_r):
                for jj, mj in enumerate(mice_r):
                    if ii == jj:
                        r_mat[ii, jj] = 1.0
                    else:
                        a, b  = mouse_medians[mi][kp], mouse_medians[mj][kp]
                        valid = ~(np.isnan(a) | np.isnan(b))
                        if valid.sum() >= 5:
                            r_mat[ii, jj] = float(
                                np.corrcoef(a[valid], b[valid])[0, 1])

            inset = ax.inset_axes([0.62, 0.60, 0.36, 0.36])
            im    = inset.imshow(r_mat, vmin=-1, vmax=1,
                                 cmap="RdBu_r", aspect="auto",
                                 interpolation="nearest")
            short = [m.split("_(")[0][-5:] for m in mice_r]
            inset.set_xticks(range(n_m))
            inset.set_yticks(range(n_m))
            inset.set_xticklabels(short, fontsize=4.5, rotation=45, ha="right")
            inset.set_yticklabels(short, fontsize=4.5)
            inset.set_title("cross-r", fontsize=5, pad=1)
            for ii in range(n_m):
                for jj in range(n_m):
                    v = r_mat[ii, jj]
                    if not np.isnan(v):
                        txt_col = "white" if abs(v) > 0.5 else "black"
                        inset.text(jj, ii, f"{v:.2f}",
                                   ha="center", va="center",
                                   fontsize=4, color=txt_col)

    fig.suptitle("Expert-day individual movement signatures (one line per mouse)",
                 fontsize=10, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = outdir / "kin_figC_intermouse"
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: kin_figC_intermouse")


# ─────────────────────────────────────────────────────────────────────────────
# Figure D — Stereotypy × performance correlation
# ─────────────────────────────────────────────────────────────────────────────

def plot_figure_D_correlation(
    learning_df: pd.DataFrame,
    outdir:      Path,
    results_dir: Path,
) -> None:
    """
    Figure D: scatter of stereotypy index vs session LearnScore.

    Each point = one (mouse × training day).
    Colour = mouse.  Pearson r and p annotated per panel.
    CSV of correlations saved to results_dir / stereotypy_vs_performance.csv.
    """
    from scipy.stats import pearsonr

    kp_list   = list(PUB_KEYPOINTS.keys())
    valid_kps = [kp for kp in kp_list
                 if f"si_{kp}" in learning_df.columns
                 and "learn_score" in learning_df.columns]
    if not valid_kps:
        print("  Figure D: no valid columns, skipping")
        return

    n   = len(valid_kps)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 4.0), sharey=False)
    if n == 1:
        axes = [axes]

    csv_rows = []

    for ax, kp in zip(axes, valid_kps):
        col = f"si_{kp}"
        sub = learning_df[["mouse", "day_index", col,
                            "learn_score"]].dropna()

        for mouse in MICE:
            msub = sub[sub["mouse"] == mouse]
            if msub.empty:
                continue
            color  = MOUSE_COLORS.get(mouse, "gray")
            marker = MOUSE_MARKERS.get(mouse, "o")
            ax.scatter(msub[col], msub["learn_score"],
                       color=color, marker=marker, s=50,
                       alpha=0.82, edgecolors="white",
                       linewidths=0.5, zorder=3,
                       label=mouse.split("_(")[0].split("_")[0])

        # Overall Pearson r + regression line
        if len(sub) >= 5:
            r, p = pearsonr(sub[col].values, sub["learn_score"].values)
            sig   = ("***" if p < 0.001 else "**" if p < 0.01
                     else "*" if p < 0.05 else "n.s.")
            clr   = "crimson" if p < 0.05 else "dimgray"
            ax.text(0.05, 0.97, f"r = {r:.2f}  {sig}",
                    transform=ax.transAxes, fontsize=8,
                    va="top", color=clr, fontweight="bold")
            x_fit = np.linspace(sub[col].min(), sub[col].max(), 60)
            coeffs = np.polyfit(sub[col].values, sub["learn_score"].values, 1)
            ax.plot(x_fit, np.polyval(coeffs, x_fit),
                    color="black", lw=1.3, ls="--", alpha=0.55, zorder=2)
            csv_rows.append({
                "keypoint": kp,
                "label":    PUB_KEYPOINT_LABELS.get(kp, kp),
                "pearson_r": round(r, 4),
                "p_value":   round(p, 4),
                "n_points":  len(sub),
            })

        ax.set_xlabel("Stereotypy index", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("LearnScore", fontsize=9)
            ax.legend(frameon=False, fontsize=7, loc="lower right")
        ax.set_title(PUB_KEYPOINT_LABELS.get(kp, kp),
                     fontsize=9, fontweight="bold", pad=4)
        style_ax(ax)

    fig.suptitle("Stereotypy ↔ Performance correlation (each point = mouse × day)",
                 fontsize=9, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = outdir / "kin_figD_performance_corr"
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: kin_figD_performance_corr")

    if csv_rows:
        csv_path = results_dir / "stereotypy_vs_performance.csv"
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point — Figures A – D
# ─────────────────────────────────────────────────────────────────────────────

def run_new_figures() -> None:
    """
    Orchestrate all Section 4.2 kinematics figures (A through D).

    Data flow
    ---------
    1. compute_learning_curves_v2() — loads every training pickle once,
       extracts hit-aligned traces for PUB_KEYPOINTS, computes SI + LearnScore.
    2. find_expert_day_per_mouse()  — identifies best day per mouse.
    3. plot_figure_A_per_mouse()    — individual-trial overlays.
    4. plot_figure_B_stereotypy()   — stereotypy learning curve.
    5. plot_figure_C_intermouse()   — expert-day signature comparison.
    6. plot_figure_D_correlation()  — stereotypy vs performance scatter.
    """
    print("\n" + "=" * 65)
    print("POSE — Section 4.2: Hit-aligned kinematics (Figures A–D)")
    print("=" * 65)

    sessions = build_session_registry(exist_only=True)

    # ── Pupil coverage check (Section 4.2 figures use PUB_KEYPOINTS only —
    #    no pupil — so this is informational only, no feature pruning needed)
    from pose_config import check_pupil_coverage
    if not check_pupil_coverage(sessions):
        print("  [INFO] Facemap pupil absent/sparse — not included in "
              "Figures A–D (PUB_KEYPOINTS contains no pupil keypoints).")

    print("\nComputing hit-aligned learning curves (v2) ...")
    df, t_rel = compute_learning_curves_v2(sessions)
    csv_out   = RESDIR / "learning_curves_v2.csv"
    df.to_csv(csv_out, index=False)
    print(f"  Saved: {csv_out}")

    expert_days = find_expert_day_per_mouse(df)

    print("\n[Figure A] Hit-aligned trajectories: novice vs expert ...")
    plot_figure_A_per_mouse(sessions, expert_days, OUTDIR)

    print("\n[Figure B] Stereotypy index across training ...")
    plot_figure_B_stereotypy(df, OUTDIR)

    print("\n[Figure C] Inter-mouse expert-day signatures ...")
    plot_figure_C_intermouse(sessions, expert_days, OUTDIR)

    print("\n[Figure D] Stereotypy vs LearnScore correlation ...")
    plot_figure_D_correlation(df, OUTDIR, RESDIR)

    print("\n" + "=" * 65)
    print("Figures A–D complete.")
    print(f"  Figures: {OUTDIR}")
    print(f"  Results: {RESDIR}")
    print("=" * 65)


if __name__ == "__main__":
    run_learning_analysis()

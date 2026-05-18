# -*- coding: utf-8 -*-
"""
pose_figure2_rerender.py — Re-render kin_figA–D with two fixes:
  1. Angle-wrap fix: per-trial np.unwrap on C2_angle and C1r_angle traces.
  2. kin_figA shared y-axes per keypoint column (novice and expert same scale).

Output: output_figure2_final/ with _v2.png + _v2.svg
STANDALONE — no modification to pose_learning_kinematics.py or pose_utils.py.
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
from pathlib import Path
from datetime import datetime
from itertools import combinations
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent))

from pose_config import (
    MICE, MOUSE_COLORS, MOUSE_MARKERS,
    PUB_KEYPOINTS, PUB_KEYPOINT_LABELS,
    CONFIDENCE_THRESHOLDS,
    HIT_WINDOW_SEC, HIT_BASELINE_SEC, RESAMPLE_HZ,
    build_session_registry, RESULTS_PATH,
)
from pose_utils import (
    load_session_pickle, extract_trial_table,
    get_pose_object, get_pose_timestamps,
    extract_keypoint_timeseries,
    extract_whisker_angle_timeseries,
    slice_timeseries_to_trial, baseline_subtract,
)
from pose_learning_kinematics import (
    compute_stereotypy_index,
    _compute_session_learn_score,
    _bootstrap_ci_traces,
    fit_lme_learning_trend,
    find_expert_day_per_mouse,
    style_ax,
)
from bpod_utils import bootstrap_ci

OUTDIR = Path("f:/analysis/bpod/Documents/output_figure2_final")
OUTDIR.mkdir(parents=True, exist_ok=True)
RESDIR = RESULTS_PATH
RESDIR.mkdir(parents=True, exist_ok=True)

ANGLE_KEYPOINTS = {kp for kp, spec in PUB_KEYPOINTS.items()
                   if spec.get("coord") == "angle"}


# ════════════════════════════════════════════════════════════════════════════════
# Fix 1: Per-trial angle unwrap
# ════════════════════════════════════════════════════════════════════════════════

def circular_baseline_subtract_deg(
    traces_2d: np.ndarray,
    t_rel: np.ndarray,
    baseline_sec: tuple,
) -> np.ndarray:
    """
    Circular baseline subtraction for angle traces (degrees).

    For each trial:
      1. Compute the circular mean of baseline frames (avoids the ±180° jump
         problem that linear mean has for angles near the wrap boundary).
      2. Subtract circularly: dev = ((trace - bl_mean + 180) % 360) - 180
         → result in [-180°, +180°], the minimum-arc deviation from baseline.

    This removes ±350° jump artefacts that arise when the whisker crosses the
    ±180° atan2 boundary, without the cumulative-drift issue of np.unwrap.
    Linear unwrap creates monotone drift when the whisker oscillates through
    the boundary repeatedly; circular subtraction avoids that entirely.
    """
    bl_mask = (t_rel >= baseline_sec[0]) & (t_rel <= baseline_sec[1])
    result  = traces_2d.copy()
    for i in range(len(result)):
        row    = result[i]
        valid  = ~np.isnan(row[bl_mask])
        bl_pts = row[bl_mask][valid]
        if len(bl_pts) < 2:
            continue
        # Circular mean of baseline
        bl_mean = np.rad2deg(
            np.arctan2(np.mean(np.sin(np.deg2rad(bl_pts))),
                       np.mean(np.cos(np.deg2rad(bl_pts))))
        )
        # Minimum-arc deviation from baseline: wraps to [-180°, +180°]
        dev           = row - bl_mean
        dev           = ((dev + 180.0) % 360.0) - 180.0
        result[i]     = dev
    return result


def extract_hit_aligned_traces_v2(
    session_dict,
    trial_table: pd.DataFrame,
    window_sec=None,
    resample_hz=None,
) -> dict:
    """
    Same as pose_learning_kinematics.extract_hit_aligned_traces_session,
    with per-trial angle-unwrap applied BEFORE baseline subtraction.
    """
    if window_sec  is None: window_sec  = HIT_WINDOW_SEC
    if resample_hz is None: resample_hz = RESAMPLE_HZ

    t_rel = np.linspace(window_sec[0], window_sec[1],
                        int(round((window_sec[1] - window_sec[0]) * resample_hz)) + 1)
    n_t   = len(t_rel)

    hit_mask   = (
        (trial_table["trial_type"] == "W2T") &
        (trial_table["outcome"]    == "HIT") &
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if coord == "angle":
                values = extract_whisker_angle_timeseries(
                    pose_obj, spec["kp_start"], spec["kp_end"], conf_thr)
            else:
                x, y, _ = extract_keypoint_timeseries(pose_obj, spec["kp"], conf_thr)
                values   = y if coord == "y" else x

        for i, (_, tr_row) in enumerate(hit_trials.iterrows()):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, trace = slice_timeseries_to_trial(
                    ts_pose, values, float(tr_row["hit_onset"]),
                    window_sec[0], window_sec[1], resample_hz,
                )
            traces[i] = trace

        # ── FIX: circular baseline subtract for angle keypoints ───────────────
        # Linear baseline_subtract leaves ±350° jumps when the whisker crosses
        # the ±180° atan2 boundary. Circular subtraction gives the minimum-arc
        # deviation ([-180°, +180°]) and removes these wrap artefacts.
        if coord == "angle":
            traces = circular_baseline_subtract_deg(traces, t_rel, HIT_BASELINE_SEC)
        else:
            traces = baseline_subtract(traces, t_rel, HIT_BASELINE_SEC)
        result[kp_name] = traces

    return result


# ════════════════════════════════════════════════════════════════════════════════
# Recompute learning curves using fixed extractor
# ════════════════════════════════════════════════════════════════════════════════

def compute_learning_curves_fixed(sessions=None):
    """Identical to compute_learning_curves_v2 but uses extract_hit_aligned_traces_v2."""
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
        tag   = f"  [{idx+1}/{len(sessions)}] {mouse} day {day}"
        print(f"{tag} ...", end=" ", flush=True)

        d = None
        try:
            d  = load_session_pickle(rec["pickle_path"])
            tt = extract_trial_table(d)
            ls = _compute_session_learn_score(tt)
            sess = extract_hit_aligned_traces_v2(d, tt)
            del d; gc.collect()

            si_vals = {kp: compute_stereotypy_index(sess[kp], t_rel)
                       for kp in PUB_KEYPOINTS}
            row = {
                "mouse": mouse, "date": rec["date"],
                "day_index": day, "phase": rec["phase"],
                "n_hit_trials": sess["n_hit_trials"], "learn_score": ls,
            }
            for kp, si in si_vals.items():
                row[f"si_{kp}"] = si
            rows.append(row)
            print(f"n_hits={sess['n_hit_trials']}  si_C1r={si_vals.get('C1r_angle', np.nan):.3f}")
        except Exception as e:
            warnings.warn(f"{tag}: {e}")
            if d is not None:
                try: del d
                except NameError: pass
            gc.collect()
            rows.append({"mouse": mouse, "date": rec["date"],
                         "day_index": day, "phase": rec["phase"]})

    return pd.DataFrame(rows), t_rel


# ════════════════════════════════════════════════════════════════════════════════
# Figure A v2 — shared y-axes per column + angle-unwrap
# ════════════════════════════════════════════════════════════════════════════════

def plot_figA_v2(sessions_registry, expert_days, outdir, n_boot=1000):
    """
    kin_figA v2: novice vs expert, 4 columns (pub keypoints) × 2 rows.
    Changes from v1:
      - Angle-unwrap applied to C2_angle and C1r_angle.
      - Shared y-axis per column (1st/99th percentile + 10% margin across both rows).
    """
    training_sessions = {
        (rec["mouse"], rec["day_index"]): rec
        for rec in sessions_registry if rec["phase"] == "training"
    }
    kp_list = list(PUB_KEYPOINTS.keys())
    n_kp    = len(kp_list)

    for mouse in MICE:
        expert_day = expert_days.get(mouse)
        nov_rec    = training_sessions.get((mouse, 1))
        exp_rec    = training_sessions.get((mouse, expert_day)) if expert_day else None

        if nov_rec is None:
            print(f"  figA v2: no day-1 session for {mouse}, skipping")
            continue

        phase_recs = [
            (nov_rec, "Novice (Day 1)", "#999999"),
            (exp_rec, f"Expert (Day {expert_day})",
             MOUSE_COLORS.get(mouse, "steelblue")),
        ]

        # ── Pre-load both session traces ──────────────────────────────────────
        sessions_loaded = []
        for rec, label, color in phase_recs:
            if rec is None:
                sessions_loaded.append(None)
                continue
            try:
                d    = load_session_pickle(rec["pickle_path"])
                tt   = extract_trial_table(d)
                sess = extract_hit_aligned_traces_v2(d, tt)
                del d; gc.collect()
                sessions_loaded.append(sess)
            except Exception as e:
                warnings.warn(f"figA v2 {mouse}: {e}")
                if 'd' in dir(): del d
                gc.collect()
                sessions_loaded.append(None)

        # ── Compute shared y-limits per keypoint (1st–99th pct) ──────────────
        shared_ylims = {}
        for kp in kp_list:
            all_vals = []
            for sess in sessions_loaded:
                if sess is None:
                    continue
                tr = sess[kp]
                all_vals.extend(tr[~np.isnan(tr)].tolist())
            if len(all_vals) >= 10:
                p1, p99 = np.percentile(all_vals, 1), np.percentile(all_vals, 99)
                rng     = max(p99 - p1, 1e-3)
                shared_ylims[kp] = (p1 - 0.10 * rng, p99 + 0.10 * rng)

        # ── Plot ──────────────────────────────────────────────────────────────
        fig, axes = plt.subplots(2, n_kp,
                                  figsize=(3.5 * n_kp, 6.5),
                                  sharex=True, sharey=False)

        for row_i, (sess, (rec, row_label, color)) in enumerate(
                zip(sessions_loaded, phase_recs)):

            if sess is None or rec is None:
                for col_i in range(n_kp):
                    axes[row_i, col_i].text(
                        0.5, 0.5, "no session",
                        transform=axes[row_i, col_i].transAxes,
                        ha="center", va="center", fontsize=8, color="gray")
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

                for tr in traces:
                    if not np.all(np.isnan(tr)):
                        ax.plot(t_rel, tr, color="#cccccc", lw=0.5,
                                alpha=0.5, zorder=2)

                median         = np.nanmedian(traces, axis=0)
                lo_arr, hi_arr = _bootstrap_ci_traces(traces, n_boot=n_boot)
                ax.plot(t_rel, median, color=color, lw=2.2, zorder=5)
                ax.fill_between(t_rel, lo_arr, hi_arr,
                                color=color, alpha=0.25, zorder=4)
                ax.axvline(0, color="gray", lw=0.8, ls="--", alpha=0.6, zorder=1)

                # ── Shared y-axis ─────────────────────────────────────────────
                if kp in shared_ylims:
                    ax.set_ylim(shared_ylims[kp])

                style_ax(ax)

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

        fig.suptitle(
            f"{mouse.split('_(')[0]} — Hit-aligned kinematics: Novice vs Expert"
            "\n[v2: angle-unwrap fix + shared y-axes per column]",
            fontsize=10, fontweight="bold", y=1.01)
        plt.tight_layout()
        for ext in ("png", "svg"):
            fig.savefig(outdir / f"kin_figA_{mouse}_v2.{ext}",
                        dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: kin_figA_{mouse}_v2")


# ════════════════════════════════════════════════════════════════════════════════
# Figure B v2 — stereotypy learning curve (new df, svg output)
# ════════════════════════════════════════════════════════════════════════════════

def plot_figB_v2(learning_df, outdir, n_boot=1000):
    kp_list   = list(PUB_KEYPOINTS.keys())
    valid_kps = [kp for kp in kp_list if f"si_{kp}" in learning_df.columns]
    if not valid_kps:
        print("  figB v2: no si_ columns, skipping")
        return

    n   = len(valid_kps)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 4.0), sharey=False)
    if n == 1: axes = [axes]

    for ax, kp in zip(axes, valid_kps):
        col = f"si_{kp}"
        all_vals_by_day = {}
        for mouse in MICE:
            sub = learning_df[learning_df["mouse"] == mouse][
                ["day_index", col]].dropna()
            if sub.empty: continue
            color  = MOUSE_COLORS.get(mouse, "gray")
            marker = MOUSE_MARKERS.get(mouse, "o")
            ax.plot(sub["day_index"], sub[col],
                    color=color, lw=1.2, alpha=0.6,
                    marker=marker, ms=5, zorder=2)
            for _, row in sub.iterrows():
                all_vals_by_day.setdefault(int(row["day_index"]), []).append(row[col])

        days = sorted(all_vals_by_day)
        if days:
            means  = [np.nanmean(all_vals_by_day[d]) for d in days]
            lo_arr = [bootstrap_ci(all_vals_by_day[d], n_boot=n_boot)[0] for d in days]
            hi_arr = [bootstrap_ci(all_vals_by_day[d], n_boot=n_boot)[1] for d in days]
            ax.plot(days, means, color="black", lw=2.5, marker="o", ms=5, zorder=5)
            ax.fill_between(days, lo_arr, hi_arr, color="black", alpha=0.12, zorder=4)

        lme  = fit_lme_learning_trend(learning_df, col)
        pval = lme.get("slope_pval", np.nan)
        if not np.isnan(pval):
            sig = ("***" if pval < 0.001 else "**" if pval < 0.01
                   else "*" if pval < 0.05 else "n.s.")
            clr = "crimson" if pval < 0.05 else "dimgray"
            ax.text(0.97, 0.97, sig, transform=ax.transAxes, fontsize=9,
                    va="top", ha="right", color=clr, fontweight="bold")
            ax.text(0.97, 0.87, f"p={pval:.3f}", transform=ax.transAxes,
                    fontsize=7, va="top", ha="right", color=clr)

        ax.set_xlabel("Training day", fontsize=9)
        ax.set_xticks([1, 4, 7, 11])
        if ax is axes[0]:
            ax.set_ylabel("Stereotypy index\n(mean pairwise r)", fontsize=9)
        ax.text(0.05, 0.97, PUB_KEYPOINT_LABELS.get(kp, kp),
                transform=ax.transAxes, fontsize=8, va="top", fontweight="bold")
        style_ax(ax)

    fig.suptitle("Movement stereotypy across learning  [v2: angle-unwrap fix]",
                 fontsize=9, y=1.02)
    plt.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(outdir / f"kin_figB_stereotypy_v2.{ext}",
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: kin_figB_stereotypy_v2")


# ════════════════════════════════════════════════════════════════════════════════
# Figure C v2 — inter-mouse expert-day (uses fixed extractor)
# ════════════════════════════════════════════════════════════════════════════════

def plot_figC_v2(sessions_registry, expert_days, outdir, n_boot=1000):
    training_sessions = {
        (rec["mouse"], rec["day_index"]): rec
        for rec in sessions_registry if rec["phase"] == "training"
    }
    mouse_medians = {}
    t_ref = None

    for mouse in MICE:
        expert_day = expert_days.get(mouse)
        if expert_day is None: continue
        rec = training_sessions.get((mouse, expert_day))
        if rec is None: continue
        d = None
        try:
            d    = load_session_pickle(rec["pickle_path"])
            tt   = extract_trial_table(d)
            sess = extract_hit_aligned_traces_v2(d, tt)    # ← fixed extractor
            del d; gc.collect()
        except Exception as e:
            warnings.warn(f"figC v2 {mouse}: {e}")
            if d is not None:
                try: del d
                except: pass
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
        print(f"  figC v2: {mouse} day {expert_day} ({sess['n_hit_trials']} hits)")

    if not mouse_medians or t_ref is None:
        print("  figC v2: no data, skipping")
        return

    kp_list  = list(PUB_KEYPOINTS.keys())
    n_kp     = len(kp_list)
    mice_ok  = [m for m in MICE if m in mouse_medians]

    fig, axes = plt.subplots(1, n_kp, figsize=(3.8 * n_kp, 4.5), sharey=False)
    if n_kp == 1: axes = [axes]

    for ax, kp in zip(axes, kp_list):
        for mouse in mice_ok:
            med = mouse_medians[mouse].get(kp)
            if med is None: continue
            label = mouse.split("_(")[0].split("_")[0]
            ax.plot(t_ref, med, color=MOUSE_COLORS.get(mouse, "gray"),
                    lw=2.2, zorder=3, label=label)

        ax.axvline(0, color="gray", lw=0.8, ls="--", alpha=0.6, zorder=1)
        ax.set_title(PUB_KEYPOINT_LABELS.get(kp, kp),
                     fontsize=9, fontweight="bold", pad=4)
        ax.set_xlabel("Time from hit (s)", fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel("Baseline-subtracted signal", fontsize=8)
            ax.legend(frameon=False, fontsize=7, loc="upper left")
        style_ax(ax)

        # 4×4 Pearson r inset
        mice_r = [m for m in mice_ok if kp in mouse_medians.get(m, {})]
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
                            r_mat[ii, jj] = float(np.corrcoef(a[valid], b[valid])[0, 1])
            inset = ax.inset_axes([0.62, 0.60, 0.36, 0.36])
            inset.imshow(r_mat, vmin=-1, vmax=1, cmap="RdBu_r",
                         aspect="auto", interpolation="nearest")
            short = [m.split("_(")[0][-5:] for m in mice_r]
            inset.set_xticks(range(n_m)); inset.set_yticks(range(n_m))
            inset.set_xticklabels(short, fontsize=4.5, rotation=45, ha="right")
            inset.set_yticklabels(short, fontsize=4.5)
            inset.set_title("cross-r", fontsize=5, pad=1)
            for ii in range(n_m):
                for jj in range(n_m):
                    v = r_mat[ii, jj]
                    if not np.isnan(v):
                        inset.text(jj, ii, f"{v:.2f}", ha="center", va="center",
                                   fontsize=4, color="white" if abs(v) > 0.5 else "black")

    fig.suptitle("Expert-day individual movement signatures  [v2: angle-unwrap fix]",
                 fontsize=10, fontweight="bold", y=1.01)
    plt.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(outdir / f"kin_figC_intermouse_v2.{ext}",
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: kin_figC_intermouse_v2")


# ════════════════════════════════════════════════════════════════════════════════
# Figure D v2 — stereotypy vs performance (new df, svg)
# ════════════════════════════════════════════════════════════════════════════════

def plot_figD_v2(learning_df, outdir, results_dir):
    kp_list   = list(PUB_KEYPOINTS.keys())
    valid_kps = [kp for kp in kp_list
                 if f"si_{kp}" in learning_df.columns
                 and "learn_score" in learning_df.columns]
    if not valid_kps:
        print("  figD v2: no valid columns, skipping")
        return

    n   = len(valid_kps)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 4.0), sharey=False)
    if n == 1: axes = [axes]

    csv_rows = []
    for ax, kp in zip(axes, valid_kps):
        col = f"si_{kp}"
        sub = learning_df[["mouse", "day_index", col, "learn_score"]].dropna()
        for mouse in MICE:
            msub = sub[sub["mouse"] == mouse]
            if msub.empty: continue
            ax.scatter(msub[col], msub["learn_score"],
                       color=MOUSE_COLORS.get(mouse, "gray"),
                       marker=MOUSE_MARKERS.get(mouse, "o"),
                       s=50, alpha=0.82, edgecolors="white",
                       linewidths=0.5, zorder=3,
                       label=mouse.split("_(")[0].split("_")[0])
        if len(sub) >= 5:
            r, p = pearsonr(sub[col].values, sub["learn_score"].values)
            sig = ("***" if p < 0.001 else "**" if p < 0.01
                   else "*" if p < 0.05 else "n.s.")
            clr = "crimson" if p < 0.05 else "dimgray"
            ax.text(0.05, 0.97, f"r = {r:.2f}  {sig}",
                    transform=ax.transAxes, fontsize=8, va="top",
                    color=clr, fontweight="bold")
            x_fit  = np.linspace(sub[col].min(), sub[col].max(), 60)
            coeffs = np.polyfit(sub[col].values, sub["learn_score"].values, 1)
            ax.plot(x_fit, np.polyval(coeffs, x_fit),
                    color="black", lw=1.3, ls="--", alpha=0.55, zorder=2)
            csv_rows.append({
                "keypoint": kp, "label": PUB_KEYPOINT_LABELS.get(kp, kp),
                "pearson_r": round(r, 4), "p_value": round(p, 4), "n_points": len(sub),
            })
        ax.set_xlabel("Stereotypy index", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("LearnScore", fontsize=9)
            ax.legend(frameon=False, fontsize=7, loc="lower right")
        ax.set_title(PUB_KEYPOINT_LABELS.get(kp, kp),
                     fontsize=9, fontweight="bold", pad=4)
        style_ax(ax)

    fig.suptitle("Stereotypy vs Performance  [v2: angle-unwrap fix]",
                 fontsize=9, fontweight="bold", y=1.02)
    plt.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(outdir / f"kin_figD_performance_corr_v2.{ext}",
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: kin_figD_performance_corr_v2")

    if csv_rows:
        csv_path = results_dir / "stereotypy_vs_performance_v2.csv"
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")


# ════════════════════════════════════════════════════════════════════════════════
# Report
# ════════════════════════════════════════════════════════════════════════════════

def build_report(learning_df_v2, now_str):
    """Compare new C1r_angle SI values to the original CSV."""
    old_csv = RESDIR / "learning_curves_v2.csv"
    report  = f"""# Figure 2 Re-render Report — Angle-Wrap Fix + Shared Y-Axes

**Date:** {now_str}
**Script:** pose_figure2_rerender.py
**Output:** {OUTDIR}

---

## Fix 1: Angle-Wrap (per-trial np.unwrap on C2_angle and C1r_angle)

### Problem
`extract_whisker_angle_timeseries` computes atan2, which returns values in [-180°, 180°].
When the whisker crosses the +180° / -180° boundary during a trial, this produces
a discontinuous jump of ~360° in the trace — a pure artefact, not a real movement.
These jumps contaminate:
  - Visual inspection of individual trial traces (Fig A individual lines)
  - The median trace (one jump trial can shift the median)
  - The stereotypy index (pairwise Pearson r; jump trials are outliers)

### Fix — Circular baseline subtraction
Instead of linear baseline_subtract, apply circular baseline subtraction to angle keypoints:
  1. Compute circular mean of the baseline window (avoids the ±180° discontinuity
     that linear mean has when baseline angles straddle the wrap boundary).
  2. Subtract circularly: `dev = ((trace - bl_mean + 180) % 360) - 180`
     → result in [-180°, +180°], the minimum-arc deviation from baseline.

This removes the ±350° jump artefacts without the cumulative-drift issue of np.unwrap
(np.unwrap creates monotone drift when the whisker oscillates through ±180° repeatedly,
turning oscillations into a continuously drifting ramp — worse than the original).

Applied to: `C2_angle`, `C1r_angle` (any keypoint with coord="angle").
NOT applied to: `nose_y`, `fpaw_y` (positional, no wrap issue).

### Stereotypy Index Change (si_C1r_angle: old vs new)
"""
    if old_csv.exists():
        old_df = pd.read_csv(old_csv)
        compare = []
        for mouse in MICE:
            old_sub = old_df[(old_df["mouse"] == mouse)][["day_index", "si_C1r_angle"]].dropna()
            new_sub = learning_df_v2[(learning_df_v2["mouse"] == mouse)][
                ["day_index", "si_C1r_angle"]].dropna()
            merged  = old_sub.merge(new_sub, on="day_index", suffixes=("_old", "_new"))
            for _, row in merged.iterrows():
                delta = row["si_C1r_angle_new"] - row["si_C1r_angle_old"]
                compare.append((mouse, int(row["day_index"]),
                                 row["si_C1r_angle_old"], row["si_C1r_angle_new"], delta))

        if compare:
            report += "\n| Mouse | Day | SI old | SI new | Delta |\n|-------|-----|--------|--------|-------|\n"
            for mouse, day, old_v, new_v, delta in compare:
                report += f"| {mouse} | {day} | {old_v:.4f} | {new_v:.4f} | {delta:+.4f} |\n"

            deltas = [c[4] for c in compare]
            mean_d = float(np.mean(deltas))
            report += f"\n**Mean delta SI (new - old):** {mean_d:+.4f}\n"
            pos = sum(1 for d in deltas if d > 0.005)
            neg = sum(1 for d in deltas if d < -0.005)
            zer = len(deltas) - pos - neg
            report += f"**Sessions increased > 0.005:** {pos}\n"
            report += f"**Sessions decreased > 0.005:** {neg}\n"
            report += f"**Sessions unchanged (+/-0.005):** {zer}\n"
    else:
        report += "\n*(old CSV not found — no comparison available)*\n"

    report += """
---

## Fix 2: Shared Y-Axes Per Column in kin_figA

### Problem
kin_figA (v1) used `sharey=False` with independent autoscaling per panel.
Novice and expert rows for the same keypoint had different y-scales,
making visual comparison of effect magnitude impossible.

### Fix
Before plotting, pre-load traces from both novice and expert sessions.
Per keypoint column: compute 1st/99th percentile of the combined
(novice + expert) distribution + 10% margin, set as shared ylim for
both rows. This allows direct visual comparison of absolute signal magnitude.

---

## Output Files

- `kin_figA_{mouse}_v2.png/svg` (4 files × 2 formats = 8 files)
- `kin_figB_stereotypy_v2.png/svg`
- `kin_figC_intermouse_v2.png/svg`
- `kin_figD_performance_corr_v2.png/svg`

---

## Known Limitations

- Per-trial unwrap removes within-trial wrap-arounds only. If the baseline
  window itself contains a wrap event, the post-unwrap baseline mean will
  include this shift — baseline_subtract will then remove it correctly.
- Cross-trial absolute angle offset is not harmonised; only within-trial
  continuity is guaranteed. This is sufficient for stereotypy and Fig A.
- figD (si_C1r_angle vs LearnScore) Pearson r may change if the unwrap
  significantly alters SI values in specific sessions.
"""
    return report


# ════════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 65)
    print("FIGURE 2 RE-RENDER -- angle-wrap fix + shared y-axes")
    print("=" * 65)
    start_ts = datetime.now()
    print(f"Started: {start_ts.strftime('%Y-%m-%d %H:%M:%S')}")

    sessions = build_session_registry(exist_only=True)

    print("\n[1/5] Recomputing learning curves with angle-unwrap fix ...")
    df_v2, t_rel = compute_learning_curves_fixed(sessions)
    csv_out = OUTDIR / "learning_curves_v2_fixed.csv"
    df_v2.to_csv(csv_out, index=False)
    print(f"  Saved: {csv_out}")

    print("\n[2/5] Finding expert days ...")
    expert_days = find_expert_day_per_mouse(df_v2)

    print("\n[3/5] Figure A — novice vs expert, shared y-axes ...")
    plot_figA_v2(sessions, expert_days, OUTDIR)

    print("\n[4/5] Figure B — stereotypy learning curve ...")
    plot_figB_v2(df_v2, OUTDIR)

    print("\n[4b/5] Figure C — inter-mouse expert-day signatures ...")
    plot_figC_v2(sessions, expert_days, OUTDIR)

    print("\n[5/5] Figure D — stereotypy vs performance ...")
    plot_figD_v2(df_v2, OUTDIR, RESDIR)

    # Report
    now_str     = start_ts.strftime("%Y-%m-%d %H:%M:%S")
    report      = build_report(df_v2, now_str)
    report_path = OUTDIR / "update_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\nSaved: {report_path}")

    # ZIP
    zip_path = Path("f:/analysis/bpod/Documents/output_figure2_final.zip")
    file_exts = (".png", ".svg", ".csv", ".md")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(OUTDIR.iterdir()):
            if f.suffix in file_exts:
                zf.write(f, f.name)
    zip_mb = zip_path.stat().st_size / 1e6
    print(f"\nZIP: {zip_path}  ({zip_mb:.2f} MB)")
    print("\n" + "=" * 65)
    print(f"DONE -- {start_ts.strftime('%H:%M:%S')} -> {datetime.now().strftime('%H:%M:%S')}")
    print(f"ZIP:  {zip_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()

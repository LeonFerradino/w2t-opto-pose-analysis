# -*- coding: utf-8 -*-
"""
pose_example_traces.py — Definitive thesis Fig 1C-style example trace.

Session : MLA-026807_4 / 20250916_1 / Trial 30 (W2T HIT, RT=632ms)
Style   : Staab/Sehara 2025 — scale bars, no axis lines, despined
Channels: Contra whisker | C2 whisker | Nose y | Forepaw y

STANDALONE — no modification to existing scripts.
"""

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

sys.path.insert(0, str(Path(__file__).parent))

from pose_config import (
    RESAMPLE_HZ, CONFIDENCE_THRESHOLDS,
    find_bpod_mat_files,
)
from pose_utils import (
    load_session_pickle, extract_trial_table, load_trial_types_from_mat,
    get_pose_object, get_pose_timestamps,
    extract_keypoint_timeseries,
    extract_whisker_angle_timeseries,
    slice_timeseries_to_trial,
)

# ── Session / trial ───────────────────────────────────────────────────────────
MOUSE        = "MLA-026807_4"
DATE         = "20250916"
SUFFIX       = "1"
PICKLE       = Path("f:/analysis/bpod/Documents/processed"
                    "/MLA-026807_4/MLA-026807_4_20250916_1.pickle")
FINAL_TRIAL_N = 30          # 1-based trial number to plot
OUTDIR        = Path("f:/analysis/bpod/Documents/output_example_traces_final")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ── Window / baseline ─────────────────────────────────────────────────────────
WINDOW_SEC   = (-1.0, 2.0)
BASELINE_SEC = (-1.0, -0.5)

# ── Confidence thresholds ─────────────────────────────────────────────────────
CONF_THR_OVERRIDE = {"dlc": 0.8, "sleap": 0.6, "facemap": None}

# ── Channels (top → bottom) ───────────────────────────────────────────────────
CHANNELS = [
    {
        "name":      "C1r_angle",
        "label":     "Contra\nwhisker",
        "source":    "dlc", "vid": 0,
        "mode":      "angle",
        "kp_start":  "C2_start_reverse", "kp_end": "C2_end_reverse",
        "baseline":  True,
        "unwrap":    True,
        "outlier":   False,
        "color":     "#ff7f0e",
        "scalebar":  10,        # degrees
        "scalelabel": "10°",
    },
    {
        "name":      "C2_angle",
        "label":     "C2\nwhisker",
        "source":    "dlc", "vid": 0,
        "mode":      "angle",
        "kp_start":  "C2_start", "kp_end": "C2_end",
        "baseline":  True,
        "unwrap":    True,
        "outlier":   False,
        "color":     "#1f77b4",
        "scalebar":  5,
        "scalelabel": "5°",
    },
    {
        "name":      "nose_y",
        "label":     "Nose\ny-pos",
        "source":    "dlc", "vid": 0,
        "mode":      "keypoint_y", "kp": "nose",
        "baseline":  True,
        "unwrap":    False,
        "outlier":   False,
        "color":     "#2ca02c",
        "scalebar":  1,
        "scalelabel": "1 px",
    },
    {
        "name":      "fpaw_y",
        "label":     "Forepaw\ny-pos",
        "source":    "sleap", "vid": 0,
        "mode":      "keypoint_y", "kp": "frontpawL",
        "baseline":  True,
        "unwrap":    False,
        "outlier":   True,
        "color":     "#d62728",
        "scalebar":  10,
        "scalelabel": "10 px",
    },
]

FPAW_JUMP_THR   = 30.0
FPAW_INTERP_MAX = 5


# ════════════════════════════════════════════════════════════════════════════════
# Signal helpers
# ════════════════════════════════════════════════════════════════════════════════

def unwrap_angle_deg(trace_deg):
    result = trace_deg.copy()
    valid  = ~np.isnan(result)
    if valid.sum() < 2:
        return result
    x = np.arange(len(result))
    filled = result.copy()
    filled[~valid] = np.interp(x[~valid], x[valid], result[valid])
    unwrapped      = np.rad2deg(np.unwrap(np.deg2rad(filled)))
    result[valid]  = unwrapped[valid]
    result[~valid] = np.nan
    return result


def filter_jumps(trace, max_jump, max_interp_gap):
    result = trace.copy()
    diff   = np.abs(np.diff(result))
    bad    = np.where(diff > max_jump)[0] + 1
    result[bad] = np.nan
    nans  = np.isnan(result)
    if not nans.any() or not (~nans).any():
        return result
    x     = np.arange(len(result))
    valid = ~nans
    filled = result.copy()
    run_start = None
    for i in range(len(nans)):
        if nans[i] and run_start is None:
            run_start = i
        elif not nans[i] and run_start is not None:
            if (i - run_start) <= max_interp_gap:
                filled[run_start:i] = np.interp(
                    x[run_start:i], x[valid], result[valid])
            run_start = None
    return filled


# ════════════════════════════════════════════════════════════════════════════════
# Scale bar drawing
# ════════════════════════════════════════════════════════════════════════════════

def add_vertical_scalebar(ax, bar_size, label, color="black",
                          x_frac=0.90, y_frac=0.12, lw=2.0, fontsize=9):
    """
    Draw a vertical scale bar inside the axes.
    x_frac / y_frac: position of bar bottom in axes-fraction coordinates.
    bar_size: size in data units.
    """
    # Convert fractions to data units
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x_data = xmin + x_frac * (xmax - xmin)
    y_bot  = ymin + y_frac * (ymax - ymin)
    y_top  = y_bot + bar_size

    ax.plot([x_data, x_data], [y_bot, y_top],
            color=color, lw=lw, solid_capstyle="butt",
            clip_on=False, zorder=10)
    ax.text(x_data + 0.025 * (xmax - xmin),
            (y_bot + y_top) / 2,
            label, ha="left", va="center",
            fontsize=fontsize, color=color, zorder=10)


def add_horizontal_scalebar(ax, bar_sec, label, color="black",
                             x_frac=0.90, y_frac=0.12, lw=2.0, fontsize=9):
    """
    Draw a horizontal time scale bar (bar_sec in seconds).
    Anchored at right side; bottom of bar aligns with vertical bar.
    """
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x_right = xmin + x_frac * (xmax - xmin)
    x_left  = x_right - bar_sec
    y_data  = ymin + y_frac * (ymax - ymin)

    ax.plot([x_left, x_right], [y_data, y_data],
            color=color, lw=lw, solid_capstyle="butt",
            clip_on=False, zorder=10)
    ax.text((x_left + x_right) / 2,
            y_data - 0.06 * (ymax - ymin),
            label, ha="center", va="top",
            fontsize=fontsize, color=color, zorder=10)


# ════════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print(f"EXAMPLE TRACES FINAL -- Trial {FINAL_TRIAL_N}")
    print("=" * 60)

    # ── Load session ──────────────────────────────────────────────────────────
    d = load_session_pickle(PICKLE)
    mat_files       = find_bpod_mat_files(MOUSE, DATE, SUFFIX)
    trial_types_arr = load_trial_types_from_mat(mat_files) if mat_files else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tt = extract_trial_table(d, trial_types_array=trial_types_arr)
    print(f"Loaded {len(tt)} trials from {PICKLE.name}")

    # ── Select trial ──────────────────────────────────────────────────────────
    trial_row = tt[tt["trial_idx"] == (FINAL_TRIAL_N - 1)].iloc[0]
    audio_t   = float(trial_row["audio_onset"])
    hit_t     = float(trial_row["hit_onset"])      if not pd.isna(trial_row["hit_onset"])      else None
    lick_t    = float(trial_row["first_lick_time"]) if not pd.isna(trial_row["first_lick_time"]) else None
    rt_ms     = float(trial_row["reaction_time"]) * 1000 if not pd.isna(trial_row["reaction_time"]) else None
    lick_n    = int(trial_row["lick_count"])

    print(f"Trial {FINAL_TRIAL_N}: audio={audio_t:.4f}s  "
          f"hit={hit_t:.4f}s  lick={lick_t:.4f}s  "
          f"RT={rt_ms:.0f}ms  lick_n={lick_n}")

    # Merge hit and lick markers if they coincide (within 15 ms)
    hit_lick_same = (hit_t is not None and lick_t is not None
                     and abs(hit_t - lick_t) < 0.015)
    if hit_lick_same:
        # Use whichever is non-None; prefer lick_t as it matches RT
        event_t   = lick_t
        event_rel = lick_t - audio_t
        event_lbl = f"Hit / First lick ({event_rel:+.3f}s)"
        print(f"  -> Hit and lick overlap (<15ms); merged to single marker")
    else:
        event_t   = None   # handled separately below

    # ── Time grid ─────────────────────────────────────────────────────────────
    n_t   = int(round((WINDOW_SEC[1] - WINDOW_SEC[0]) * RESAMPLE_HZ)) + 1
    t_rel = np.linspace(WINDOW_SEC[0], WINDOW_SEC[1], n_t)
    bl_mask = (t_rel >= BASELINE_SEC[0]) & (t_rel <= BASELINE_SEC[1])

    # ── Extract + process traces ──────────────────────────────────────────────
    traces    = {}
    nan_pcts  = {}
    available = []

    for ch in CHANNELS:
        name     = ch["name"]
        source   = ch["source"]
        conf_thr = CONF_THR_OVERRIDE.get(source, CONFIDENCE_THRESHOLDS.get(source))
        pose_obj = get_pose_object(d, source, ch["vid"])
        if pose_obj is None:
            print(f"  [{name}] SKIP — pose object not found")
            traces[name] = None
            continue
        ts_pose = get_pose_timestamps(pose_obj)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if ch["mode"] == "angle":
                values = extract_whisker_angle_timeseries(
                    pose_obj, ch["kp_start"], ch["kp_end"], conf_thr)
            else:
                _, y, _ = extract_keypoint_timeseries(pose_obj, ch["kp"], conf_thr)
                values = y
            _, trace = slice_timeseries_to_trial(
                ts_pose, values, audio_t,
                WINDOW_SEC[0], WINDOW_SEC[1], RESAMPLE_HZ)

        if ch["unwrap"]:
            trace = unwrap_angle_deg(trace)
        if ch["outlier"]:
            trace = filter_jumps(trace, FPAW_JUMP_THR, FPAW_INTERP_MAX)

        frac_nan = float(np.isnan(trace).mean())
        nan_pcts[name] = frac_nan * 100

        if frac_nan > 0.50:
            print(f"  [{name}] SKIP — {frac_nan*100:.0f}% NaN")
            traces[name] = None
            continue

        if ch["baseline"] and bl_mask.any():
            bl_mean = np.nanmean(trace[bl_mask])
            if not np.isnan(bl_mean):
                trace = trace - bl_mean

        traces[name] = trace
        available.append(ch)
        print(f"  [{name}] ok  NaN={frac_nan*100:.1f}%")

    n_rows = len(available)
    if n_rows == 0:
        raise RuntimeError("No channels available.")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(7, 9))

    # Left margin large enough for channel labels, right margin for scale bars
    gs = fig.add_gridspec(
        n_rows, 1,
        hspace=0.05,
        left=0.22, right=0.85,
        top=0.91, bottom=0.06,
    )
    axes = [fig.add_subplot(gs[i]) for i in range(n_rows)]

    for i, (ax, ch) in enumerate(zip(axes, available)):
        trace = traces[ch["name"]]
        color = ch["color"]

        ax.plot(t_rel, trace, color=color, lw=1.5, zorder=3)

        # Baseline shading
        ax.axvspan(BASELINE_SEC[0], BASELINE_SEC[1],
                   alpha=0.08, color="gray", zorder=1)

        # Audio cue line
        ax.axvline(0, color="black", lw=1.4, ls="--", zorder=4, alpha=0.9)

        # Event marker(s)
        if hit_lick_same:
            ax.axvline(event_rel, color="#c0392b", lw=1.4,
                       ls="--", zorder=4, alpha=0.9)
        else:
            if hit_t is not None:
                ax.axvline(hit_t - audio_t, color="#c0392b", lw=1.3,
                           ls="--", zorder=4, alpha=0.85)
            if lick_t is not None:
                ax.axvline(lick_t - audio_t, color="#1565c0", lw=1.3,
                           ls="--", zorder=4, alpha=0.80)

        # Y limits: 5th–95th percentile + 15% margin
        valid = trace[~np.isnan(trace)]
        if len(valid) >= 5:
            p5, p95 = np.percentile(valid, 5), np.percentile(valid, 95)
            rng = max(p95 - p5, ch["scalebar"] * 1.5)   # at least 1.5× scalebar
            margin = 0.20 * rng
            ax.set_ylim(p5 - margin, p95 + margin)

        # Remove all spines and ticks
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

        # Zero reference line (thin)
        ax.axhline(0, color="gray", lw=0.6, ls=":", zorder=2, alpha=0.6)

        # Channel label — horizontal, left of axes, in axes fraction coordinates
        ax.text(-0.06, 0.50, ch["label"],
                transform=ax.transAxes,
                ha="right", va="center",
                fontsize=12, fontweight="bold",
                color=ch["color"])

        # Vertical scale bar (lower right, after ylim is set)
        add_vertical_scalebar(ax, ch["scalebar"], ch["scalelabel"],
                              x_frac=0.91, y_frac=0.10, lw=2.0, fontsize=10)

        # Horizontal time scale bar in bottom panel only
        if i == n_rows - 1:
            add_horizontal_scalebar(ax, 0.5, "500 ms",
                                    x_frac=0.91, y_frac=0.10, lw=2.0, fontsize=10)

    axes[0].set_xlim(WINDOW_SEC)   # propagates via sharex — but axes not shared; set all
    for ax in axes:
        ax.set_xlim(WINDOW_SEC)

    # ── Legend (top panel, upper-left area) ──────────────────────────────────
    leg_handles = [
        plt.Line2D([0], [0], color="black", lw=1.4, ls="--",
                   label="Audio cue (t = 0)"),
    ]
    if hit_lick_same:
        leg_handles.append(
            plt.Line2D([0], [0], color="#c0392b", lw=1.4, ls="--",
                       label=event_lbl)
        )
    else:
        if hit_t is not None:
            leg_handles.append(
                plt.Line2D([0], [0], color="#c0392b", lw=1.3, ls="--",
                           label=f"Hit onset ({hit_t - audio_t:+.3f}s)"))
        if lick_t is not None:
            leg_handles.append(
                plt.Line2D([0], [0], color="#1565c0", lw=1.3, ls="--",
                           label=f"First lick ({lick_t - audio_t:+.3f}s)"))

    axes[0].legend(handles=leg_handles, loc="upper left",
                   fontsize=10, framealpha=0.0, handlelength=2.0)

    # ── Title (mouse/session info removed — goes in caption) ─────────────────
    fig.text(0.50, 0.965,
             "Example W2T HIT trial",
             ha="center", va="top", fontsize=14, fontweight="bold")

    # ── Save ─────────────────────────────────────────────────────────────────
    png_path = OUTDIR / f"figure1_example_trial{FINAL_TRIAL_N}.png"
    svg_path = OUTDIR / f"figure1_example_trial{FINAL_TRIAL_N}.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {png_path}")
    print(f"Saved: {svg_path}")

    # ── README ────────────────────────────────────────────────────────────────
    rt_str  = f"{rt_ms:.0f} ms" if rt_ms is not None else "N/A"
    hit_str = f"{hit_t - audio_t:+.3f} s" if hit_t is not None else "N/A"

    readme = f"""# Figure 1 — Example Traces Metadata

## Thesis Caption Data

| Field | Value |
|-------|-------|
| Mouse | {MOUSE} |
| Session date | {DATE} |
| Session suffix | {SUFFIX} |
| Trial number (1-based) | {FINAL_TRIAL_N} |
| Trial type | W2T |
| Outcome | HIT |
| audio_onset | {audio_t:.4f} s |
| hit_onset (rel. audio) | {hit_str} |
| Reaction time | {rt_str} |
| Lick count (trial) | {lick_n} |

## Channels (top → bottom)

| Channel | Source | Signal | Processing |
|---------|--------|--------|------------|
| Contra whisker | DLC vid0 | atan2(C2_start_reverse→C2_end_reverse) | unwrap, baseline-sub (-1.0 to -0.5 s) |
| C2 whisker | DLC vid0 | atan2(C2_start→C2_end) | unwrap, baseline-sub |
| Nose y-pos | DLC vid0 | y-coordinate of "nose" keypoint | baseline-sub |
| Forepaw y-pos | SLEAP vid0 | y-coordinate of "frontpawL" | jump-filter (>30px), baseline-sub |

## Event Markers

- **Black dashed**: Audio cue onset (t = 0)
- **Red dashed**: Hit onset / First lick (t = {hit_str}) — merged because |Δt| < 15 ms

## Scale Bars

- Contra whisker: 10°
- C2 whisker: 5°
- Nose y: 1 px
- Forepaw y: 10 px
- Time: 500 ms

## Analysis Parameters

- Window: {WINDOW_SEC[0]} to {WINDOW_SEC[1]} s relative to audio_onset
- Baseline: {BASELINE_SEC[0]} to {BASELINE_SEC[1]} s
- DLC confidence threshold: {CONF_THR_OVERRIDE['dlc']}
- SLEAP score threshold: {CONF_THR_OVERRIDE['sleap']}
- Gray shading: baseline window ({BASELINE_SEC[0]} to {BASELINE_SEC[1]} s)
- Forepaw jump filter: |Δframe| > {FPAW_JUMP_THR} px → NaN; gaps ≤ {FPAW_INTERP_MAX} frames interpolated
- Y-axis: 5th–95th percentile of trace + 20% margin (scale bar context)

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Script:** pose_example_traces.py
"""

    readme_path = OUTDIR / "README_figure1_metadata.md"
    readme_path.write_text(readme, encoding="utf-8")
    print(f"Saved: {readme_path}")

    # ── ZIP ───────────────────────────────────────────────────────────────────
    zip_path = Path("f:/analysis/bpod/Documents/output_example_traces_final.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in [png_path, svg_path, readme_path]:
            if f.exists():
                zf.write(f, f.name)
    zip_mb = zip_path.stat().st_size / 1e6
    print(f"\nZIP: {zip_path}  ({zip_mb:.2f} MB)")
    print("\n" + "=" * 60)
    print(f"DONE -- figure1_example_trial{FINAL_TRIAL_N}.png/svg")
    print(f"ZIP:  {zip_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

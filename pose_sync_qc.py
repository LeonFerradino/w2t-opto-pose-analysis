# -*- coding: utf-8 -*-
"""
pose_sync_qc.py — Sync quality check for already-synchronised pickle files.

Background
----------
The w2t-bkin pipeline synchronises bpod and video timestamps using the DLC
``trial_light`` keypoint, which is an LED controlled by bpod that turns on
exactly when the Audio state starts (audio cue).  The pipeline uses these
rising edges to align bpod and camera clocks before creating the pickle files.
The timestamps in ``trials_table`` and ``pose_estimations[i]['timestamp']``
are therefore already in the same time base — no additional alignment is
needed in the analysis scripts.

What this script does
---------------------
Post-hoc sanity check: it verifies that the residuals between
``trial_light`` DLC edges (video side) and bpod Audio state onsets are
within the expected range (< 1 camera frame ≈ 7 ms at 150 fps).  Large
residuals would indicate a problem in the pipeline's sync step for that
session.

Outputs
-------
results/pose/sync_qc_results.csv  — one row per session
figures/pose/sync_qc/residual_distribution.pdf/.png
figures/pose/sync_qc/match_rate_per_session.pdf/.png

Usage
-----
    python pose_sync_qc.py

Run this before any analysis if you want to verify pipeline quality.
The script processes one pickle at a time (peak RAM ≈ 400 MB).
"""

import gc
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from pose_config import (
    MICE, FIGURES_PATH, RESULTS_PATH,
    SYNC_FLAG_MEAN_OFFSET_SEC, SYNC_FLAG_STD_SEC, SYNC_FLAG_MATCH_RATE,
    MOUSE_COLORS, build_session_registry,
)
from pose_utils import (
    load_session_pickle, compute_sync_alignment_error,
)

OUTDIR_FIG = FIGURES_PATH / "sync_qc"
OUTDIR_RES = RESULTS_PATH
OUTDIR_FIG.mkdir(parents=True, exist_ok=True)
OUTDIR_RES.mkdir(parents=True, exist_ok=True)

# Expected residual at 150 fps = 1/150 ≈ 6.7 ms
EXPECTED_RESIDUAL_MS = 1000.0 / 150.0


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def run_sync_qc_all_sessions(verbose: bool = True) -> pd.DataFrame:
    """
    Run the trial_light residual check for every session, one at a time.

    Saves CSV to RESULTS_PATH / sync_qc_results.csv.

    Parameters
    ----------
    verbose : bool
        Print per-session progress if True.

    Returns
    -------
    pd.DataFrame
        One row per session with sync quality metrics.
    """
    sessions = build_session_registry()
    results  = []

    for idx, rec in enumerate(sessions):
        path = rec["pickle_path"]
        tag  = f"[{idx+1}/{len(sessions)}] {rec['mouse']} {rec['date']}"
        if rec["suffix"]:
            tag += f"_{rec['suffix']}"

        if verbose:
            print(f"{tag} ...", end=" ", flush=True)

        try:
            d    = load_session_pickle(path)
            sync = compute_sync_alignment_error(d)
            del d; gc.collect()

            row = {
                "mouse":                rec["mouse"],
                "date":                 rec["date"],
                "suffix":               rec["suffix"] or "",
                "phase":                rec["phase"],
                "day_index":            rec["day_index"],
                "pickle_path":          str(path),
                "n_trial_light_edges":  sync["n_trial_light_edges"],
                "n_audio_onsets":       sync["n_audio_onsets"],
                "n_matched_pairs":      sync["n_matched_pairs"],
                "match_rate":           sync["match_rate"],
                "mean_residual_ms":     sync["mean_residual_sec"] * 1000
                                        if not np.isnan(sync["mean_residual_sec"])
                                        else np.nan,
                "std_residual_ms":      sync["std_residual_sec"] * 1000
                                        if not np.isnan(sync["std_residual_sec"])
                                        else np.nan,
                "max_abs_residual_ms":  sync["max_abs_residual_sec"] * 1000
                                        if not np.isnan(sync["max_abs_residual_sec"])
                                        else np.nan,
                "flag_misaligned":      sync["flag_misaligned"],
            }
            results.append(row)

            if verbose:
                flag_str = " *** FLAGGED ***" if sync["flag_misaligned"] else " OK"
                mean_ms  = sync["mean_residual_sec"] * 1000 if not np.isnan(
                           sync["mean_residual_sec"]) else float("nan")
                print(f"match={sync['match_rate']:.2f}  "
                      f"residual={mean_ms:.1f} ms  "
                      f"(expected <{EXPECTED_RESIDUAL_MS:.1f} ms){flag_str}")

        except Exception as e:
            warnings.warn(f"{tag}: ERROR — {e}")
            results.append({
                "mouse": rec["mouse"], "date": rec["date"],
                "suffix": rec["suffix"] or "", "phase": rec["phase"],
                "day_index": rec["day_index"], "pickle_path": str(path),
                "flag_misaligned": True, "error": str(e),
            })
            try:
                del d
            except NameError:
                pass
            gc.collect()

    df = pd.DataFrame(results)
    csv_path = OUTDIR_RES / "sync_qc_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Summary
    n_total   = len(df)
    n_flagged = int(df["flag_misaligned"].sum()) if "flag_misaligned" in df.columns else 0
    print(f"\n{'='*60}")
    print(f"Sync QC summary: {n_total} sessions, {n_flagged} flagged")
    if "mean_residual_ms" in df.columns:
        ok = df[~df["flag_misaligned"]]["mean_residual_ms"].dropna()
        if len(ok) > 0:
            print(f"Residual across OK sessions: "
                  f"mean={ok.mean():.2f} ms  max={ok.max():.2f} ms  "
                  f"(1 frame = {EXPECTED_RESIDUAL_MS:.1f} ms)")
    if n_flagged > 0:
        print("\nFlagged sessions:")
        flagged = df[df["flag_misaligned"] == True]
        for _, row in flagged.iterrows():
            print(f"  {row['mouse']} {row['date']} {row['suffix']}"
                  f"  match={row.get('match_rate', '?'):.2f}"
                  f"  residual={row.get('mean_residual_ms', float('nan')):.1f} ms")
    print(f"{'='*60}\n")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_sync_qc_summary(df: pd.DataFrame) -> None:
    """
    Figure 1: Residual distribution per mouse (violin).
    Figure 2: Match rate per session.
    """
    if df.empty:
        print("No data to plot.")
        return

    # ── Figure 1: residual distribution per mouse ────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    positions, labels = [], []
    for i, mouse in enumerate(MICE):
        sub = df[(df["mouse"] == mouse) & df["mean_residual_ms"].notna()]
        if sub.empty:
            continue
        vals = sub["mean_residual_ms"].values
        vp = ax.violinplot(vals, positions=[i], widths=0.6,
                           showmedians=True, showextrema=True)
        for pc in vp["bodies"]:
            pc.set_facecolor(MOUSE_COLORS.get(mouse, "gray"))
            pc.set_alpha(0.6)
        ax.scatter([i] * len(vals), vals,
                   color=MOUSE_COLORS.get(mouse, "gray"),
                   s=20, alpha=0.8, zorder=3)
        positions.append(i)
        labels.append(mouse.split("_")[0])

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axhline(EXPECTED_RESIDUAL_MS, color="green", linewidth=1,
               linestyle=":", label=f"1 frame ({EXPECTED_RESIDUAL_MS:.1f} ms)")
    ax.axhline(SYNC_FLAG_MEAN_OFFSET_SEC * 1000, color="red", linewidth=1,
               linestyle=":", label=f"Flag threshold ({SYNC_FLAG_MEAN_OFFSET_SEC*1000:.0f} ms)")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean residual: trial_light − Audio onset (ms)")
    ax.set_title("Sync QC: trial_light vs bpod Audio onset residuals\n"
                 "(Pipeline used trial_light for sync — residuals should be < 1 frame)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUTDIR_FIG / f"residual_distribution.{ext}", dpi=150)
    plt.close(fig)

    # ── Figure 2: match rate per session ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    for mouse in MICE:
        sub = df[df["mouse"] == mouse].copy()
        if sub.empty:
            continue
        ax.scatter(sub["day_index"], sub["match_rate"],
                   color=MOUSE_COLORS.get(mouse, "gray"),
                   label=mouse.split("_")[0], s=40, alpha=0.8)

    ax.axhline(SYNC_FLAG_MATCH_RATE, color="red", linewidth=1, linestyle=":",
               label=f"Flag threshold ({SYNC_FLAG_MATCH_RATE:.0%})")
    ax.set_xlabel("Session (day index within phase)")
    ax.set_ylabel("trial_light / Audio onset match rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Sync QC: trial_light match rate per session")
    ax.legend(frameon=False, fontsize=8, ncol=5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUTDIR_FIG / f"match_rate_per_session.{ext}", dpi=150)
    plt.close(fig)

    print(f"Figures saved to {OUTDIR_FIG}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Pose sync QC — verifying trial_light alignment quality")
    print("(Sync was already applied by the w2t-bkin pipeline;")
    print(" this checks that residuals are within 1 camera frame)\n")

    df_qc = run_sync_qc_all_sessions(verbose=True)
    plot_sync_qc_summary(df_qc)

    print("\nDone. If residuals are < ~7 ms for all sessions, sync is clean.")
    print("Sessions with flag_misaligned=True should be inspected manually.")

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 18:22:02 2025

@author: AG_Larkum
"""

import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from scipy import stats
import statsmodels.formula.api as smf
from pathlib import Path

from bpod_utils import (load_mat, latency, extract_sort_key,
                        extract_day_from_filename, sem, BASE_PATH, MIN_TRIALS,
                        bootstrap_ci, fdr_bh, rank_biserial_r, flag_outlier_sessions,
                        build_session_path)
from session_config import mice, sorted_mice
from plot_config import MOUSE_COLORS, PHASE_COLORS, style_ax

_OUTDIR = Path(os.environ.get('BPOD_OUTDIR',
               str(Path(__file__).parent / 'figures' / 'bpod' / 'default')))
_OUTDIR.mkdir(parents=True, exist_ok=True)
_fig_n = [0]

def _savefig(name='figure'):
    _fig_n[0] += 1
    safe = re.sub(r'[^\w]', '_', str(name))[:60]
    p = str(_OUTDIR / f'{_fig_n[0]:02d}_{safe}')
    for ext in ('png', 'pdf'):
        plt.savefig(f'{p}.{ext}', dpi=150, bbox_inches='tight')
    plt.close()


def collect_latencies_per_day(mouselist, phase, base_path):
    latency_per_mouse_day = {mouse: defaultdict(list) for mouse in mouselist}

    for mouse in mouselist:
        conds = mice[mouse]
        if phase == 'Training':
            tasks = conds['Training']
            subfolder = 'Training'
        else:
            tasks = conds['Expert']
            subfolder = 'Expert'

        for task in tasks:
            _mbase = mice[mouse].get('base_path', base_path)
            data_path = build_session_path(_mbase, mouse, subfolder, task)
            if not os.path.exists(data_path):
                continue

            session_files = [f for f in os.listdir(data_path) if f.endswith('.mat')]
            session_files.sort(key=extract_sort_key)

            for session_file in session_files:
                full_path = os.path.join(data_path, session_file)
                data = load_mat(full_path)

                try:
                    ntrials = data['SessionData']['nTrials']
                except Exception:
                    print(f"Warnung: nTrials nicht aus {session_file} extrahierbar")
                    continue

                if ntrials < MIN_TRIALS:
                    print(f"Session {session_file} übersprungen: nur {ntrials} Trials")
                    continue

                day = extract_day_from_filename(session_file)
                if day is None:
                    print(f"Tag aus {session_file} nicht extrahierbar")
                    continue

                trials = data['SessionData']['RawEvents']['Trial']
                if isinstance(trials, dict):
                    trials = [trials]

                if phase == 'Expert':
                    trial_types = data['SessionData']['TrialTypes']
                    lat_list = [latency(trial) for j, trial in enumerate(trials)
                                if trial_types[j] == 1 and not np.isnan(latency(trial))]
                else:
                    lat_list = [lat for trial in trials
                                for lat in [latency(trial)] if not np.isnan(lat)]

                latency_per_mouse_day[mouse][day].append(
                    np.nanmedian(lat_list) if lat_list else np.nan
                )

    # Build array using relative day index per mouse (not calendar-date alignment).
    # This ensures cohorts with different absolute dates align at day 1.
    per_mouse_vals = {}
    for mouse in mouselist:
        sorted_days = sorted(latency_per_mouse_day[mouse])
        per_mouse_vals[mouse] = [
            np.nanmedian(latency_per_mouse_day[mouse][d])
            if latency_per_mouse_day[mouse][d] else np.nan
            for d in sorted_days
        ]

    max_days = max((len(v) for v in per_mouse_vals.values()), default=0)
    arr = np.full((len(mouselist), max_days), np.nan)
    for i_mouse, mouse in enumerate(mouselist):
        vals = per_mouse_vals[mouse]
        arr[i_mouse, :len(vals)] = vals

    return arr, np.arange(1, max_days + 1), list(range(1, max_days + 1))


train_arr, train_days_idx, _ = collect_latencies_per_day(sorted_mice, 'Training', BASE_PATH)
exp_arr,   exp_days_idx,   _ = collect_latencies_per_day(sorted_mice, 'Expert',   BASE_PATH)

# Novice plot
if train_arr.size > 0:
    median_train = np.nanmedian(train_arr, axis=0)

    # bootstrap CI per day across mice
    boot_lo_tr = np.full(train_arr.shape[1], np.nan)
    boot_hi_tr = np.full(train_arr.shape[1], np.nan)
    for d in range(train_arr.shape[1]):
        lo, hi = bootstrap_ci(train_arr[:, d])
        boot_lo_tr[d] = lo
        boot_hi_tr[d] = hi

    # QC
    print("\n[QC] Novice latency sessions:")
    for i, mouse in enumerate(sorted_mice):
        row = train_arr[i, :]
        flag_outlier_sessions(row[~np.isnan(row)], mouse_id=mouse)

    fig, ax = plt.subplots(figsize=(7, 4))
    style_ax(ax)
    ax.plot(train_days_idx, median_train, color=PHASE_COLORS['Training'],
            linewidth=2.5, marker='o', markersize=5, label='Group Median', zorder=5)
    ax.fill_between(train_days_idx, boot_lo_tr, boot_hi_tr,
                    color=PHASE_COLORS['Training'], alpha=0.2, label='95% Bootstrap CI')
    ax.set_ylim(0.3, 1.2)
    ax.set_xlim(train_days_idx[0] - 0.5, train_days_idx[-1] + 0.5)
    ax.set_title('Novice Latency (Median ± 95% Bootstrap CI)')
    ax.set_xlabel('Day')
    ax.set_ylabel('Median Latency (s)')
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    plt.tight_layout()
    _savefig('novice_latency_curve')

    # Heatmap Training
    fig, ax = plt.subplots(figsize=(max(6, train_arr.shape[1] * 0.5), 2.5))
    im = ax.imshow(train_arr, aspect='auto', cmap='RdYlGn_r',
                   vmin=0.4, vmax=1.0, interpolation='nearest')
    ax.set_yticks(range(len(sorted_mice)))
    ax.set_yticklabels([m.split('_(')[0] for m in sorted_mice], fontsize=8)
    ax.set_xticks(range(train_arr.shape[1]))
    ax.set_xticklabels(range(1, train_arr.shape[1] + 1), fontsize=7)
    ax.set_xlabel("Day")
    ax.set_title("Heatmap — Novice Median Latency per Mouse × Day (s)")
    plt.colorbar(im, ax=ax, label='Median Latency (s)')
    plt.tight_layout()
    _savefig('novice_latency_heatmap')
else:
    print('Keine Trainingsdaten gefunden.')

# Expert plot
if exp_arr.size > 0:
    median_exp = np.nanmedian(exp_arr, axis=0)

    boot_lo_exp = np.full(exp_arr.shape[1], np.nan)
    boot_hi_exp = np.full(exp_arr.shape[1], np.nan)
    for d in range(exp_arr.shape[1]):
        lo, hi = bootstrap_ci(exp_arr[:, d])
        boot_lo_exp[d] = lo
        boot_hi_exp[d] = hi

    print("\n[QC] Expert latency sessions:")
    for i, mouse in enumerate(sorted_mice):
        row = exp_arr[i, :]
        flag_outlier_sessions(row[~np.isnan(row)], mouse_id=mouse)

    fig, ax = plt.subplots(figsize=(7, 4))
    style_ax(ax)
    ax.plot(exp_days_idx, median_exp, color=PHASE_COLORS['Expert'],
            linewidth=2.5, marker='o', markersize=5, label='Group Median', zorder=5)
    ax.fill_between(exp_days_idx, boot_lo_exp, boot_hi_exp,
                    color=PHASE_COLORS['Expert'], alpha=0.2, label='95% Bootstrap CI')
    ax.set_ylim(0.3, 1.2)
    ax.set_xlim(exp_days_idx[0] - 0.5, exp_days_idx[-1] + 0.5)
    ax.set_title('Expert Latency (Median ± 95% Bootstrap CI)')
    ax.set_xlabel('Day')
    ax.set_ylabel('Median Latency (s)')
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    plt.tight_layout()
    _savefig('expert_latency_curve')

    # Heatmap Expert
    fig, ax = plt.subplots(figsize=(max(6, exp_arr.shape[1] * 0.5), 2.5))
    im = ax.imshow(exp_arr, aspect='auto', cmap='RdYlGn_r',
                   vmin=0.4, vmax=1.0, interpolation='nearest')
    ax.set_yticks(range(len(sorted_mice)))
    ax.set_yticklabels([m.split('_(')[0] for m in sorted_mice], fontsize=8)
    ax.set_xticks(range(exp_arr.shape[1]))
    ax.set_xticklabels(range(1, exp_arr.shape[1] + 1), fontsize=7)
    ax.set_xlabel("Day")
    ax.set_title("Heatmap — Expert Median Latency per Mouse × Day (s)")
    plt.colorbar(im, ax=ax, label='Median Latency (s)')
    plt.tight_layout()
    _savefig('expert_latency_heatmap')
else:
    print('Keine Expertendaten gefunden.')

# ── Combined Training → Expert latency timeline ───────────────────────────────
if train_arr.size > 0 and exp_arr.size > 0:
    max_train_len = train_arr.shape[1]
    exp_offset    = max_train_len  # expert day 1 → x = max_train_len + 1

    # x-coordinates
    x_train = train_days_idx                          # 1 … N_train
    x_exp   = exp_days_idx + exp_offset               # N_train+1 … N_train+N_exp

    # group medians
    median_train_c = np.nanmedian(train_arr, axis=0)
    median_exp_c   = np.nanmedian(exp_arr,   axis=0)

    # bootstrap CI (reuse already-computed arrays from above)
    # (boot_lo_tr / boot_hi_tr and boot_lo_exp / boot_hi_exp were built above)

    fig, ax = plt.subplots(figsize=(max(10, (max_train_len + exp_arr.shape[1]) * 0.5), 4))
    style_ax(ax)

    # group median line
    x_all    = np.concatenate([x_train, x_exp])
    med_all  = np.concatenate([median_train_c, median_exp_c])
    ax.plot(x_all, med_all, color='black', linewidth=2.5,
            marker='o', markersize=5, label='Group Median', zorder=5)

    # bootstrap CI bands
    ax.fill_between(x_train, boot_lo_tr,  boot_hi_tr,
                    color=PHASE_COLORS['Training'], alpha=0.18, label='95% CI (Novice)')
    ax.fill_between(x_exp,   boot_lo_exp, boot_hi_exp,
                    color=PHASE_COLORS['Expert'],   alpha=0.18, label='95% CI (Expert)')

    # phase transition marker
    ax.axvline(max_train_len + 0.5, color='red', linestyle=':', linewidth=1.5,
               label='Novice → Expert')

    # x-tick labels: 1…N_train then 1…N_exp
    all_x     = list(x_train) + list(x_exp)
    all_xlabs = list(range(1, max_train_len + 1)) + list(range(1, exp_arr.shape[1] + 1))
    ax.set_xticks(all_x)
    ax.set_xticklabels([str(l) for l in all_xlabs], fontsize=7)

    # phase labels below x-axis (data coordinates, clip_on=False so they sit under the spine)
    ax.text(np.mean(x_train), 0.26,
            'Novice', ha='center', va='top', fontsize=9,
            color=PHASE_COLORS['Training'], clip_on=False)
    ax.text(np.mean(x_exp), 0.26,
            'Expert', ha='center', va='top', fontsize=9,
            color=PHASE_COLORS['Expert'], clip_on=False)

    ax.set_ylim(0.3, 1.2)
    ax.set_xlabel('Day (within phase)')
    ax.set_ylabel('Median Latency (s)')
    ax.set_title('Combined Latency Learning Curve: Novice → Expert')
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8, ncol=3)
    plt.tight_layout()
    _savefig('combined_latency_curve')


# ── Statistics ────────────────────────────────────────────────────────────────

def run_latency_statistics(sorted_mice, train_arr, exp_arr):
    print("\n" + "=" * 65)
    print("STATISTICS — Response Latency Learning Curves")
    print("=" * 65)

    # Build long-format DataFrame from arrays (mice × days)
    rows = []
    for i, mouse in enumerate(sorted_mice):
        for d_idx in range(train_arr.shape[1]):
            val = train_arr[i, d_idx]
            if not np.isnan(val):
                rows.append({'mouse': mouse, 'phase': 'Training', 'day': d_idx + 1, 'latency': val})
        for d_idx in range(exp_arr.shape[1]):
            val = exp_arr[i, d_idx]
            if not np.isnan(val):
                rows.append({'mouse': mouse, 'phase': 'Expert', 'day': d_idx + 1, 'latency': val})
    df = pd.DataFrame(rows)
    if df.empty:
        print("No data for statistics.")
        return

    # ── Descriptive ──────────────────────────────────────────────────
    print("\n--- Descriptive (per phase) ---")
    desc = df.groupby('phase')['latency'].agg(n='count', median='median', std='std').round(3)
    print(desc.to_string())

    # ── LME: trend over days per phase ───────────────────────────────
    for phase_label, phase_df in [('Training', df[df['phase'] == 'Training']),
                                   ('Expert',   df[df['phase'] == 'Expert'])]:
        print(f"\n--- LME: Latency ~ day ({phase_label} phase) ---")
        print("  latency ~ day  |  random intercept: mouse\n")
        if phase_df['mouse'].nunique() < 2:
            print("  — insufficient mice for LME —")
            continue
        try:
            model = smf.mixedlm("latency ~ day", phase_df, groups=phase_df['mouse'])
            result = model.fit(reml=True, method='lbfgs')
            fe = result.fe_params
            ci = result.conf_int()
            pvals = result.pvalues
            print(f"  {'Parameter':<20} {'Coef':>8}  {'[0.025':>7}  {'0.975]':>7}  {'p':>7}")
            print("  " + "-" * 55)
            for param in fe.index:
                coef = fe[param]
                lo, hi = ci.loc[param]
                p = pvals[param]
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
                print(f"  {param:<20} {coef:>8.3f}  {lo:>7.3f}  {hi:>7.3f}  {p:>7.4f}  {sig}")
            print(f"\n  Random effect (mouse) variance: {result.cov_re.values[0][0]:.4f}")
        except Exception as e:
            print(f"  LME failed: {e}")

    # ── Wilcoxon + FDR + effect size ─────────────────────────────────
    comparisons = []

    early_vals, late_vals = [], []
    for i in range(train_arr.shape[0]):
        row = train_arr[i, :]
        valid = row[~np.isnan(row)]
        if len(valid) >= 3:
            early_vals.append(np.median(valid[:3]))
            late_vals.append(np.median(valid[-3:]))
    comparisons.append(('early vs late training', early_vals, late_vals))

    last_train, first_exp = [], []
    for i in range(len(sorted_mice)):
        t_valid = train_arr[i, ~np.isnan(train_arr[i, :])]
        e_valid = exp_arr[i, ~np.isnan(exp_arr[i, :])]
        if len(t_valid) > 0 and len(e_valid) > 0:
            last_train.append(t_valid[-1])
            first_exp.append(e_valid[0])
    comparisons.append(('last train vs first expert', last_train, first_exp))

    raw_results = []
    for label, a, b in comparisons:
        n = len(a)
        if n >= 2:
            w, p = stats.wilcoxon(a, b)
            r = rank_biserial_r(a, b)
            raw_results.append((label, n, np.median(a), np.median(b), w, p, r))
        else:
            raw_results.append((label, n, np.nan, np.nan, np.nan, np.nan, np.nan))

    valid_p = np.array([rv[5] for rv in raw_results if not np.isnan(rv[5])])
    adj_p = fdr_bh(valid_p) if len(valid_p) > 0 else []
    adj_iter = iter(adj_p)

    print("\n--- Wilcoxon comparisons (FDR-corrected, rank-biserial r) ---")
    print(f"  {'Comparison':<30} {'n':>3}  {'A med':>7}  {'B med':>7}  "
          f"{'W':>6}  {'p':>7}  {'p_FDR':>7}  {'r':>5}  sig")
    print("  " + "-" * 80)
    for label, n, ma, mb, w, p, r in raw_results:
        if np.isnan(p):
            print(f"  {label:<30} {n:>3}  — insufficient n —")
            continue
        p_fdr = next(adj_iter)
        sig = '***' if p_fdr < 0.001 else '**' if p_fdr < 0.01 else '*' if p_fdr < 0.05 else 'ns'
        print(f"  {label:<30} {n:>3}  {ma:>7.3f}  {mb:>7.3f}  "
              f"{w:>6.1f}  {p:>7.4f}  {p_fdr:>7.4f}  {r:>5.2f}  {sig}")

    print("=" * 65)


run_latency_statistics(sorted_mice, train_arr, exp_arr)

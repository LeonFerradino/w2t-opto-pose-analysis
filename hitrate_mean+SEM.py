# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 13:39:21 2025

@author: admin
"""

import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
from collections import defaultdict
from scipy import stats
import statsmodels.formula.api as smf
from pathlib import Path

from bpod_utils import (load_mat, is_hit, extract_sort_key,
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

training_all = []
expert_all = []

# collect data
for mouse in sorted_mice:
    conds = mice[mouse]

    # ----------------- Training -----------------
    training_day_values = defaultdict(list)
    for train_task in conds['Training']:
        _mbase = conds.get('base_path', BASE_PATH)
        train_path = build_session_path(_mbase, mouse, 'Training', train_task)
        if not os.path.exists(train_path):
            print(f"Ordner nicht gefunden: {train_path}")
            continue

        session_files = [f for f in os.listdir(train_path) if f.endswith('.mat')]
        session_files.sort(key=extract_sort_key)

        for session_file in session_files:
            full_path = os.path.join(train_path, session_file)
            data = load_mat(full_path)

            try:
                ntrials = data['SessionData']['nTrials']
            except Exception:
                print(f"Warning: Could not extract nTrials from {session_file}")
                continue

            if ntrials < MIN_TRIALS:
                print(f"Session {session_file} skipped, has only {ntrials} trials.")
                continue

            day = extract_day_from_filename(session_file)
            if day is None:
                print(f"Cannot extract day from filename {session_file}")
                continue

            trials = data['SessionData']['RawEvents']['Trial']
            if isinstance(trials, dict):
                trials = [trials]

            hits = [1 if is_hit(trial) else 0 for trial in trials]
            hit_rate = np.sum(hits) / len(hits) * 100 if hits else np.nan
            training_day_values[day].append(hit_rate)

    training_days_sorted = sorted(training_day_values)
    training_hit_rates = [np.nanmean(training_day_values[day]) for day in training_days_sorted]
    training_all.append(training_hit_rates)

    # ----------------- Expert -----------------
    expert_day_values = defaultdict(list)
    for expert_task in conds['Expert']:
        _mbase = conds.get('base_path', BASE_PATH)
        expert_path = build_session_path(_mbase, mouse, 'Expert', expert_task)
        if not os.path.exists(expert_path):
            print(f"Ordner nicht gefunden: {expert_path}")
            continue

        session_files = [f for f in os.listdir(expert_path) if f.endswith('.mat')]
        session_files.sort(key=extract_sort_key)

        for session_file in session_files:
            full_path = os.path.join(expert_path, session_file)
            data = load_mat(full_path)

            try:
                ntrials = data['SessionData']['nTrials']
            except Exception:
                print(f"Warning: Could not extract nTrials from {session_file}")
                continue

            if ntrials < MIN_TRIALS:
                print(f"Session {session_file} skipped, has only {ntrials} trials.")
                continue

            day = extract_day_from_filename(session_file)
            if day is None:
                print(f"Cannot extract day from filename {session_file}")
                continue

            trials = data['SessionData']['RawEvents']['Trial']
            if isinstance(trials, dict):
                trials = [trials]

            trial_types = data['SessionData']['TrialTypes']
            hits = []
            valid_trials = 0
            for j, trial in enumerate(trials):
                if trial_types[j] == 1:
                    hits.append(1 if is_hit(trial) else 0)
                    valid_trials += 1

            hit_rate = np.sum(hits) / valid_trials * 100 if valid_trials > 0 else np.nan
            expert_day_values[day].append(hit_rate)

    expert_days_sorted = sorted(expert_day_values)
    expert_hit_rates = [np.nanmean(expert_day_values[day]) for day in expert_days_sorted]
    expert_all.append(expert_hit_rates)

# Plot Training
if not training_all or all(len(val) == 0 for val in training_all):
    print(f"No mouse has training sessions with >={MIN_TRIALS} trials after filtering.")
else:
    max_train_len = max(len(x) for x in training_all)
    training_array = np.full((len(training_all), max_train_len), np.nan)
    for i, arr in enumerate(training_all):
        training_array[i, :len(arr)] = arr

    training_mean = np.nanmean(training_array, axis=0)
    training_sem = sem(training_array)
    sessions_training = np.arange(1, max_train_len + 1)

    # bootstrap CI across mice per day
    boot_lo_train = np.full(max_train_len, np.nan)
    boot_hi_train = np.full(max_train_len, np.nan)
    for d in range(max_train_len):
        col_vals = training_array[:, d]
        lo, hi = bootstrap_ci(col_vals)
        boot_lo_train[d] = lo
        boot_hi_train[d] = hi

    # session QC per mouse
    print("\n[QC] Novice hit-rate sessions:")
    for i, mouse in enumerate(sorted_mice):
        row = training_array[i, :]
        flag_outlier_sessions(row[~np.isnan(row)], mouse_id=mouse)

    fig, ax = plt.subplots(figsize=(7, 4))
    style_ax(ax)
    # group mean + bootstrap CI
    ax.plot(sessions_training, training_mean, color=PHASE_COLORS['Training'],
            linewidth=2.5, marker="o", markersize=5, label="Group Mean", zorder=5)
    ax.fill_between(sessions_training, boot_lo_train, boot_hi_train,
                    color=PHASE_COLORS['Training'], alpha=0.2, label="95% Bootstrap CI")
    ax.set_ylim(40, 105)
    ax.set_xlim(0.5, max_train_len + 0.5)
    ax.set_title("Novice Success Rate (Mean ± 95% Bootstrap CI)")
    ax.set_xlabel("Day")
    ax.set_ylabel("Success Rate (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    plt.tight_layout()
    _savefig('novice_hitrate_curve')

# Plot Expert
if not expert_all or all(len(val) == 0 for val in expert_all):
    print(f"No mouse has expert sessions with >={MIN_TRIALS} trials after filtering.")
else:
    max_expert_len = max(len(x) for x in expert_all)
    expert_array = np.full((len(expert_all), max_expert_len), np.nan)
    for i, arr in enumerate(expert_all):
        expert_array[i, :len(arr)] = arr

    expert_mean = np.nanmean(expert_array, axis=0)
    expert_sem = sem(expert_array)
    sessions_expert = np.arange(1, max_expert_len + 1)

    boot_lo_exp = np.full(max_expert_len, np.nan)
    boot_hi_exp = np.full(max_expert_len, np.nan)
    for d in range(max_expert_len):
        col_vals = expert_array[:, d]
        lo, hi = bootstrap_ci(col_vals)
        boot_lo_exp[d] = lo
        boot_hi_exp[d] = hi

    print("\n[QC] Expert hit-rate sessions:")
    for i, mouse in enumerate(sorted_mice):
        row = expert_array[i, :]
        flag_outlier_sessions(row[~np.isnan(row)], mouse_id=mouse)

    fig, ax = plt.subplots(figsize=(7, 4))
    style_ax(ax)
    ax.plot(sessions_expert, expert_mean, color=PHASE_COLORS['Expert'],
            linewidth=2.5, marker="o", markersize=5, label="Group Mean", zorder=5)
    ax.fill_between(sessions_expert, boot_lo_exp, boot_hi_exp,
                    color=PHASE_COLORS['Expert'], alpha=0.2, label="95% Bootstrap CI")
    ax.set_ylim(40, 105)
    ax.set_xlim(0.5, max_expert_len + 0.5)
    ax.set_title("Expert Success Rate (Mean ± 95% Bootstrap CI)")
    ax.set_xlabel("Day")
    ax.set_ylabel("Success Rate (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    plt.tight_layout()
    _savefig('expert_hitrate_curve')

# Heatmap: mice × days (Training)
if training_all and any(len(v) > 0 for v in training_all):
    fig, ax = plt.subplots(figsize=(max(6, max_train_len * 0.5), 2.5))
    im = ax.imshow(training_array, aspect='auto', cmap='RdYlGn',
                   vmin=40, vmax=100, interpolation='nearest')
    ax.set_yticks(range(len(sorted_mice)))
    ax.set_yticklabels([m.split('_(')[0] for m in sorted_mice], fontsize=8)
    ax.set_xticks(range(max_train_len))
    ax.set_xticklabels(range(1, max_train_len + 1), fontsize=7)
    ax.set_xlabel("Day")
    ax.set_title("Heatmap — Novice Success Rate per Mouse × Day (%)")
    plt.colorbar(im, ax=ax, label='Success Rate (%)')
    plt.tight_layout()
    _savefig('novice_hitrate_heatmap')

# Heatmap: mice × days (Expert)
if expert_all and any(len(v) > 0 for v in expert_all):
    fig, ax = plt.subplots(figsize=(max(6, max_expert_len * 0.5), 2.5))
    im = ax.imshow(expert_array, aspect='auto', cmap='RdYlGn',
                   vmin=40, vmax=100, interpolation='nearest')
    ax.set_yticks(range(len(sorted_mice)))
    ax.set_yticklabels([m.split('_(')[0] for m in sorted_mice], fontsize=8)
    ax.set_xticks(range(max_expert_len))
    ax.set_xticklabels(range(1, max_expert_len + 1), fontsize=7)
    ax.set_xlabel("Day")
    ax.set_title("Heatmap — Expert Success Rate per Mouse × Day (%)")
    plt.colorbar(im, ax=ax, label='Success Rate (%)')
    plt.tight_layout()
    _savefig('expert_hitrate_heatmap')

# Combined Training → Expert timeline
if (training_all and any(len(v) > 0 for v in training_all) and
        expert_all and any(len(v) > 0 for v in expert_all)):

    fig, ax = plt.subplots(figsize=(max(10, (max_train_len + max_expert_len) * 0.55), 4))
    style_ax(ax)

    # group means
    combined_mean = np.concatenate([training_mean,  expert_mean])
    combined_x    = np.concatenate([sessions_training, sessions_expert + max_train_len])

    # bootstrap CI for combined
    combined_lo = np.concatenate([boot_lo_train, boot_lo_exp])
    combined_hi = np.concatenate([boot_hi_train, boot_hi_exp])

    ax.plot(combined_x, combined_mean, color='black',
            linewidth=2.5, marker='o', markersize=5, label='Group Mean', zorder=6)
    ax.fill_between(combined_x, combined_lo, combined_hi,
                    color='black', alpha=0.15, label='95% Bootstrap CI')

    # phase transition marker
    ax.axvline(max_train_len + 0.5, color='red', linewidth=1.8,
               linestyle=':', alpha=0.8, label='Novice → Expert')
    ax.text(max_train_len + 0.6, 42, 'Expert', color='red', fontsize=8, va='bottom')
    ax.text(max_train_len - 0.6, 42, 'Novice', color='red', fontsize=8,
            va='bottom', ha='right')

    all_x = np.concatenate([sessions_training, sessions_expert + max_train_len])
    ax.set_xticks(all_x)
    ax.set_xticklabels(
        [str(d) for d in sessions_training] + [str(d) for d in sessions_expert],
        fontsize=7)
    ax.set_xlabel("Day within phase")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(40, 105)
    ax.set_title("Combined Learning Curve: Novice → Expert\n"
                 "(dashed line = transition, shading = 95% bootstrap CI)")
    ax.legend(frameon=False, fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _savefig('combined_hitrate_curve')


# ── Statistics ────────────────────────────────────────────────────────────────

def run_learning_statistics(sorted_mice, training_all, expert_all):
    print("\n" + "=" * 65)
    print("STATISTICS — Success Rate Learning Curves")
    print("=" * 65)

    # Build long-format DataFrame: mouse, phase, day, hitrate
    rows = []
    for i, mouse in enumerate(sorted_mice):
        for d_idx, hr in enumerate(training_all[i], start=1):
            if not np.isnan(hr):
                rows.append({'mouse': mouse, 'phase': 'Training', 'day': d_idx, 'hitrate': hr})
        for d_idx, hr in enumerate(expert_all[i], start=1):
            if not np.isnan(hr):
                rows.append({'mouse': mouse, 'phase': 'Expert', 'day': d_idx, 'hitrate': hr})
    df = pd.DataFrame(rows)
    if df.empty:
        print("No data for statistics.")
        return

    # ── Descriptive ──────────────────────────────────────────────────
    print("\n--- Descriptive (per phase) ---")
    desc = df.groupby('phase')['hitrate'].agg(n='count', mean='mean', std='std').round(2)
    print(desc.to_string())

    # ── LME: trend over training days ────────────────────────────────
    for phase_label, phase_df in [('Training', df[df['phase'] == 'Training']),
                                   ('Expert',   df[df['phase'] == 'Expert'])]:
        print(f"\n--- LME: Hit rate ~ day ({phase_label} phase) ---")
        print("  hitrate ~ day  |  random intercept: mouse\n")
        if phase_df['mouse'].nunique() < 2:
            print("  — insufficient mice for LME —")
            continue
        try:
            model = smf.mixedlm("hitrate ~ day", phase_df, groups=phase_df['mouse'])
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
                print(f"  {param:<20} {coef:>8.2f}  {lo:>7.2f}  {hi:>7.2f}  {p:>7.4f}  {sig}")
            print(f"\n  Random effect (mouse) variance: {result.cov_re.values[0][0]:.3f}")
        except Exception as e:
            print(f"  LME failed: {e}")

    # ── Wilcoxon + FDR + effect size ─────────────────────────────────
    comparisons = []
    # early vs late training
    early_vals, late_vals = [], []
    for arr in training_all:
        valid = [v for v in arr if not np.isnan(v)]
        if len(valid) >= 3:
            early_vals.append(np.mean(valid[:3]))
            late_vals.append(np.mean(valid[-3:]))
    comparisons.append(('early vs late training', early_vals, late_vals))

    # last training vs first expert
    last_train, first_exp = [], []
    for i in range(len(sorted_mice)):
        t_valid = [v for v in training_all[i] if not np.isnan(v)]
        e_valid = [v for v in expert_all[i] if not np.isnan(v)]
        if t_valid and e_valid:
            last_train.append(t_valid[-1])
            first_exp.append(e_valid[0])
    comparisons.append(('last train vs first expert', last_train, first_exp))

    # collect p-values for FDR
    raw_p = []
    results = []
    for label, a, b in comparisons:
        n = len(a)
        if n >= 2:
            w, p = stats.wilcoxon(a, b)
            r = rank_biserial_r(a, b)
            raw_p.append(p)
            results.append((label, n, np.mean(a), np.mean(b), w, p, r))
        else:
            results.append((label, n, np.nan, np.nan, np.nan, np.nan, np.nan))

    adj_p = fdr_bh(np.array([r[5] for r in results if not np.isnan(r[5])]))
    adj_iter = iter(adj_p)

    print("\n--- Wilcoxon comparisons (FDR-corrected, rank-biserial r) ---")
    print(f"  {'Comparison':<30} {'n':>3}  {'A mean':>7}  {'B mean':>7}  "
          f"{'W':>6}  {'p':>7}  {'p_FDR':>7}  {'r':>5}  sig")
    print("  " + "-" * 80)
    for label, n, ma, mb, w, p, r in results:
        if np.isnan(p):
            print(f"  {label:<30} {n:>3}  — insufficient n —")
            continue
        p_fdr = next(adj_iter)
        sig = '***' if p_fdr < 0.001 else '**' if p_fdr < 0.01 else '*' if p_fdr < 0.05 else 'ns'
        print(f"  {label:<30} {n:>3}  {ma:>7.1f}  {mb:>7.1f}  "
              f"{w:>6.1f}  {p:>7.4f}  {p_fdr:>7.4f}  {r:>5.2f}  {sig}")

    print("=" * 65)


run_learning_statistics(sorted_mice, training_all, expert_all)

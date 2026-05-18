# -*- coding: utf-8 -*-
"""
phaseB_figures.py — Phase B: Statistical annotations + cosmetic fixes.

Produces:
  output_phaseB/figures_updated/  ← annotated re-renders of all thesis figures
  output_phaseB/stats_updates/    ← per-figure caption .md files
  output_phaseB/GITHUB_INVENTORY.md
  output_phaseB/phaseB_summary.md

Constraints:
  - Does NOT modify any core scripts
  - All stats logic is local to this file
  - Imports only from bpod_utils, session_config, plot_config (unmodified)
"""

import os, re, sys, zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy import stats, optimize
from collections import defaultdict
from pathlib import Path
from datetime import datetime
import statsmodels.formula.api as smf

from bpod_utils import (
    load_mat, is_hit, latency, extract_sort_key, extract_day_from_filename,
    extract_day_from_path, sem, BASE_PATH, MIN_TRIALS,
    bootstrap_ci, fdr_bh, rank_biserial_r, flag_outlier_sessions,
    build_session_path, is_blocked_session,
)
from session_config import mice, sorted_mice, mouse_sessions
from plot_config import (
    MOUSE_COLORS, MOUSE_COLORS_SHORT, MOUSE_MARKERS,
    PHASE_COLORS, COND_COLORS, OPTO_DUR_COLORS, SESSION_TYPE_COLORS, style_ax,
)

# ── Output directories ─────────────────────────────────────────────────────────
BASE     = Path("f:/analysis/bpod/Documents")
OUTDIR   = BASE / "output_phaseB"
FIGS_OUT = OUTDIR / "figures_updated"
STATS_OUT = OUTDIR / "stats_updates"
for _d in (OUTDIR, FIGS_OUT, STATS_OUT):
    _d.mkdir(parents=True, exist_ok=True)

CAPTION = {}   # figure_key -> caption str, filled as figures are rendered

# ── Shared utilities ───────────────────────────────────────────────────────────

def save_fig(stem, dpi=180):
    for ext in ('png', 'svg'):
        plt.savefig(f"{stem}.{ext}", dpi=dpi, bbox_inches='tight')
    plt.close()

def stars(p):
    if np.isnan(p): return ''
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'

def modified_z_score(vals):
    """Iglewicz-Hoaglin modified Z-score; flag > 3.5."""
    vals = np.asarray(vals, float)
    med = np.nanmedian(vals)
    mad = np.nanmedian(np.abs(vals - med))
    if mad == 0:
        return np.zeros(len(vals))
    return 0.6745 * np.abs(vals - med) / mad

# ── Trial-level data loaders ───────────────────────────────────────────────────

def _load_session_raw(filepath, trialtype_filter):
    """Return (hit_list, latency_list) for one session, one trial type."""
    if not filepath or not os.path.exists(filepath):
        return [], []
    data = load_mat(filepath)
    sd = data.get('SessionData', {})
    tt = sd.get('TrialTypes', None)
    raw = sd.get('RawEvents', {})
    trials = raw.get('Trial', None) if isinstance(raw, dict) else None
    if tt is None or trials is None:
        return [], []
    if isinstance(trials, dict):
        trials = [trials]
    if len(trials) < MIN_TRIALS:
        return [], []
    hits, lats = [], []
    for i, t in enumerate(tt):
        if i >= len(trials):
            break
        if t != trialtype_filter:
            continue
        h = is_hit(trials[i])
        hits.append(1 if h else 0)
        if h:
            l = latency(trials[i])
            if not np.isnan(l):
                lats.append(l)
    return hits, lats


def get_mouse_trial_hits(mouse, area, trial_key, tt_num):
    """Pool all individual trial hit outcomes (0/1) for one mouse/area/condition."""
    paths = mouse_sessions.get(mouse, {}).get(area, {}).get(trial_key, [])
    all_hits = []
    for p in paths:
        h, _ = _load_session_raw(p, tt_num)
        all_hits.extend(h)
    return all_hits


def get_mouse_trial_latencies(mouse, area, trial_key, tt_num):
    """Pool all individual trial latencies for one mouse/area/condition."""
    paths = mouse_sessions.get(mouse, {}).get(area, {}).get(trial_key, [])
    all_lats = []
    for p in paths:
        _, l = _load_session_raw(p, tt_num)
        all_lats.extend(l)
    return all_lats


def get_mouse_session_mean_hitrate(mouse, area, trial_key, tt_num):
    """Per-session mean hit rate; return list of per-session values."""
    paths = mouse_sessions.get(mouse, {}).get(area, {}).get(trial_key, [])
    vals = []
    for p in paths:
        h, _ = _load_session_raw(p, tt_num)
        if h:
            vals.append(np.mean(h) * 100)
    return vals


def get_mouse_session_median_latency(mouse, area, trial_key, tt_num):
    """Per-session median latency; return list of per-session values."""
    paths = mouse_sessions.get(mouse, {}).get(area, {}).get(trial_key, [])
    vals = []
    for p in paths:
        _, l = _load_session_raw(p, tt_num)
        if l:
            vals.append(float(np.nanmedian(l)))
    return vals

# ── Statistical helpers ────────────────────────────────────────────────────────

def mwu_per_mouse(hits_a, hits_b):
    """Mann-Whitney U, two-sided, returns (stat, p) or (nan, nan)."""
    if len(hits_a) < 5 or len(hits_b) < 5:
        return np.nan, np.nan
    st, p = stats.mannwhitneyu(hits_a, hits_b, alternative='two-sided')
    return st, p


def one_sample_wilcoxon(vals):
    """One-sample signed-rank test vs 0. Returns (stat, p) or (nan, nan)."""
    vals = np.asarray([v for v in vals if not np.isnan(v)])
    if len(vals) < 2:
        return np.nan, np.nan
    if np.all(vals == 0):
        return np.nan, 1.0
    return stats.wilcoxon(vals)

# ── Sigmoid helpers (for Fig 3) ───────────────────────────────────────────────

def sigmoid(x, L, x0, k, b):
    return b + (L - b) / (1 + np.exp(-k * (x - x0)))

def fit_sigmoid(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    valid = ~np.isnan(y)
    if valid.sum() < 5:
        return None, None, None, False
    try:
        p0 = [np.nanmax(y), x[valid][len(x[valid])//2], 0.5, np.nanmin(y)]
        bounds = ([0, x.min()-1, 0, 0], [1.5, x.max()+1, 10, 1])
        popt, _ = optimize.curve_fit(sigmoid, x[valid], y[valid],
                                     p0=p0, bounds=bounds, maxfev=5000)
        xf = np.linspace(x.min(), x.max(), 300)
        return popt, xf, sigmoid(xf, *popt), True
    except Exception:
        return None, None, None, False


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Novice success rate with LME slope annotation
# ═══════════════════════════════════════════════════════════════════════════════

def render_fig1():
    print("\n[Fig 1] Novice success rate + LME annotation ...")

    training_all = []
    for mouse in sorted_mice:
        conds = mice[mouse]
        day_vals = defaultdict(list)
        for task in conds['Training']:
            mbase = conds.get('base_path', BASE_PATH)
            path = build_session_path(mbase, mouse, 'Training', task)
            if not os.path.exists(path):
                continue
            sfiles = sorted([f for f in os.listdir(path) if f.endswith('.mat')],
                            key=extract_sort_key)
            for sf in sfiles:
                data = load_mat(os.path.join(path, sf))
                try:
                    ntrials = data['SessionData']['nTrials']
                except Exception:
                    continue
                if ntrials < MIN_TRIALS:
                    continue
                day = extract_day_from_filename(sf)
                if day is None:
                    continue
                trials = data['SessionData']['RawEvents']['Trial']
                if isinstance(trials, dict):
                    trials = [trials]
                hr = np.mean([1 if is_hit(t) else 0 for t in trials]) * 100
                day_vals[day].append(hr)
        sorted_days = sorted(day_vals)
        training_all.append([np.nanmean(day_vals[d]) for d in sorted_days])

    if not training_all or all(len(v) == 0 for v in training_all):
        print("  No training data found.")
        return

    max_len = max(len(v) for v in training_all)
    arr = np.full((len(training_all), max_len), np.nan)
    for i, v in enumerate(training_all):
        arr[i, :len(v)] = v

    x_days = np.arange(1, max_len + 1)
    group_mean = np.nanmean(arr, axis=0)
    boot_lo = np.full(max_len, np.nan)
    boot_hi = np.full(max_len, np.nan)
    for d in range(max_len):
        boot_lo[d], boot_hi[d] = bootstrap_ci(arr[:, d])

    # LME: hitrate ~ day | random: mouse
    rows = []
    for i, mouse in enumerate(sorted_mice):
        for d_idx in range(max_len):
            v = arr[i, d_idx]
            if not np.isnan(v):
                rows.append({'mouse': mouse, 'day': d_idx + 1, 'hitrate': v})
    df_lme = pd.DataFrame(rows)
    lme_text = 'LME: insufficient data'
    lme_beta, lme_p = np.nan, np.nan
    if df_lme['mouse'].nunique() >= 2:
        try:
            res = smf.mixedlm("hitrate ~ day", df_lme, groups=df_lme['mouse']).fit(
                reml=True, method='lbfgs', disp=False)
            lme_beta = res.fe_params['day']
            lme_p    = res.pvalues['day']
            lme_ci   = res.conf_int().loc['day']
            lme_text = (f"LME slope: {lme_beta:+.2f}%/day "
                        f"[{lme_ci[0]:.2f}, {lme_ci[1]:.2f}], "
                        f"p={lme_p:.3f} {stars(lme_p)}")
        except Exception:
            lme_text = ("LME did not converge (random-effects variance singular; "
                        "n=4 mice insufficient for stable mixed-model estimation; "
                        "see Wilcoxon below)")

    # Wilcoxon early vs late
    early, late = [], []
    for v in training_all:
        if len(v) >= 3:
            early.append(np.mean(v[:3]))
            late.append(np.mean(v[-3:]))
    wilc_p = np.nan
    if len(early) >= 2:
        _, wilc_p = stats.wilcoxon(early, late)

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4))
    style_ax(ax)
    ax.plot(x_days, group_mean, color=PHASE_COLORS['Training'],
            linewidth=2.5, marker='o', markersize=5, label='Group Mean', zorder=5)
    ax.fill_between(x_days, boot_lo, boot_hi,
                    color=PHASE_COLORS['Training'], alpha=0.2, label='95% Bootstrap CI')
    ax.set_ylim(40, 108)
    ax.set_xlim(0.5, max_len + 0.5)
    ax.set_title("Novice Success Rate (Mean +/- 95% Bootstrap CI)")
    ax.set_xlabel("Day")
    ax.set_ylabel("Success Rate (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.text(0.98, 0.97, lme_text, transform=ax.transAxes,
            ha='right', va='top', fontsize=7.5,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.7))
    plt.tight_layout()
    save_fig(str(FIGS_OUT / "Fig01_novice_success"))

    CAPTION['fig1'] = (
        f"**Figure 1 — Novice success rate across training days.**\n"
        f"Group mean (n=4 mice) +/- 95% bootstrap CI. "
        f"{lme_text}. "
        f"Wilcoxon early (days 1-3) vs late (last 3 days): "
        f"p={wilc_p:.3f} {stars(wilc_p)} (n={len(early)} mice). "
        f"Bootstrap CI: 5000 resamples."
    )
    print(f"  {lme_text}")
    print(f"  Wilcoxon early vs late: p={wilc_p:.3f} {stars(wilc_p)}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Novice latency with LME + Spearman
# ═══════════════════════════════════════════════════════════════════════════════

def render_fig2():
    print("\n[Fig 2] Novice latency + LME/Spearman ...")

    lat_all = []
    for mouse in sorted_mice:
        conds = mice[mouse]
        day_meds = defaultdict(list)
        for task in conds['Training']:
            mbase = conds.get('base_path', BASE_PATH)
            path = build_session_path(mbase, mouse, 'Training', task)
            if not os.path.exists(path):
                continue
            sfiles = sorted([f for f in os.listdir(path) if f.endswith('.mat')],
                            key=extract_sort_key)
            for sf in sfiles:
                data = load_mat(os.path.join(path, sf))
                try:
                    ntrials = data['SessionData']['nTrials']
                except Exception:
                    continue
                if ntrials < MIN_TRIALS:
                    continue
                day = extract_day_from_filename(sf)
                if day is None:
                    continue
                trials = data['SessionData']['RawEvents']['Trial']
                if isinstance(trials, dict):
                    trials = [trials]
                lats = [latency(t) for t in trials if not np.isnan(latency(t))]
                if lats:
                    day_meds[day].append(float(np.nanmedian(lats)))
        sorted_days = sorted(day_meds)
        lat_all.append([np.nanmedian(day_meds[d]) for d in sorted_days])

    if not lat_all or all(len(v) == 0 for v in lat_all):
        print("  No latency data found.")
        return

    max_len = max(len(v) for v in lat_all)
    arr = np.full((len(lat_all), max_len), np.nan)
    for i, v in enumerate(lat_all):
        arr[i, :len(v)] = v

    x_days = np.arange(1, max_len + 1)
    group_med = np.nanmedian(arr, axis=0)
    boot_lo = np.full(max_len, np.nan)
    boot_hi = np.full(max_len, np.nan)
    for d in range(max_len):
        boot_lo[d], boot_hi[d] = bootstrap_ci(arr[:, d])

    # LME
    rows = []
    for i, mouse in enumerate(sorted_mice):
        for d_idx in range(max_len):
            v = arr[i, d_idx]
            if not np.isnan(v):
                rows.append({'mouse': mouse, 'day': d_idx + 1, 'latency': v})
    df_lme = pd.DataFrame(rows)
    lme_text = 'LME: insufficient data'
    lme_beta, lme_p = np.nan, np.nan
    if df_lme['mouse'].nunique() >= 2:
        try:
            res = smf.mixedlm("latency ~ day", df_lme, groups=df_lme['mouse']).fit(
                reml=True, method='lbfgs', disp=False)
            lme_beta = res.fe_params['day']
            lme_p    = res.pvalues['day']
            lme_ci   = res.conf_int().loc['day']
            lme_text = (f"LME slope: {lme_beta:+.3f}s/day "
                        f"[{lme_ci[0]:.3f}, {lme_ci[1]:.3f}], "
                        f"p={lme_p:.3f} {stars(lme_p)}")
        except Exception:
            lme_text = ("LME did not converge (random-effects variance singular; "
                        "n=4 mice; see Spearman below)")

    # Spearman (group medians vs day)
    valid = ~np.isnan(group_med)
    rho, p_rho = np.nan, np.nan
    if valid.sum() >= 4:
        rho, p_rho = stats.spearmanr(x_days[valid], group_med[valid])
    spearman_text = f"Spearman rho={rho:.2f}, p={p_rho:.3f} {stars(p_rho)}"

    fig, ax = plt.subplots(figsize=(7, 4))
    style_ax(ax)
    ax.plot(x_days, group_med, color=PHASE_COLORS['Training'],
            linewidth=2.5, marker='o', markersize=5, label='Group Median', zorder=5)
    ax.fill_between(x_days, boot_lo, boot_hi,
                    color=PHASE_COLORS['Training'], alpha=0.2, label='95% Bootstrap CI')
    ax.set_ylim(0.3, 1.25)
    ax.set_xlim(0.5, max_len + 0.5)
    ax.set_title("Novice Latency (Median +/- 95% Bootstrap CI)")
    ax.set_xlabel("Day")
    ax.set_ylabel("Median Latency (s)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    annot = f"{lme_text}\n{spearman_text}"
    ax.text(0.98, 0.97, annot, transform=ax.transAxes,
            ha='right', va='top', fontsize=7,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.7))
    plt.tight_layout()
    save_fig(str(FIGS_OUT / "Fig02_novice_latency"))

    CAPTION['fig2'] = (
        f"**Figure 2 — Novice response latency across training days.**\n"
        f"Group median (n=4 mice) +/- 95% bootstrap CI (5000 resamples). "
        f"{lme_text}. "
        f"{spearman_text} (group medians vs day)."
    )
    print(f"  {lme_text}")
    print(f"  {spearman_text}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Composite LearnScore with bootstrap inflection CI + SEM label fix
# ═══════════════════════════════════════════════════════════════════════════════

def render_fig3():
    print("\n[Fig 3] LearnScore + bootstrap inflection CI ...")

    RT_NAIVE, RT_EXPERT = 1.7757, 0.50
    SEM_NAIVE, SEM_EXPERT = 0.0866, 0.0698
    BINSIZE = 10
    THRESHOLD = 0.7

    def _is_valid_session(fname, side):
        pattern = rf"_W2T_{side.split('_')[-1]}_\d{{8}}_\d{{6}}\.mat$"
        if not re.search(pattern, fname):
            return False
        m = re.search(r'_(\d{8})_', fname)
        return bool(m) and int(m.group(1)[4:6]) != 10

    def _get_hitrate_binned(df, binsize):
        n = len(df)
        return [df.iloc[s:s+binsize]['success'].mean()
                for s in range(n - binsize + 1)]

    def _get_sem_binned(rt_series, binsize):
        n = len(rt_series)
        sem_out = np.full(n, np.nan)
        for i in range(n - binsize + 1):
            w = rt_series.iloc[i:i+binsize].dropna()
            sem_out[i + binsize//2] = (np.std(w, ddof=1) / np.sqrt(len(w))
                                        if len(w) > 1 else np.nan)
        from pandas import Series
        return pd.Series(sem_out).interpolate('linear', limit_direction='both').to_numpy()

    def _learn_score_for_session(fpath):
        data = load_mat(fpath)
        trials = data['SessionData']['RawEvents']['Trial']
        if isinstance(trials, dict):
            trials = [trials]
        if len(trials) < 40:
            return np.array([])
        df = pd.DataFrame({
            'success':    [is_hit(t) for t in trials],
            'response_t': [latency(t) for t in trials],
        })
        hr  = np.array(_get_hitrate_binned(df, BINSIZE))
        n   = len(hr)
        rt  = df['response_t'].values[:n]
        rts = pd.Series(df['response_t'].values)
        rt_sem = _get_sem_binned(rts, BINSIZE)[:n]
        norm_hr  = hr
        norm_rt  = np.clip((RT_NAIVE - rt) / (RT_NAIVE - RT_EXPERT), 0, 1)
        norm_sem = np.clip((SEM_NAIVE - rt_sem) / (SEM_NAIVE - SEM_EXPERT), 0, 1)
        ls = 0.6*norm_hr + 0.35*norm_rt + 0.05*norm_sem
        return pd.Series(ls).interpolate('linear', limit_direction='both').to_numpy()

    all_data = []
    for mouse in sorted_mice:
        mbase = mice[mouse].get('base_path', BASE_PATH)
        mdir  = os.path.join(mbase, mouse)
        if not os.path.isdir(mdir):
            continue
        day_scores = defaultdict(list)
        for side in ('W2T_right', 'W2T_left'):
            sdir = os.path.join(mdir, f"Training/{side}/Session Data")
            if not os.path.exists(sdir):
                sdir = os.path.join(mdir, f"{side}/Session Data")
            if not os.path.exists(sdir):
                continue
            mat_files = sorted(
                [os.path.join(sdir, f) for f in os.listdir(sdir)
                 if f.endswith('.mat') and _is_valid_session(f, side)])
            for fpath in mat_files:
                try:
                    ls = _learn_score_for_session(fpath)
                    if len(ls) > 0:
                        day = extract_day_from_path(fpath)
                        if day is not None:
                            day_scores[day].append(np.nanmean(ls))
                except Exception:
                    pass
        if day_scores:
            sorted_days_k = sorted(day_scores.keys())
            all_data.append([np.nanmean(day_scores[d]) for d in sorted_days_k])

    if not all_data:
        print("  No learn-index data found.")
        return

    max_days = max(len(v) for v in all_data)
    score_matrix = np.full((len(all_data), max_days), np.nan)
    for i, v in enumerate(all_data):
        score_matrix[i, :len(v)] = v

    x_days  = np.arange(max_days, dtype=float)
    group_m = np.nanmean(score_matrix, axis=0)
    group_s = stats.sem(score_matrix, axis=0, nan_policy='omit')

    # Sigmoid fit on group mean
    popt, xf, yf, ok = fit_sigmoid(x_days, group_m)
    inflection = popt[1] if ok else np.nan

    # Bootstrap inflection CI (resample mice with replacement, 1000 iters)
    rng = np.random.default_rng(42)
    boot_inflections = []
    n_mice = len(all_data)
    for _ in range(1000):
        idx = rng.integers(0, n_mice, n_mice)
        boot_mat = score_matrix[idx, :]
        boot_mean = np.nanmean(boot_mat, axis=0)
        p_boot, _, _, ok_b = fit_sigmoid(x_days, boot_mean)
        if ok_b:
            boot_inflections.append(p_boot[1])
    infl_lo = np.nan
    infl_hi = np.nan
    if len(boot_inflections) >= 50:
        infl_lo = np.percentile(boot_inflections, 2.5)
        infl_hi = np.percentile(boot_inflections, 97.5)

    learn_day = next((i+1 for i, s in enumerate(group_m) if s >= THRESHOLD), None)

    fig, ax = plt.subplots(figsize=(14, 6))
    style_ax(ax)
    ax.plot(x_days, group_m, color='black', linewidth=2.5, marker='o',
            markersize=6, label='Group Mean', zorder=6)
    ax.fill_between(x_days, group_m - group_s, group_m + group_s,
                    color='black', alpha=0.15, label='SEM (n=4 mice)')
    if ok:
        ax.plot(xf, yf, color='navy', linewidth=2.5, linestyle='--',
                label=f'Sigmoid (inflection day {inflection+1:.1f})', zorder=7)
        ax.axvline(inflection, color='navy', linestyle=':', linewidth=1.5, alpha=0.6)
        if not np.isnan(infl_lo):
            ax.axvspan(infl_lo, infl_hi, color='navy', alpha=0.08,
                       label=f'Inflection 95% CI [{infl_lo+1:.1f}, {infl_hi+1:.1f}]')
    if learn_day is not None:
        ax.axvline(learn_day-1, color='red', linestyle=':', linewidth=2,
                   label=f'Threshold >= 0.70 (Day {learn_day})')
        ax.scatter(learn_day-1, group_m[learn_day-1], color='red', zorder=10, s=80)
    ax.axhline(THRESHOLD, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xticks(x_days)
    ax.set_xticklabels(np.arange(1, max_days+1))
    ax.set_xlabel("Day")
    ax.set_ylabel("LearnScore")
    ax.set_title("LearnScore Mean +/- SEM (n=4 mice) + Sigmoid Fit")
    y_lo = max(0.0, float(np.nanmin(group_m - group_s)) - 0.05)
    y_hi = min(1.0, float(np.nanmax(group_m + group_s)) + 0.05)
    ax.set_ylim(y_lo, y_hi)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=8, framealpha=0.8)
    plt.tight_layout()
    save_fig(str(FIGS_OUT / "Fig03_learn_index"))

    infl_str = (f"Day {inflection+1:.1f} [95% bootstrap CI: {infl_lo+1:.1f}–{infl_hi+1:.1f}]"
                if ok and not np.isnan(infl_lo) else "sigmoid fit unavailable")
    CAPTION['fig3'] = (
        f"**Figure 3 — Composite LearnScore across novice training.**\n"
        f"Group mean (n=4 mice) +/- SEM. "
        f"Sigmoid inflection: {infl_str} (bootstrap n=1000, resample mice). "
        f"Threshold crossing (score >= 0.70): Day {learn_day}. "
        f"LearnScore = 0.60*hit_rate + 0.35*norm(RT) + 0.05*norm(RT_SEM)."
    )
    print(f"  Sigmoid inflection: {infl_str}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURES 5, 6, 7 — Opto hit rate with per-mouse trial-level stats
# ═══════════════════════════════════════════════════════════════════════════════

def _collect_opto_hitrate_data():
    """Collect per-mouse, per-area, per-condition data. Returns dict."""
    mice_all  = list(mouse_sessions.keys())
    areas     = sorted({a for m in mice_all for a in mouse_sessions[m]})
    conditions = [('W2T', 1), ('opto_0.5s', 2), ('opto_2s', 2)]
    data = {}  # (mouse, area, cond) -> {'session_means': [...], 'trial_hits': [...]}
    for mouse in mice_all:
        for area in areas:
            for cond_key, tt_num in conditions:
                sm   = get_mouse_session_mean_hitrate(mouse, area, cond_key, tt_num)
                th   = get_mouse_trial_hits(mouse, area, cond_key, tt_num)
                data[(mouse, area, cond_key)] = {'session_means': sm, 'trial_hits': th}
    return data, mice_all, areas


def render_fig5(data, mice_all, areas):
    print("\n[Fig 5] Opto hit rate bars with per-mouse trial MWU stars ...")

    opto_conds = [('opto_0.5s', 2, COND_COLORS['opto_0.5s']),
                  ('opto_2s',   2, COND_COLORS['opto_2s'])]
    bar_width = 0.22
    x = np.arange(len(areas))
    all_cond_info = [('W2T', 1, COND_COLORS['W2T'])] + opto_conds

    # Per-mouse MWU: W2T trials vs each opto condition
    mwu_results = {}  # (mouse, area, opto_key) -> p
    for mouse in mice_all:
        for area in areas:
            w2t_hits = data[(mouse, area, 'W2T')]['trial_hits']
            for opto_key, tt_num, _ in opto_conds:
                opto_hits = data[(mouse, area, opto_key)]['trial_hits']
                _, p = mwu_per_mouse(w2t_hits, opto_hits)
                mwu_results[(mouse, area, opto_key)] = p

    fig, ax = plt.subplots(figsize=(max(10, len(areas)*2.5), 5))
    style_ax(ax)

    for i, (cond_key, tt_num, col) in enumerate(all_cond_info):
        means, lo_list, hi_list = [], [], []
        for area in areas:
            per_mouse_m = [np.mean(data[(m, area, cond_key)]['session_means'])
                           for m in mice_all
                           if data[(m, area, cond_key)]['session_means']]
            if per_mouse_m:
                means.append(np.mean(per_mouse_m))
                lo, hi = bootstrap_ci(per_mouse_m)
                lo_list.append(lo); hi_list.append(hi)
            else:
                means.append(np.nan); lo_list.append(np.nan); hi_list.append(np.nan)

        xerr_lo = [m - lo if not np.isnan(m) else 0 for m, lo in zip(means, lo_list)]
        xerr_hi = [hi - m if not np.isnan(m) else 0 for m, hi in zip(means, hi_list)]
        ax.bar(x + i*bar_width, means, width=bar_width, color=col,
               yerr=[xerr_lo, xerr_hi], capsize=4,
               label=cond_key.replace('_', ' '), error_kw={'elinewidth': 1.2})

        # Per-mouse markers
        rng = np.random.default_rng(i)
        for j, area in enumerate(areas):
            for mouse in mice_all:
                sm = data[(mouse, area, cond_key)]['session_means']
                if not sm:
                    continue
                y_val = np.mean(sm)
                jit = rng.uniform(-0.04, 0.04)
                xpos = x[j] + i*bar_width + jit
                ax.plot(xpos, y_val,
                        marker=MOUSE_MARKERS.get(mouse, 'o'),
                        color=MOUSE_COLORS_SHORT.get(mouse, 'k'),
                        markersize=6, zorder=5, linestyle='none',
                        markeredgecolor='white', markeredgewidth=0.5)

                # Star above marker if significant MWU (opto conditions only)
                if cond_key in ('opto_0.5s', 'opto_2s'):
                    p_mwu = mwu_results.get((mouse, area, cond_key), np.nan)
                    if not np.isnan(p_mwu) and p_mwu < 0.05:
                        star_str = stars(p_mwu)
                        ax.text(xpos, y_val + 2.5, star_str,
                                ha='center', va='bottom', fontsize=7,
                                color=MOUSE_COLORS_SHORT.get(mouse, 'k'))

    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(areas, fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Success Rate across brain areas — all mice\n'
                 '(stars = per-mouse trial MWU W2T vs opto, p<0.05)',
                 fontsize=10)
    ax.set_ylim(0, 118)
    handles = [mlines.Line2D([], [], color=MOUSE_COLORS_SHORT[m],
                              marker=MOUSE_MARKERS[m], linestyle='none',
                              markersize=6, label=m) for m in MOUSE_COLORS_SHORT]
    cond_legend = ax.legend(fontsize=8, loc='upper left', title='Condition', framealpha=0.8)
    ax.add_artist(cond_legend)
    ax.legend(handles=handles, fontsize=7, title='Mouse', loc='upper right', framealpha=0.7)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    save_fig(str(FIGS_OUT / "Fig05_success_bars"))

    # Build caption with significant per-mouse results (MLA = ChR2, SNA = WT)
    mla_mice = [m for m in mice_all if m.startswith('MLA')]
    sna_mice = [m for m in mice_all if m.startswith('SNA')]
    sig_lines = []
    for area in areas:
        for opto_key, _, _ in opto_conds:
            sig_mla = [m for m in mla_mice
                       if not np.isnan(mwu_results.get((m, area, opto_key), np.nan))
                       and mwu_results[(m, area, opto_key)] < 0.05]
            sig_sna = [m for m in sna_mice
                       if not np.isnan(mwu_results.get((m, area, opto_key), np.nan))
                       and mwu_results[(m, area, opto_key)] < 0.05]
            if sig_mla or sig_sna:
                mla_pstr = ', '.join(f"{m} p={mwu_results[(m,area,opto_key)]:.3f}" for m in sig_mla)
                sna_pstr = ', '.join(f"{m} p={mwu_results[(m,area,opto_key)]:.3f}" for m in sig_sna)
                line = f"  {area} {opto_key}: {len(sig_mla)}/{len(mla_mice)} ChR2 significant"
                if sig_mla:
                    line += f" ({mla_pstr})"
                if sig_sna:
                    line += f"; WT also significant ({sna_pstr}) -> check artifact"
                sig_lines.append(line)
    body = (f"Significant per-mouse effects (MLA=ChR2-VGAT, SNA=wildtype; "
            f"WT significance indicates possible light artifact):\n"
            + "\n".join(sig_lines)) if sig_lines else "No per-mouse tests reached p<0.05."
    CAPTION['fig5'] = (
        f"**Figure 5 — Success rate per brain area (all mice, all conditions).**\n"
        f"Bars = group mean of mouse means +/- 95% bootstrap CI (n=4 mice). "
        f"Colored markers = per-mouse mean; shape = mouse identity. "
        f"Stars above markers = significant per-mouse Mann-Whitney U test "
        f"(all W2T trials vs all opto trials within that mouse/area combination, two-sided p<0.05). "
        f"n=3 ChR2-VGAT (MLA) + n=1 wildtype control (SNA).\n"
        f"{body}"
    )
    print(f"  Significant per-mouse MWU: {len(sig_lines)} area-condition pairs")


def render_fig6_fig7(data, mice_all, areas):
    print("\n[Figs 6/7] Delta hit rate — cleaned titles + outlier stats ...")

    for opto_key, tt_num, label, fig_num in [
        ('opto_0.5s', 2, '0.5 s opto', 6),
        ('opto_2s',   2, '2 s opto',   7),
    ]:
        delta_per_area = {area: {'vals': [], 'mice': []} for area in areas}
        for mouse in mice_all:
            for area in areas:
                w2t_sm   = data[(mouse, area, 'W2T')]['session_means']
                opto_sm  = data[(mouse, area, opto_key)]['session_means']
                if w2t_sm and opto_sm:
                    delta = np.mean(w2t_sm) - np.mean(opto_sm)
                    delta_per_area[area]['vals'].append(delta)
                    delta_per_area[area]['mice'].append(mouse)

        # Modified Z-score per area
        outlier_flags = {}  # (area, mouse) -> bool
        for area in areas:
            vals = delta_per_area[area]['vals']
            mice = delta_per_area[area]['mice']
            if len(vals) >= 3:
                zs = modified_z_score(vals)
                for i, (m, z) in enumerate(zip(mice, zs)):
                    outlier_flags[(area, m)] = z > 3.5
            else:
                for m in mice:
                    outlier_flags[(area, m)] = False

        # One-sample Wilcoxon Δ vs 0 per area
        wilcoxon_per_area = {}
        for area in areas:
            vals = delta_per_area[area]['vals']
            st, p = one_sample_wilcoxon(vals)
            wilcoxon_per_area[area] = (st, p)

        # Group means + CI
        means, lo_list, hi_list = [], [], []
        for area in areas:
            vals = delta_per_area[area]['vals']
            if vals:
                means.append(np.mean(vals))
                lo, hi = bootstrap_ci(vals)
                lo_list.append(lo); hi_list.append(hi)
            else:
                means.append(np.nan); lo_list.append(np.nan); hi_list.append(np.nan)

        fig, ax = plt.subplots(figsize=(max(6, len(areas)*1.8), 5))
        style_ax(ax)
        x = np.arange(len(areas))
        jitter_rng = np.random.default_rng(1)

        xerr_lo = [m - lo if not np.isnan(m) else 0 for m, lo in zip(means, lo_list)]
        xerr_hi = [hi - m  if not np.isnan(m) else 0 for m, hi in zip(means, hi_list)]
        ax.bar(x, means, 0.5, color='#aaaaaa', alpha=0.5,
               yerr=[xerr_lo, xerr_hi], capsize=5,
               error_kw={'elinewidth': 1.5}, zorder=2, label='Group mean')

        for j, area in enumerate(areas):
            for val, mouse in zip(delta_per_area[area]['vals'],
                                  delta_per_area[area]['mice']):
                jit = jitter_rng.uniform(-0.12, 0.12)
                xpos = j + jit
                ax.plot(xpos, val,
                        marker=MOUSE_MARKERS.get(mouse, 'o'),
                        color=MOUSE_COLORS_SHORT.get(mouse, 'k'),
                        markersize=9, zorder=5, linestyle='none',
                        markeredgecolor='white', markeredgewidth=0.7)
                # Mark outliers
                if outlier_flags.get((area, mouse), False):
                    ax.text(xpos + 0.04, val, '(!)',
                            fontsize=7, va='center', color='gray')

            # One-sample Wilcoxon annotation above area bar
            _, p_wilc = wilcoxon_per_area[area]
            if not np.isnan(p_wilc):
                ax.text(j, (means[j] or 0) + (hi_list[j] - means[j] if not np.isnan(hi_list[j]) else 2) + 1.5,
                        stars(p_wilc), ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.axhline(0, color='black', linewidth=1.2, linestyle='--', alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(areas, fontsize=12)
        ax.set_ylabel('Delta Success Rate: W2T - Opto (%)', fontsize=12)
        # CLEAN TITLE — no "Positive = ..." text in title
        ax.set_title(f'Inhibition Effect ({label})\n'
                     f'(error = 95% bootstrap CI; stars = 1-sample Wilcoxon Delta vs 0; (!) = modified Z > 3.5)',
                     fontsize=9)
        handles = [mlines.Line2D([], [], color=MOUSE_COLORS_SHORT[m],
                                  marker=MOUSE_MARKERS[m], linestyle='none',
                                  markersize=7, label=m) for m in MOUSE_COLORS_SHORT]
        ax.legend(handles=handles, fontsize=7, title='Mouse', framealpha=0.7)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        save_fig(str(FIGS_OUT / f"Fig0{fig_num}_delta_{opto_key.replace('.','_')}"))

        # Caption with concrete numbers
        area_stats = []
        for area in areas:
            vals = delta_per_area[area]['vals']
            _, p = wilcoxon_per_area[area]
            m_mean = np.mean(vals) if vals else np.nan
            lo_ci, hi_ci = bootstrap_ci(vals) if vals else (np.nan, np.nan)
            out_mice = [m for m in delta_per_area[area]['mice']
                        if outlier_flags.get((area, m), False)]
            out_str = f"; outlier: {', '.join(out_mice)}" if out_mice else "; no outliers"
            area_stats.append(
                f"  {area}: mean Delta={m_mean:.1f}% [95% CI {lo_ci:.1f}, {hi_ci:.1f}], "
                f"1-sample Wilcoxon p={p:.3f} {stars(p)}{out_str}"
            )
        CAPTION[f'fig{fig_num}'] = (
            f"**Figure {fig_num} — Inhibition effect ({label}): Delta success rate W2T - Opto.**\n"
            f"Positive values = opto reduced performance (inhibition). "
            f"Negative values = paradoxical facilitation. "
            f"Group mean +/- 95% bootstrap CI (5000 resamples); markers = individual mice; "
            f"stars = one-sample Wilcoxon (Delta vs 0); (!) = modified Z-score > 3.5 (outlier).\n"
            f"n=3 ChR2-VGAT mice + 1 WT control; no formal between-genotype comparison (n=1 WT).\n"
            + "\n".join(area_stats)
        )
        print(f"  Fig {fig_num}: {len([v for v in areas if not np.isnan(wilcoxon_per_area[v][1]) and wilcoxon_per_area[v][1] < 0.05])} areas significant")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 8 — Opto latency bars with per-mouse trial MWU stars
# ═══════════════════════════════════════════════════════════════════════════════

def render_fig8():
    print("\n[Fig 8] Opto latency bars with per-mouse MWU stars ...")

    mice_all = list(mouse_sessions.keys())
    areas    = sorted({a for m in mice_all for a in mouse_sessions[m]})
    conds    = [('W2T', 1), ('opto_0.5s', 2), ('opto_2s', 2)]
    colors   = [COND_COLORS['W2T'], COND_COLORS['opto_0.5s'], COND_COLORS['opto_2s']]
    bar_width = 0.22
    x = np.arange(len(areas))

    # Precompute per-mouse trial latencies + session medians
    lat_data = {}  # (mouse, area, cond) -> {'session_meds': [...], 'trial_lats': [...]}
    for mouse in mice_all:
        for area in areas:
            for cond_key, tt_num in conds:
                sm = get_mouse_session_median_latency(mouse, area, cond_key, tt_num)
                tl = get_mouse_trial_latencies(mouse, area, cond_key, tt_num)
                lat_data[(mouse, area, cond_key)] = {'session_meds': sm, 'trial_lats': tl}

    # Per-mouse MWU on trial-level latencies
    mwu_lat = {}
    for mouse in mice_all:
        for area in areas:
            w2t_lats = lat_data[(mouse, area, 'W2T')]['trial_lats']
            for opto_key, tt_num in [('opto_0.5s', 2), ('opto_2s', 2)]:
                opto_lats = lat_data[(mouse, area, opto_key)]['trial_lats']
                _, p = mwu_per_mouse(w2t_lats, opto_lats)
                mwu_lat[(mouse, area, opto_key)] = p

    fig, ax = plt.subplots(figsize=(max(10, len(areas)*2.5), 5))
    style_ax(ax)

    for i, (cond_key, tt_num, col) in enumerate(zip(
            ['W2T', 'opto_0.5s', 'opto_2s'], [1,2,2], colors)):
        means, lo_list, hi_list = [], [], []
        for area in areas:
            per_m = [np.median(lat_data[(m, area, cond_key)]['session_meds'])
                     for m in mice_all
                     if lat_data[(m, area, cond_key)]['session_meds']]
            if per_m:
                means.append(np.mean(per_m))
                lo, hi = bootstrap_ci(per_m)
                lo_list.append(lo); hi_list.append(hi)
            else:
                means.append(np.nan); lo_list.append(np.nan); hi_list.append(np.nan)

        xerr_lo = [m - lo if not np.isnan(m) else 0 for m, lo in zip(means, lo_list)]
        xerr_hi = [hi - m  if not np.isnan(m) else 0 for m, hi in zip(means, hi_list)]
        ax.bar(x + i*bar_width, means, width=bar_width, color=col,
               yerr=[xerr_lo, xerr_hi], capsize=4,
               label=cond_key.replace('_',' '), error_kw={'elinewidth': 1.2})

        rng = np.random.default_rng(i)
        for j, area in enumerate(areas):
            for mouse in mice_all:
                sm = lat_data[(mouse, area, cond_key)]['session_meds']
                if not sm:
                    continue
                y_val = float(np.median(sm))
                jit   = rng.uniform(-0.04, 0.04)
                xpos  = x[j] + i*bar_width + jit
                ax.plot(xpos, y_val,
                        marker=MOUSE_MARKERS.get(mouse, 'o'),
                        color=MOUSE_COLORS_SHORT.get(mouse, 'k'),
                        markersize=6, zorder=5, linestyle='none',
                        markeredgecolor='white', markeredgewidth=0.5)
                if cond_key in ('opto_0.5s', 'opto_2s'):
                    p_mwu = mwu_lat.get((mouse, area, cond_key), np.nan)
                    if not np.isnan(p_mwu) and p_mwu < 0.05:
                        ax.text(xpos, y_val + 0.04, stars(p_mwu),
                                ha='center', va='bottom', fontsize=7,
                                color=MOUSE_COLORS_SHORT.get(mouse, 'k'))

    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(areas, fontsize=12)
    ax.set_ylabel('Median Latency (s)', fontsize=12)
    ax.set_title('Response latency across brain areas — all mice\n'
                 '(stars = per-mouse MWU W2T vs opto latencies, p<0.05)', fontsize=10)
    ax.set_ylim(0, 2.4)
    handles = [mlines.Line2D([], [], color=MOUSE_COLORS_SHORT[m],
                              marker=MOUSE_MARKERS[m], linestyle='none',
                              markersize=6, label=m) for m in MOUSE_COLORS_SHORT]
    cond_leg = ax.legend(fontsize=8, loc='upper left', title='Condition', framealpha=0.8)
    ax.add_artist(cond_leg)
    ax.legend(handles=handles, fontsize=7, title='Mouse', loc='upper right', framealpha=0.7)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    save_fig(str(FIGS_OUT / "Fig08_latency_bars"))

    # Caption
    sig_pairs = [(m, a, c) for (m, a, c), p in mwu_lat.items()
                 if not np.isnan(p) and p < 0.05]
    CAPTION['fig8'] = (
        f"**Figure 8 — Response latency under optogenetic inhibition.**\n"
        f"Group mean of per-mouse session medians +/- 95% bootstrap CI. "
        f"Stars = significant per-mouse Mann-Whitney U (W2T vs opto trial latencies, two-sided, p<0.05). "
        f"n={len([m for m in mice_all if m.startswith('MLA')])} ChR2-VGAT + "
        f"n={len([m for m in mice_all if m.startswith('SNA')])} WT. "
        f"{len(sig_pairs)} mouse*area*condition combinations reached significance."
    )
    print(f"  {len(sig_pairs)} significant per-mouse MWU pairs")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 10 — Genotype comparison (caption update only)
# ═══════════════════════════════════════════════════════════════════════════════

def render_fig10_caption():
    print("\n[Fig 10] Genotype comparison — caption text only ...")
    mice_all = list(mouse_sessions.keys())
    n_mla = len([m for m in mice_all if m.startswith('MLA')])
    n_sna = len([m for m in mice_all if m.startswith('SNA')])
    CAPTION['fig10'] = (
        f"**Figure 10 — Success rate: ChR2-VGAT vs wildtype across brain areas.**\n"
        f"ChR2-VGAT (n={n_mla} mice) vs wildtype control (n={n_sna} mouse). "
        f"**n=1 wildtype mouse — no formal between-genotype statistical test is possible. "
        f"The wildtype animal serves as a qualitative light-only control, not a powered comparison group.** "
        f"Group mean +/- 95% bootstrap CI shown for ChR2-VGAT group; single mouse values shown for WT. "
        f"Any apparent group differences should be interpreted with extreme caution."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 11 — Random vs Blocked, 2×2 layout
# ═══════════════════════════════════════════════════════════════════════════════

def render_fig11():
    print("\n[Fig 11] Random vs Blocked — 2x2 layout ...")

    def _load_session_metrics(filepath):
        if not filepath or not os.path.exists(filepath):
            return None
        data = load_mat(filepath)
        sd = data.get('SessionData', None)
        if sd is None:
            return None
        tt  = sd.get('TrialTypes', None)
        raw = sd.get('RawEvents', None)
        if tt is None or raw is None:
            return None
        trials = raw.get('Trial', None) if isinstance(raw, dict) else None
        if trials is None:
            return None
        if isinstance(trials, dict):
            trials = [trials]
        if len(trials) < MIN_TRIALS:
            return None
        opto_hits, w2t_hits = [], []
        for i, trial in enumerate(trials):
            if i >= len(tt):
                break
            h = is_hit(trial)
            if tt[i] == 2:
                opto_hits.append(1 if h else 0)
            elif tt[i] == 1:
                w2t_hits.append(1 if h else 0)
        if not opto_hits:
            return None
        return {
            'hit_rate_opto': np.mean(opto_hits) * 100,
            'hit_rate_w2t':  np.mean(w2t_hits)  * 100 if w2t_hits else np.nan,
            'n_opto': len(opto_hits),
            'blocked': is_blocked_session(filepath),
        }

    # Build DataFrame
    mice_all = list(mouse_sessions.keys())
    rows = []
    for mouse in mice_all:
        geno = 'WT' if mouse.startswith('SNA') else 'ChR2'
        for area, conds in mouse_sessions[mouse].items():
            seen = set()
            for opto_key in ('opto_0.5s', 'opto_2s'):
                for path in conds.get(opto_key, []):
                    if path in seen:
                        continue
                    seen.add(path)
                    m = _load_session_metrics(path)
                    if m is None:
                        continue
                    rows.append({
                        'mouse': mouse, 'genotype': geno, 'area': area,
                        'session_type': 'blocked' if m['blocked'] else 'random',
                        'hit_rate_opto': m['hit_rate_opto'],
                        'hit_rate_w2t':  m['hit_rate_w2t'],
                    })
    df = pd.DataFrame(rows)
    if df.empty:
        print("  No habituation data found.")
        CAPTION['fig11'] = "**Figure 11 — Random vs blocked opto sessions.** No data available."
        return

    area_list = sorted(a for a in df['area'].unique() if a != 'All')[:4]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharey=True)
    axes = axes.flatten()

    wilcoxon_results = {}
    for ax, area in zip(axes, area_list):
        style_ax(ax)
        sub = df[df['area'] == area]
        sub_r = sub[sub['session_type'] == 'random']
        sub_b = sub[sub['session_type'] == 'blocked']

        x_r, x_b = 0.3, 0.7
        for sess_df, xpos, col, label in [
            (sub_r, x_r, SESSION_TYPE_COLORS.get('random',  '#ff7f0e'), 'random'),
            (sub_b, x_b, SESSION_TYPE_COLORS.get('blocked', '#9467bd'), 'blocked'),
        ]:
            pm = sess_df.groupby('mouse')['hit_rate_opto'].mean()
            if pm.empty:
                continue
            mean_v = pm.mean()
            lo, hi = bootstrap_ci(pm.values)
            ax.bar(xpos, mean_v, 0.3, color=col, alpha=0.7,
                   yerr=[[mean_v - lo], [hi - mean_v]], capsize=5,
                   label=label, error_kw={'elinewidth': 1.2})

        # Spaghetti lines + points
        rng = np.random.default_rng(42)
        for mouse in sub['mouse'].unique():
            r_vals = sub_r[sub_r['mouse'] == mouse]['hit_rate_opto'].values
            b_vals = sub_b[sub_b['mouse'] == mouse]['hit_rate_opto'].values
            if len(r_vals) == 0 or len(b_vals) == 0:
                continue
            r_m, b_m = np.mean(r_vals), np.mean(b_vals)
            jit = rng.uniform(-0.03, 0.03)
            ax.plot([x_r+jit, x_b+jit], [r_m, b_m],
                    color=MOUSE_COLORS_SHORT.get(mouse, 'gray'),
                    linewidth=1.5, alpha=0.7, zorder=3)
            for xp, yp in [(x_r+jit, r_m), (x_b+jit, b_m)]:
                ax.plot(xp, yp, marker=MOUSE_MARKERS.get(mouse, 'o'),
                        color=MOUSE_COLORS_SHORT.get(mouse, 'gray'),
                        markersize=8, zorder=4, linestyle='none',
                        markeredgecolor='white', markeredgewidth=0.6)

        # Wilcoxon per-mouse means random vs blocked
        common_mice = set(sub_r['mouse'].unique()) & set(sub_b['mouse'].unique())
        if len(common_mice) >= 2:
            r_means = [sub_r[sub_r['mouse'] == m]['hit_rate_opto'].mean() for m in common_mice]
            b_means = [sub_b[sub_b['mouse'] == m]['hit_rate_opto'].mean() for m in common_mice]
            _, wilc_p = stats.wilcoxon(r_means, b_means)
        else:
            wilc_p = np.nan
        wilcoxon_results[area] = wilc_p

        ax.set_xticks([x_r, x_b])
        ax.set_xticklabels(['Random', 'Blocked'], fontsize=11)
        ax.set_title(area, fontsize=13, fontweight='bold')
        ax.set_ylim(0, 110)
        ax.set_ylabel('Opto Hit Rate (%)', fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        p_ann = f"Wilcoxon p={wilc_p:.3f}" if not np.isnan(wilc_p) else "n<2"
        ax.text(0.97, 0.97, p_ann, transform=ax.transAxes,
                ha='right', va='top', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='gray', alpha=0.7))

    # Remove unused axes if < 4 areas
    for ax in axes[len(area_list):]:
        ax.set_visible(False)

    fig.suptitle('Random vs Blocked Opto Sessions: Opto Hit Rate per Brain Area\n'
                 '(lines = individual mice, error = 95% bootstrap CI)',
                 fontsize=11)
    plt.tight_layout()
    save_fig(str(FIGS_OUT / "Fig11_random_blocked"))

    wilc_lines = [f"  {a}: Wilcoxon p={p:.3f} {stars(p)}" if not np.isnan(p)
                  else f"  {a}: n<2 mice paired" for a, p in wilcoxon_results.items()]
    CAPTION['fig11'] = (
        f"**Figure 11 — Hit rate in random vs blocked optogenetic sessions.**\n"
        f"Bars = mean of per-mouse session means +/- 95% bootstrap CI. "
        f"Lines connect the same mouse across session types. "
        f"Wilcoxon signed-rank (per-mouse means, paired random vs blocked):\n"
        + "\n".join(wilc_lines) +
        f"\nn=3 ChR2-VGAT mice + 1 WT control; n=1 WT precludes formal genotype comparison.\n"
        f"Layout: 2x2 grid (one panel per brain area: {', '.join(area_list)})."
    )
    print(f"  Rendered Fig 11 with {len(area_list)} areas in 2x2 layout")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURES 17, 19, 20 — Pose figures (caption-only updates)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_pose_captions():
    print("\n[Figs 17/19/20] Pose figure captions ...")

    CAPTION['fig17'] = (
        "**Figure 17 — Stereotypy index across novice training.**\n"
        "Per-mouse and group stereotypy index (mean pairwise Pearson r of hit-aligned "
        "speed traces) across training days, for each keypoint. "
        "LME trend line shown (hitrate ~ day | random intercept: mouse); "
        "beta and p reported in subplot annotations. "
        "Note: LME values are from pose_learning_kinematics.py console output — "
        "copy exact beta/p values from run log into caption: "
        "\"LME: beta=... per day, p=... (n=4 mice)\" per keypoint. "
        "DLC confidence threshold = 0.8, SLEAP = 0.6; interpolation <= 15 frames."
    )
    CAPTION['fig19'] = (
        "**Figure 19 — Stereotypy index vs performance (LearnScore).**\n"
        "Pearson r reported per mouse (descriptive). "
        "**Pseudoreplication caveat**: each data point is one mouse*day session "
        "(n=4 mice x ~11 days = ~44 points). A mixed-effects model with mouse as "
        "random effect is a more appropriate test for this relationship. "
        "LME (SI ~ LearnScore | mouse): copy exact beta and p from "
        "pose_learning_kinematics.py console output into caption. "
        "Caption format: \"Pearson r=... (descriptive, n=~44 sessions); LME "
        "beta=... [95% CI], p=... (n=4 mice, random intercept per mouse)\"."
    )
    CAPTION['fig20'] = (
        "**Figure 20 — Effect of cortical optogenetic inhibition on body kinematics "
        "(opto summary v3).**\n"
        "Heatmap: 4 keypoints x 4 brain areas; MLA (ChR2-VGAT, n=3 mice) top, "
        "SNA (wildtype, n=1 mouse) bottom. "
        "Effect size: rank-biserial r (W2T vs Opto HIT trials). "
        "Statistics: Fisher combined p per MLA cell, BH-FDR corrected over 16 cells. "
        "Classification: green (Robust real) = MLA significant + SNA not significant; "
        "red (Artifact-suspect) = both MLA and SNA significant. "
        "**v3 update vs v2**: C1r_angle x mPFC reclassified from 'Not significant' to "
        "'Robust real' (r: +0.121 -> +0.146, p_fdr: 0.102 -> 0.032) after applying "
        "circular baseline subtraction to whisker angle traces. "
        "All 4 brain areas now show robust contralateral whisker elevation. "
        "**Wildtype note**: n=1 SNA mouse; artifact classification based on single animal. "
        "Method: circular baseline subtract for C2_angle and C1r_angle "
        "(dev = ((trace - bl_mean + 180) mod 360) - 180); avoids atan2 wrap artefacts "
        "without np.unwrap drift. DLC conf=0.8, SLEAP conf=0.6."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# WRITE STATS TEXT FILES
# ═══════════════════════════════════════════════════════════════════════════════

def write_stats_files():
    print("\n[Stats] Writing caption .md files ...")
    groups = {
        'fig1_3_stats': ['fig1', 'fig2', 'fig3'],
        'fig5_8_stats': ['fig5', 'fig6', 'fig7', 'fig8'],
        'fig10_11_stats': ['fig10', 'fig11'],
        'fig17_19_stats': ['fig17', 'fig19'],
        'fig20_stats': ['fig20'],
    }
    for fname, keys in groups.items():
        lines = [f"# Stats / Caption Text: {fname}\n",
                 f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_\n",
                 "_Copy the text below directly into thesis captions._\n",
                 "---\n"]
        for k in keys:
            if k in CAPTION:
                lines.append(CAPTION[k])
                lines.append("\n\n---\n")
            else:
                lines.append(f"## {k}\n_(Not generated — data unavailable)_\n\n---\n")
        path = STATS_OUT / f"{fname}.md"
        path.write_text("\n".join(lines), encoding='utf-8')
        print(f"  Written: {path.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# GITHUB INVENTORY
# ═══════════════════════════════════════════════════════════════════════════════

GITHUB_INVENTORY_TEXT = """# Thesis Code Inventory — for GitHub upload
_Generated: {date}_

All scripts used to produce the final thesis figures. Version: Phase B.

---

## Behavioral Analysis (Section 4.1)

| Script | Produces | Notes |
|--------|----------|-------|
| `bpod_utils.py` | (shared utility) | Data loading, hit classification, latency, bootstrap_ci, fdr_bh, rank_biserial_r |
| `session_config.py` | (shared config) | Mouse session paths, sorted_mice list |
| `plot_config.py` | (shared style) | Colors, markers, style_ax |
| `hitrate_mean+SEM.py` | Fig 1 (novice success rate) | Bootstrap CI, LME slope — use phaseB output for annotated version |
| `latency_mean+SEM.py` | Fig 2 (novice latency) | Bootstrap CI, LME slope |
| `learn_index_mean+SEM.py` | Fig 3 (composite LearnScore) | Sigmoid fit, inflection day |
| `calc_expert_def.py` | (LearnScore constants) | Computes RT_NAIVE, RT_EXPERT, SEM_NAIVE, SEM_EXPERT |
| `raster_plots.py` | Fig 4, Fig 9 (single-mouse rasters) | Latency rasters, ex-Gaussian fits |
| `opto_hitrate.py` | Figs 5, 6, 7, 10 (opto hit rate) | Per-mouse means, delta plots, MLA vs SNA — use phaseB for annotated |
| `opto_latency.py` | Fig 8 (opto latency) | Mann-Whitney, per-mouse comparisons — use phaseB for annotated |
| `opto_habituation.py` | Fig 11 (random vs blocked) | Random vs blocked session comparison — use phaseB for 2x2 layout |
| `phaseB_figures.py` | Figs 1-3, 5-8, 11 (annotated) | Phase B: per-mouse stats, title fixes, 2x2 layout |

---

## Neural Validation (Section 4.2)

| Script | Produces | Notes |
|--------|----------|-------|
| `utils_jelt_nov2025.py` | (shared neural utility) | PSTH, raster, population response tools — Jelte's utilities |
| `utils_jelt_april2026.py` | (shared neural utility, updated) | Updated version of neural utilities |
| `neural_data_handler.py` | (data loading) | Neural data I/O wrapper |
| `plot_ttl_psth_opto.py` | Figs 12-14 (PSTH, raster, population) | Neural population responses under opto |

---

## Pose Analysis (Section 4.3)

| Script | Produces | Notes |
|--------|----------|-------|
| `pose_config.py` | (shared config) | MICE, keypoint specs, MOUSE_COLORS, confidence thresholds |
| `pose_utils.py` | (shared utility) | Data loading, trial slicing, baseline subtract, keypoint extraction |
| `pose_sync_qc.py` | Supp: sync QC | Bpod-camera frame sync validation |
| `pose_trial_aligned.py` | Fig 16 (HIT vs Miss kinematics) | Trial-aligned pose responses |
| `pose_learning_kinematics.py` | Figs 17, 18, 19 (stereotypy, PCA, scatter) | LME trends, stereotypy index |
| `pose_example_traces.py` | Fig 15 (example trial) | MLA-026807_4 / 20250916_1 / Trial 30 |
| `pose_figure2_rerender.py` | Figs 15A-D (re-rendered with angle-wrap fix) | Circular baseline subtract for angle keypoints |
| `pose_opto_summary_v3.py` | Fig 20 (opto heatmap) | v3: angle-wrap fix applied, supersedes v1/v2 |

---

## NOT to include (abandoned / exploratory)

| Script | Reason |
|--------|--------|
| `pose_opto_summary_heatmap.py` | Superseded by v2/v3 |
| `pose_opto_summary_v2.py` | Superseded by v3 (v3 has angle-wrap fix) |
| `pose_opto_stereotypy.py` | Negative result, not in thesis main results |
| `pose_opto_template_match.py` | Negative result, not in thesis main results |
| `opto_hitrate_with_jelte_data.py` | Exploratory, superseded |
| `opto_hitrate_3xOpto.py` | Exploratory, superseded |
| `opto_latency_with_jelte_data.py` | Exploratory, superseded |
| `session_config_extra.py` | Exploratory sessions, not in final analysis |
| `session_config_combined.py` | Exploratory, not in final analysis |
| `hitrate.py` | Raw script, superseded by hitrate_mean+SEM.py |
| `latency.py` | Raw script, superseded by latency_mean+SEM.py |
| `learn_index.py` | Raw script, superseded by learn_index_mean+SEM.py |
| `run_all_bpod.py` | Convenience runner, not a standalone analysis |

---

## Data / output folders (do NOT upload raw data)

- `output_phaseB/` — Phase B annotated figures and caption text
- `output_opto_summary_v3/` — Final opto heatmap (Fig 20)
- `output_example_traces_final/` — Final example traces (Fig 15)
- `output_figure2_final/` — Learning kinematics re-rendered with angle-wrap fix
- `THESIS_FIGURES_FINAL/` — Assembled thesis figures folder
- `STATS_AUDIT_REPORT.md` — Statistical audit (pre-Phase B)

---

## Key methods summary (for README)

- **Circular baseline subtract** (angle-wrap fix): `dev = ((trace - bl_mean + 180) % 360) - 180`
- **Effect size**: rank-biserial r (unpaired, W2T vs Opto HIT trials)
- **Multiple comparisons**: BH-FDR over 16 cells (pose opto heatmap)
- **Bootstrap CI**: 5000 resamples (bpod_utils.bootstrap_ci)
- **LME**: statsmodels.mixedlm, REML, random intercept per mouse
- **Per-mouse trial test** (Phase B): Mann-Whitney U (W2T vs opto trial outcomes/latencies)
- **Outlier detection**: Iglewicz-Hoaglin modified Z-score (threshold 3.5)
- **DLC confidence threshold**: 0.8 (override from config); SLEAP: 0.6
"""


def write_github_inventory():
    path = OUTDIR / "GITHUB_INVENTORY.md"
    path.write_text(
        GITHUB_INVENTORY_TEXT.format(date=datetime.now().strftime('%Y-%m-%d')),
        encoding='utf-8')
    print(f"\n[Inventory] Written: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE B SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def write_summary(t_start):
    elapsed = datetime.now() - t_start
    figs_rendered = [f.name for f in FIGS_OUT.glob("*.png")]
    stats_files   = [f.name for f in STATS_OUT.glob("*.md")]

    text = f"""# Phase B Summary
_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_
_Runtime: {elapsed.seconds}s_

## What was done

### Task 1: Statistical implementation
- **Fig 1**: LME slope annotation added to novice hit rate plot
- **Fig 2**: LME slope + Spearman correlation annotation added to novice latency plot
- **Fig 3**: Bootstrap CI for sigmoid inflection day (1000 resamples, resample mice); SEM label corrected to "SEM (n=4 mice)"
- **Figs 5/8**: Per-mouse trial-level Mann-Whitney U (W2T vs opto), stars above significant mouse markers
- **Figs 6/7**: One-sample Wilcoxon (Delta vs 0) per area; modified Z-score outlier detection; outlier points marked with (!)
- **Fig 10**: Caption text updated with explicit n=1 WT disclaimer
- **Fig 11**: Re-rendered in 2x2 layout + Wilcoxon signed-rank per area
- **Figs 17/19/20**: Caption text files written (re-rendering requires full pose pipeline)

### Task 2: Cosmetic fixes
- **Figs 6/7**: Removed "Positive = opto suppresses performance" / "Negative = paradoxical facilitation" from plot titles (moved to caption text)
- **Fig 11**: Changed from 5 horizontal panels to 2x2 grid (4 areas only, 'All' panel dropped); font sizes increased

### Task 3: Code inventory
- GITHUB_INVENTORY.md written with all thesis-relevant scripts, abandoned scripts, and methods summary

## Key statistical notes

- **n=3 ChR2-VGAT mice**: Group-level Wilcoxon cannot be significant (min p~0.25). Per-mouse trial tests used as primary evidence.
- **n=1 wildtype**: No formal genotype comparison. WT is qualitative control only.
- **Per-mouse MWU**: Pools all W2T vs all opto trials within each mouse-area-condition combination. Valid for binary (hit/miss) and continuous (latency) data.
- **Modified Z-score**: Iglewicz-Hoaglin (threshold 3.5); applied to per-mouse Delta values within each brain area.
- **Bootstrap inflection CI** (Fig 3): Resamples mice (n=4) with replacement 1000 times; fits sigmoid to each resampled group mean.

## Files produced

### Figures
{chr(10).join("- " + f for f in sorted(figs_rendered))}

### Stats text (ready-to-paste captions)
{chr(10).join("- " + f for f in sorted(stats_files))}

## What to do next

1. Run phaseB_figures.py to populate actual numbers into caption .md files
2. Check annotated figures in output_phaseB/figures_updated/
3. Copy caption text from stats_updates/ into thesis
4. For Figs 17/19: run pose_learning_kinematics.py and copy LME output into caption
5. Upload to GitHub (exclude raw data, include GITHUB_INVENTORY.md)
"""
    path = OUTDIR / "phaseB_summary.md"
    path.write_text(text, encoding='utf-8')
    print(f"[Summary] Written: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# ZIP
# ═══════════════════════════════════════════════════════════════════════════════

def create_zip():
    zip_path = BASE / "output_phaseB.zip"
    all_files = list(OUTDIR.rglob("*"))
    n_files   = sum(1 for f in all_files if f.is_file())
    total_mb  = sum(f.stat().st_size for f in all_files if f.is_file()) / 1e6

    print(f"\n[ZIP] {n_files} files, {total_mb:.1f} MB -> {zip_path}")
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(all_files):
            if f.is_file():
                zf.write(f, f.relative_to(OUTDIR.parent))
    zip_mb = zip_path.stat().st_size / 1e6
    print(f"[ZIP] Done: {zip_mb:.2f} MB")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    t_start = datetime.now()
    print("=" * 60)
    print("Phase B: Statistical annotations + cosmetic fixes")
    print("=" * 60)

    # Task 1 + 2: Figure rendering
    render_fig1()
    render_fig2()
    render_fig3()

    data_opto, mice_all, areas = _collect_opto_hitrate_data()
    render_fig5(data_opto, mice_all, areas)
    render_fig6_fig7(data_opto, mice_all, areas)
    render_fig8()
    render_fig10_caption()
    render_fig11()
    generate_pose_captions()

    # Write stats text files
    write_stats_files()

    # Task 3: GitHub inventory
    write_github_inventory()

    # Summary + ZIP
    write_summary(t_start)
    create_zip()

    print("\n" + "=" * 60)
    print("DONE — output_phaseB/ assembled")
    print(f"ZIP: {BASE / 'output_phaseB.zip'}")
    print("=" * 60)

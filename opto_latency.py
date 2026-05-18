# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 16:29:36 2025

@author: Larkum_Practical_01
"""

import numpy as np
import re, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
from collections import defaultdict
from scipy import stats
from scipy.stats import gaussian_kde
import statsmodels.formula.api as smf
from pathlib import Path

from bpod_utils import (load_mat, latency, extract_day_from_path, MIN_TRIALS,
                        fdr_bh, rank_biserial_r, bootstrap_ci,
                        flag_outlier_sessions, is_blocked_session)
from session_config import mouse_sessions
from plot_config import MOUSE_COLORS_SHORT as MOUSE_COLORS, MOUSE_MARKERS, COND_COLORS, style_ax

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


# ── Session loader ────────────────────────────────────────────────────────────

def load_latency_from_session(filepath, trialtype_filter, return_all=False):
    """Return median latency (or all values if return_all=True) for one session."""
    if not os.path.exists(filepath):
        return (np.nan if not return_all else [])
    data = load_mat(filepath)
    session_data = data.get('SessionData', None)
    if session_data is None:
        return (np.nan if not return_all else [])
    trialtypes = session_data.get('TrialTypes', None)
    raw_events  = session_data.get('RawEvents', None)
    trials = raw_events.get('Trial', None) if isinstance(raw_events, dict) else getattr(raw_events, 'Trial', None)
    if trials is None:
        return (np.nan if not return_all else [])
    if isinstance(trials, dict):
        trials = [trials]
    if len(trials) < MIN_TRIALS:
        return (np.nan if not return_all else [])
    lats = [latency(trials[i]) for i, tt in enumerate(trialtypes)
            if tt == trialtype_filter and not np.isnan(latency(trials[i]))]
    if return_all:
        return lats
    return np.nanmedian(lats) if lats else np.nan


def get_mouse_session_values(mouse_sessions, mouse, brain_area, trial_key, tt_num):
    paths = [p for p in mouse_sessions.get(mouse, {}).get(brain_area, {}).get(trial_key, []) if p]
    return [v for v in (load_latency_from_session(p, tt_num) for p in paths)
            if not np.isnan(v)]


def get_all_trial_latencies(mouse_sessions, mice_list, brain_area, trial_key, tt_num):
    """Pool all individual trial latencies across mice/sessions for a condition."""
    all_lats = []
    for mouse in mice_list:
        paths = [p for p in mouse_sessions.get(mouse, {}).get(brain_area, {}).get(trial_key, []) if p]
        for p in paths:
            all_lats.extend(load_latency_from_session(p, tt_num, return_all=True))
    return [v for v in all_lats if not np.isnan(v)]


def compute_group_stat(mouse_sessions, mice_list, brain_area, trial_key, trialtype_num):
    mouse_medians, mouse_ids = [], []
    for mouse in mice_list:
        vals = get_mouse_session_values(mouse_sessions, mouse, brain_area, trial_key, trialtype_num)
        if vals:
            flag_outlier_sessions(vals, mouse_id=f'{mouse}/{brain_area}/{trial_key}')
            mouse_medians.append(np.median(vals))
            mouse_ids.append(mouse)
    if not mouse_medians:
        return np.nan, np.nan, np.nan, [], []
    mean_val = np.mean(mouse_medians)
    lo, hi = bootstrap_ci(mouse_medians)
    return mean_val, lo, hi, mouse_medians, mouse_ids


# ── ex-Gaussian fit ───────────────────────────────────────────────────────────

def fit_exgaussian(data, n_boot=200):
    """
    Fit ex-Gaussian (µ, σ, τ) to response-time data via MLE.
    Returns (mu, sigma, tau, fit_ok).
    ex-Gaussian pdf: convolution of Normal(µ,σ) and Exponential(τ).
    """
    from scipy.stats import exponnorm
    data = np.asarray(data)
    data = data[~np.isnan(data)]
    if len(data) < 10:
        return np.nan, np.nan, np.nan, False
    try:
        # exponnorm: K = tau/sigma, loc = mu, scale = sigma
        K, mu, sigma = exponnorm.fit(data, floc=None)
        tau = K * sigma
        return mu, sigma, tau, True
    except Exception:
        return np.nan, np.nan, np.nan, False


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _add_individual_points(ax, xpos, mouse_medians, mouse_ids, jitter=0.04):
    rng = np.random.default_rng(0)
    for m, val in zip(mouse_ids, mouse_medians):
        jit = rng.uniform(-jitter, jitter)
        ax.plot(xpos + jit, val,
                marker=MOUSE_MARKERS.get(m, 'o'),
                color=MOUSE_COLORS.get(m, 'k'),
                markersize=6, zorder=5, linestyle='none',
                markeredgecolor='white', markeredgewidth=0.5)


def _spaghetti_legend(ax):
    handles = [mlines.Line2D([], [], color=MOUSE_COLORS[m], marker=MOUSE_MARKERS[m],
                             linestyle='-', markersize=6, label=m)
               for m in MOUSE_COLORS]
    ax.legend(handles=handles, fontsize=7, title='Mouse', title_fontsize=7,
              loc='upper right', framealpha=0.7)


# ── Plot 1: Per mouse & brain area ────────────────────────────────────────────

def plot_latencies(mouse_sessions):
    mice = list(mouse_sessions.keys())
    trial_types = [('W2T', 1), ('opto_0.5s', 2), ('opto_2s', 2)]
    colors = [COND_COLORS['W2T'], COND_COLORS['opto_0.5s'], COND_COLORS['opto_2s']]
    brain_areas = sorted({area for m in mice for area in mouse_sessions[m].keys()})

    for brain_area in brain_areas:
        fig, ax = plt.subplots(figsize=(max(8, len(mice) * 2), 5))
        style_ax(ax)
        bar_width = 0.22
        x = np.arange(len(mice))

        for i, (tt_name, tt_num) in enumerate(trial_types):
            means, lo_list, hi_list = [], [], []
            for mouse in mice:
                vals = get_mouse_session_values(mouse_sessions, mouse, brain_area, tt_name, tt_num)
                if vals:
                    lo, hi = bootstrap_ci(vals)
                    means.append(np.mean(vals)); lo_list.append(lo); hi_list.append(hi)
                else:
                    means.append(np.nan); lo_list.append(np.nan); hi_list.append(np.nan)
            xerr_lo = [m - lo if not np.isnan(m) else 0 for m, lo in zip(means, lo_list)]
            xerr_hi = [hi - m  if not np.isnan(m) else 0 for m, hi in zip(means, hi_list)]
            ax.bar(x + i * bar_width, means, width=bar_width, color=colors[i],
                   yerr=[xerr_lo, xerr_hi], capsize=4,
                   label=tt_name.replace('_', ' '), error_kw={'elinewidth': 1.2})
            for j, mouse in enumerate(mice):
                vals = get_mouse_session_values(mouse_sessions, mouse, brain_area, tt_name, tt_num)
                for v in vals:
                    ax.plot(x[j] + i * bar_width, v, 'k.', markersize=4, zorder=5, alpha=0.6)

        ax.set_xticks(x + bar_width)
        ax.set_xticklabels(mice, rotation=15, ha='right', fontsize=8)
        ax.set_ylabel('Median Latency (s)')
        ax.set_title(f'Latency per mouse — {brain_area} OptoStim\n(bars = mean of session medians, dots = sessions, error = 95% bootstrap CI)')
        ax.legend(fontsize=8)
        ax.set_ylim(0, 2.2)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        _savefig('opto_latency')


# ── Plot 2: MLA vs SNA per area — spaghetti ──────────────────────────────────

def plot_grouped_latency_MLA_vs_SNA(mouse_sessions):
    mice_all = list(mouse_sessions.keys())
    mla_mice = [m for m in mice_all if m.startswith('MLA')]
    sna_mice = [m for m in mice_all if m.startswith('SNA')]
    brain_areas = sorted({area for m in mice_all for area in mouse_sessions[m].keys()})
    trial_types = [('W2T', 1), ('opto_0.5s', 2), ('opto_2s', 2)]
    colors_mla = ['#2166ac', '#4dac26', '#d01c8b']
    colors_sna = ['#92c5de', '#b8e186', '#f1b6da']

    for area in brain_areas:
        fig, ax = plt.subplots(figsize=(9, 5))
        style_ax(ax)
        bar_width = 0.32
        indices = np.arange(len(trial_types))

        for i, (tt_name, tt_num) in enumerate(trial_types):
            for grp_mice, cols, offset, label_sfx in [
                (mla_mice, colors_mla, -bar_width/2, 'ChR2-VGAT'),
                (sna_mice, colors_sna,  bar_width/2, 'WT'),
            ]:
                mean_v, lo, hi, mmeds, mids = compute_group_stat(
                    mouse_sessions, grp_mice, area, tt_name, tt_num)
                xpos = indices[i] + offset
                if not np.isnan(mean_v):
                    ax.bar(xpos, mean_v, bar_width, color=cols[i],
                           yerr=[[mean_v - lo], [hi - mean_v]], capsize=4,
                           error_kw={'elinewidth': 1.2},
                           label=f'{label_sfx} {tt_name.replace("_"," ")}' if i == 0 else '')
                    _add_individual_points(ax, xpos, mmeds, mids, jitter=0.05)
                    for m, val in zip(mids, mmeds):
                        ax.plot(xpos, val, marker=MOUSE_MARKERS.get(m, 'o'),
                                color=MOUSE_COLORS.get(m, 'k'), markersize=7,
                                zorder=6, linestyle='none',
                                markeredgecolor='white', markeredgewidth=0.6)

        for grp_mice, offset in [(mla_mice, -bar_width/2), (sna_mice, bar_width/2)]:
            for mouse in grp_mice:
                mx, my = [], []
                for i, (tt_name, tt_num) in enumerate(trial_types):
                    vals = get_mouse_session_values(mouse_sessions, mouse, area, tt_name, tt_num)
                    if vals:
                        mx.append(indices[i] + offset)
                        my.append(np.median(vals))
                if len(mx) > 1:
                    ax.plot(mx, my, color=MOUSE_COLORS.get(mouse, 'gray'),
                            linewidth=1, alpha=0.5, zorder=4)

        ax.set_xticks(indices)
        ax.set_xticklabels([t[0].replace('_', ' ') for t in trial_types])
        ax.set_ylim(0, 2.2)
        ax.set_ylabel('Median Latency (s)')
        ax.set_title(f'Latency: ChR2-VGAT vs WT — {area}\n(bars = group mean, lines = individual mice, error = 95% bootstrap CI)')
        group_legend = ax.legend(
            handles=[
                mlines.Line2D([], [], color='#2166ac', marker='s', linestyle='none',
                              markersize=8, label='ChR2-VGAT'),
                mlines.Line2D([], [], color='#92c5de', marker='s', linestyle='none',
                              markersize=8, label='WT'),
            ],
            fontsize=7, loc='upper left', title='Group', framealpha=0.8)
        ax.add_artist(group_legend)
        _spaghetti_legend(ax)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        _savefig('opto_latency')


# ── Plot 3: All mice, all areas ───────────────────────────────────────────────

def plot_all_areas_all_mice_latency(mouse_sessions):
    mice = list(mouse_sessions.keys())
    trial_types = [('W2T', 1), ('opto_0.5s', 2), ('opto_2s', 2)]
    colors = [COND_COLORS['W2T'], COND_COLORS['opto_0.5s'], COND_COLORS['opto_2s']]
    brain_areas = sorted({area for m in mice for area in mouse_sessions[m].keys()})
    bar_width = 0.22
    x = np.arange(len(brain_areas))

    fig, ax = plt.subplots(figsize=(max(10, len(brain_areas) * 2.5), 5))
    style_ax(ax)
    for i, (tt_name, tt_num) in enumerate(trial_types):
        means, lo_list, hi_list = [], [], []
        for area in brain_areas:
            per_mouse = []
            for mouse in mice:
                vals = get_mouse_session_values(mouse_sessions, mouse, area, tt_name, tt_num)
                if vals:
                    per_mouse.append(np.median(vals))
            if per_mouse:
                means.append(np.mean(per_mouse))
                lo, hi = bootstrap_ci(per_mouse)
                lo_list.append(lo); hi_list.append(hi)
            else:
                means.append(np.nan); lo_list.append(np.nan); hi_list.append(np.nan)

        xerr_lo = [m - lo if not np.isnan(m) else 0 for m, lo in zip(means, lo_list)]
        xerr_hi = [hi - m  if not np.isnan(m) else 0 for m, hi in zip(means, hi_list)]
        ax.bar(x + i * bar_width, means, width=bar_width, color=colors[i],
               yerr=[xerr_lo, xerr_hi], capsize=4,
               label=tt_name.replace('_', ' '), error_kw={'elinewidth': 1.2})
        for j, area in enumerate(brain_areas):
            for mouse in mice:
                vals = get_mouse_session_values(mouse_sessions, mouse, area, tt_name, tt_num)
                if vals:
                    ax.plot(x[j] + i * bar_width, np.median(vals),
                            marker=MOUSE_MARKERS.get(mouse, 'o'),
                            color=MOUSE_COLORS.get(mouse, 'k'),
                            markersize=6, zorder=5, linestyle='none',
                            markeredgecolor='white', markeredgewidth=0.5)

    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(brain_areas)
    ax.set_ylabel('Median Latency (s)')
    ax.set_title('Latency across brain areas — all mice\n(markers = per-mouse median, error = 95% bootstrap CI)')
    ax.set_ylim(0, 2.2)
    cond_legend = ax.legend(fontsize=8, loc='upper left', title='Condition', framealpha=0.8)
    ax.add_artist(cond_legend)
    ax.grid(axis='y', alpha=0.3)
    _spaghetti_legend(ax)
    plt.tight_layout()
    _savefig('opto_latency')


# ── Plot 4: ChR2 vs WT, all areas ────────────────────────────────────────────

def plot_latency_all_areas_MLA_vs_SNA(mouse_sessions):
    mice_all = list(mouse_sessions.keys())
    mla_mice = [m for m in mice_all if m.startswith('MLA')]
    sna_mice = [m for m in mice_all if m.startswith('SNA')]
    trial_types = [('W2T', 1), ('opto_0.5s', 2), ('opto_2s', 2)]
    colors_mla = ['#2166ac', '#4dac26', '#d01c8b']
    colors_sna = ['#92c5de', '#b8e186', '#f1b6da']
    brain_areas = sorted({area for m in mice_all for area in mouse_sessions[m].keys()})
    bar_width = 0.12
    x = np.arange(len(brain_areas))

    fig, ax = plt.subplots(figsize=(max(12, len(brain_areas) * 3), 5))
    style_ax(ax)
    for i, (tt_name, tt_num) in enumerate(trial_types):
        for grp_mice, cols, grp_offset, grp_label in [
            (mla_mice, colors_mla, 0,         'ChR2-VGAT'),
            (sna_mice, colors_sna, bar_width,  'WT'),
        ]:
            means, lo_list, hi_list, all_mmeds = [], [], [], []
            for area in brain_areas:
                mean_v, lo, hi, mmeds, _ = compute_group_stat(
                    mouse_sessions, grp_mice, area, tt_name, tt_num)
                means.append(mean_v); lo_list.append(lo); hi_list.append(hi)
                all_mmeds.append(mmeds)
            xpos = x + (i * 2 * bar_width) + grp_offset
            xerr_lo = [m - lo if not np.isnan(m) else 0 for m, lo in zip(means, lo_list)]
            xerr_hi = [hi - m  if not np.isnan(m) else 0 for m, hi in zip(means, hi_list)]
            ax.bar(xpos, means, bar_width, color=cols[i],
                   yerr=[xerr_lo, xerr_hi], capsize=3,
                   label=f'{grp_label} {tt_name.replace("_"," ")}',
                   error_kw={'elinewidth': 1.0})
            for j, mmeds_j in enumerate(all_mmeds):
                for v in mmeds_j:
                    ax.plot(xpos[j], v, 'k.', markersize=5, zorder=5, alpha=0.7)

    ax.set_xticks(x + bar_width * 3)
    ax.set_xticklabels(brain_areas)
    ax.set_ylabel('Median Latency (s)')
    ax.set_title('Latency: ChR2-VGAT vs WT across all brain areas\n(error = 95% bootstrap CI, dots = per-mouse medians)')
    ax.set_ylim(0, 2.2)
    ax.legend(fontsize=7, ncol=3)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    _savefig('opto_latency')


# ── Plot 5: Δ Latency (Opto − W2T) per mouse & area ─────────────────────────

def plot_delta_latency(mouse_sessions):
    """
    For each opto condition: plot Opto_median − W2T_median per mouse per area.
    Positive = opto slows responses. Reference line at 0.
    """
    mice_all    = list(mouse_sessions.keys())
    brain_areas = sorted({area for m in mice_all for area in mouse_sessions[m].keys()})
    opto_conds  = [('opto_0.5s', 2, '0.5 s opto'), ('opto_2s', 2, '2 s opto')]

    for opto_key, tt_num, opto_label in opto_conds:
        fig, ax = plt.subplots(figsize=(max(6, len(brain_areas) * 1.8), 5))
        style_ax(ax)
        x = np.arange(len(brain_areas))
        jitter_rng = np.random.default_rng(1)

        delta_per_area = {area: {'vals': [], 'mice': []} for area in brain_areas}

        for mouse in mice_all:
            for area in brain_areas:
                w2t_vals  = get_mouse_session_values(mouse_sessions, mouse, area, 'W2T',    1)
                opto_vals = get_mouse_session_values(mouse_sessions, mouse, area, opto_key, tt_num)
                if w2t_vals and opto_vals:
                    delta = np.median(opto_vals) - np.median(w2t_vals)
                    delta_per_area[area]['vals'].append(delta)
                    delta_per_area[area]['mice'].append(mouse)

        means, lo_list, hi_list = [], [], []
        for area in brain_areas:
            vals = delta_per_area[area]['vals']
            if vals:
                means.append(np.mean(vals))
                lo, hi = bootstrap_ci(vals)
                lo_list.append(lo); hi_list.append(hi)
            else:
                means.append(np.nan); lo_list.append(np.nan); hi_list.append(np.nan)

        xerr_lo = [m - lo if not np.isnan(m) else 0 for m, lo in zip(means, lo_list)]
        xerr_hi = [hi - m  if not np.isnan(m) else 0 for m, hi in zip(means, hi_list)]
        ax.bar(x, means, 0.5, color='#aaaaaa', alpha=0.5,
               yerr=[xerr_lo, xerr_hi], capsize=5,
               error_kw={'elinewidth': 1.5}, zorder=2, label='Group mean')

        for j, area in enumerate(brain_areas):
            for val, mouse in zip(delta_per_area[area]['vals'],
                                  delta_per_area[area]['mice']):
                jit = jitter_rng.uniform(-0.12, 0.12)
                ax.plot(j + jit, val,
                        marker=MOUSE_MARKERS.get(mouse, 'o'),
                        color=MOUSE_COLORS.get(mouse, 'k'),
                        markersize=9, zorder=5, linestyle='none',
                        markeredgecolor='white', markeredgewidth=0.7)

        ax.axhline(0, color='black', linewidth=1.2, linestyle='--', alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(brain_areas)
        ax.set_ylabel('Δ Latency: Opto − W2T (s)')
        ax.set_title(f'Latency Shift ({opto_label})\n'
                     f'Positive = opto slows responses\n'
                     f'(error = 95% bootstrap CI, markers = individual mice)')
        _spaghetti_legend(ax)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        _savefig('opto_latency')


# ── Plot 6: Violin plots latency distribution per area ───────────────────────

def plot_latency_violins(mouse_sessions):
    """
    Violin + box + individual points for all trial latencies per condition per area.
    W2T (dark) vs opto_0.5s vs opto_2s — shows full distribution for reviewers.
    """
    mice_all    = list(mouse_sessions.keys())
    brain_areas = sorted({area for m in mice_all for area in mouse_sessions[m].keys()})
    trial_types = [('W2T', 1), ('opto_0.5s', 2), ('opto_2s', 2)]
    colors = COND_COLORS
    positions_offsets = {'W2T': -0.28, 'opto_0.5s': 0.0, 'opto_2s': 0.28}

    for area in brain_areas:
        fig, ax = plt.subplots(figsize=(max(7, len(brain_areas) * 2), 5))
        style_ax(ax)
        fig.suptitle(f'Latency Distribution — {area} (all trials)', fontsize=11)

        x_base = np.arange(1, 2)   # single area → position 1

        all_data, all_pos, all_cols = [], [], []
        for tt_name, tt_num in trial_types:
            lats = np.array(get_all_trial_latencies(mouse_sessions, mice_all,
                                                    area, tt_name, tt_num))
            lats = lats[(lats > 0) & (lats < 2.0)]
            if len(lats) < 5:
                continue
            pos = 1 + positions_offsets[tt_name]
            all_data.append(lats)
            all_pos.append(pos)
            all_cols.append(colors[tt_name])

        if not all_data:
            plt.close()
            continue

        parts = ax.violinplot(all_data, positions=all_pos,
                              widths=0.22, showmedians=False, showextrema=False)
        for pc, col in zip(parts['bodies'], all_cols):
            pc.set_facecolor(col)
            pc.set_alpha(0.45)
            pc.set_edgecolor(col)

        # box plot overlay
        bp = ax.boxplot(all_data, positions=all_pos, widths=0.08,
                        patch_artist=True, showfliers=False,
                        medianprops={'color': 'white', 'linewidth': 2},
                        whiskerprops={'linewidth': 1.2},
                        capprops={'linewidth': 1.2},
                        boxprops={'linewidth': 0})
        for patch, col in zip(bp['boxes'], all_cols):
            patch.set_facecolor(col)
            patch.set_alpha(0.85)

        # jittered per-mouse median points on top
        rng = np.random.default_rng(7)
        for (tt_name, tt_num), pos, col in zip(trial_types, all_pos, all_cols):
            for mouse in mice_all:
                vals = get_mouse_session_values(mouse_sessions, mouse, area, tt_name, tt_num)
                if vals:
                    jit = rng.uniform(-0.04, 0.04)
                    ax.plot(pos + jit, np.median(vals),
                            marker=MOUSE_MARKERS.get(mouse, 'o'),
                            color=MOUSE_COLORS.get(mouse, 'k'),
                            markersize=7, zorder=6, linestyle='none',
                            markeredgecolor='white', markeredgewidth=0.6)

        present_names = [tt_name for (tt_name, tt_num), pos in zip(trial_types, all_pos)]
        ax.set_xticks(all_pos)
        ax.set_xticklabels([n.replace('_', ' ') for n in present_names], fontsize=9)
        ax.set_ylabel('Latency (s)')
        ax.set_ylim(0, 2.1)
        ax.set_title(f'{area} — Latency Distributions\n'
                     f'(violin = all trials, box = IQR+median, markers = per-mouse median)')
        ax.grid(axis='y', alpha=0.3)
        _spaghetti_legend(ax)
        plt.tight_layout()
        _savefig('opto_latency')


# ── Plot 7: Quantile plots per area (W2T vs opto) ────────────────────────────

def plot_latency_quantiles(mouse_sessions):
    """
    For each brain area: CDF + quantile-quantile comparison W2T vs opto conditions.
    Panel A: overlaid CDFs (all trial latencies pooled across mice).
    Panel B: Q-Q plot W2T vs opto_0.5s and opto_2s.
    """
    mice_all = list(mouse_sessions.keys())
    brain_areas = sorted({area for m in mice_all for area in mouse_sessions[m].keys()})
    trial_types = [('W2T', 1), ('opto_0.5s', 2), ('opto_2s', 2)]
    colors = COND_COLORS
    quantiles = np.linspace(0.05, 0.95, 19)

    for area in brain_areas:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for _ax in axes: style_ax(_ax)
        fig.suptitle(f'Latency Distribution — {area}', fontsize=11)

        # Collect all trial latencies per condition
        cond_lats = {}
        for tt_name, tt_num in trial_types:
            lats = get_all_trial_latencies(mouse_sessions, mice_all, area, tt_name, tt_num)
            cond_lats[tt_name] = np.array(lats)

        # Panel 1: overlaid CDFs
        ax = axes[0]
        for tt_name, _ in trial_types:
            lats = np.sort(cond_lats[tt_name])
            if len(lats) == 0:
                continue
            ecdf = np.arange(1, len(lats) + 1) / len(lats)
            ax.plot(lats, ecdf, color=colors[tt_name],
                    linewidth=2, label=f'{tt_name.replace("_"," ")} (n={len(lats)})')
        ax.set_xlabel('Latency (s)')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title('Empirical CDF')
        ax.legend(fontsize=8)
        ax.set_xlim(0, 2.05)
        ax.grid(alpha=0.3)

        # Panel 2 & 3: Q-Q plots W2T vs opto
        for ax_idx, (opto_key, ax) in enumerate(
                zip(['opto_0.5s', 'opto_2s'], axes[1:])):
            w2t_lats  = cond_lats['W2T']
            opto_lats = cond_lats[opto_key]
            if len(w2t_lats) < 5 or len(opto_lats) < 5:
                ax.set_title(f'Q-Q: W2T vs {opto_key}\n(insufficient data)')
                continue
            q_w2t  = np.quantile(w2t_lats, quantiles)
            q_opto = np.quantile(opto_lats, quantiles)
            ax.scatter(q_w2t, q_opto, color=colors[opto_key], s=40, zorder=3)
            lim = max(q_w2t.max(), q_opto.max()) * 1.05
            ax.plot([0, lim], [0, lim], 'k--', linewidth=1, alpha=0.5, label='identity')
            ax.set_xlabel('W2T quantile (s)')
            ax.set_ylabel(f'{opto_key.replace("_"," ")} quantile (s)')
            ax.set_title(f'Q-Q: W2T vs {opto_key.replace("_"," ")}')
            ax.set_xlim(0, lim); ax.set_ylim(0, lim)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
            # KS test annotation
            if len(w2t_lats) > 1 and len(opto_lats) > 1:
                ks_stat, ks_p = stats.ks_2samp(w2t_lats, opto_lats)
                sig = '***' if ks_p < 0.001 else '**' if ks_p < 0.01 else '*' if ks_p < 0.05 else 'ns'
                ax.text(0.05, 0.92, f'KS={ks_stat:.2f}, p={ks_p:.3f} {sig}',
                        transform=ax.transAxes, fontsize=8)

        plt.tight_layout()
        _savefig('opto_latency')


# ── Plot 6: ex-Gaussian fit per area ─────────────────────────────────────────

def plot_exgaussian_fits(mouse_sessions):
    """
    Fit ex-Gaussian to trial latency distributions per condition and area.
    Shows KDE + fitted ex-Gaussian curve. Prints fit parameters.
    Only shows plot if fit succeeded for at least one condition.
    """
    from scipy.stats import exponnorm
    mice_all = list(mouse_sessions.keys())
    brain_areas = sorted({area for m in mice_all for area in mouse_sessions[m].keys()})
    trial_types = [('W2T', 1), ('opto_0.5s', 2), ('opto_2s', 2)]
    colors = COND_COLORS
    x_grid = np.linspace(0.01, 2.0, 200)

    print("\n--- ex-Gaussian fit parameters (mu=mean Gaussian component, sigma=SD, tau=exponential tail) ---")
    print(f"  {'Area':<6} {'Condition':<12} {'n':>5}  {'mu':>6}  {'sig':>6}  {'tau':>6}  {'mu+t':>6}  fit")
    print("  " + "-" * 60)

    for area in brain_areas:
        fig, axes = plt.subplots(1, len(trial_types), figsize=(13, 4), sharey=True)
        for _ax in axes: style_ax(_ax)
        fig.suptitle(f'ex-Gaussian fit — {area}', fontsize=11)
        any_fit = False

        for ax, (tt_name, tt_num) in zip(axes, trial_types):
            lats = np.array(get_all_trial_latencies(mouse_sessions, mice_all, area, tt_name, tt_num))
            lats = lats[(lats > 0) & (lats < 2.0)]

            mu, sigma, tau, fit_ok = fit_exgaussian(lats)
            n = len(lats)
            fit_str = 'ok' if fit_ok else 'fail'
            print(f"  {area:<6} {tt_name:<12} {n:>5}  "
                  + (f"{mu:>6.3f}  {sigma:>6.3f}  {tau:>6.3f}  {mu+tau:>6.3f}  {fit_str}"
                     if fit_ok else f"{'—':>6}  {'—':>6}  {'—':>6}  {'—':>6}  {fit_str}"))

            # KDE of data
            if n > 1:
                kde = gaussian_kde(lats, bw_method='scott')
                ax.fill_between(x_grid, kde(x_grid), alpha=0.25,
                                color=colors[tt_name])
                ax.plot(x_grid, kde(x_grid), color=colors[tt_name],
                        linewidth=2, label='KDE')

            # ex-Gaussian curve
            if fit_ok:
                from scipy.stats import exponnorm
                K = tau / sigma
                pdf = exponnorm.pdf(x_grid, K, loc=mu, scale=sigma)
                ax.plot(x_grid, pdf, color=colors[tt_name], linewidth=2,
                        linestyle='--', label=f'ex-G (µ={mu:.2f}, τ={tau:.2f})')
                # mark median and mean+tau
                ax.axvline(np.median(lats), color=colors[tt_name],
                           linestyle=':', linewidth=1.5, alpha=0.7)
                any_fit = True

            ax.set_xlim(0, 2.05)
            ax.set_xlabel('Latency (s)')
            ax.set_title(f'{tt_name.replace("_"," ")} (n={n})')
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)

        axes[0].set_ylabel('Density')
        plt.tight_layout()
        if any_fit:
            _savefig('opto_latency_exgaussian')
        else:
            plt.close()
            print(f"  [{area}] no successful fits — plot skipped")


# ── Statistics ────────────────────────────────────────────────────────────────

def build_latency_dataframe(mouse_sessions):
    """One row per session: mouse, genotype, area, condition, latency, is_blocked."""
    trial_type_map = {'W2T': 1, 'opto_0.5s': 2, 'opto_2s': 2}
    rows = []
    for mouse, areas in mouse_sessions.items():
        geno = 'WT' if mouse.startswith('SNA') else 'ChR2'
        for area, conds in areas.items():
            for cond_key, tt_num in trial_type_map.items():
                for path in conds.get(cond_key, []):
                    val = load_latency_from_session(path, tt_num)
                    if not np.isnan(val):
                        rows.append({'mouse': mouse, 'genotype': geno,
                                     'area': area, 'condition': cond_key,
                                     'latency': val,
                                     'blocked': is_blocked_session(path)})
    return pd.DataFrame(rows)


def run_statistics(mouse_sessions):
    df = build_latency_dataframe(mouse_sessions)
    if df.empty:
        print("No data for statistics.")
        return

    print("\n" + "=" * 70)
    print("STATISTICS — Response Latency")
    print("=" * 70)

    # ── Descriptive ──────────────────────────────────────────────────
    print("\n--- Descriptive (per-mouse session medians) ---")
    mouse_medians = (df.groupby(['mouse', 'area', 'condition'])['latency']
                     .median().reset_index())
    desc = (mouse_medians.groupby(['area', 'condition'])['latency']
            .agg(n='count', median='median', std='std').round(3))
    print(desc.to_string())

    # ── KS tests: W2T vs opto (all trials pooled) ────────────────────
    print("\n--- KS test: latency distribution W2T vs opto (all trials pooled per area) ---")
    mice_all = list(mouse_sessions.keys())
    for area in sorted(df['area'].unique()):
        w2t_lats = np.array(get_all_trial_latencies(mouse_sessions, mice_all, area, 'W2T', 1))
        for opto_cond, tt_num in [('opto_0.5s', 2), ('opto_2s', 2)]:
            opto_lats = np.array(get_all_trial_latencies(mouse_sessions, mice_all, area, opto_cond, tt_num))
            if len(w2t_lats) < 5 or len(opto_lats) < 5:
                print(f"  {area:<6} W2T vs {opto_cond:<11} — insufficient trials")
                continue
            ks, p = stats.ks_2samp(w2t_lats, opto_lats)
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            print(f"  {area:<6} W2T (n={len(w2t_lats)}) vs {opto_cond:<11} (n={len(opto_lats)}): "
                  f"KS={ks:.3f}  p={p:.4f}  {sig}")

    # ── Wilcoxon + FDR + effect size (per-mouse medians) ─────────────
    print("\n--- Wilcoxon signed-rank: W2T vs opto (per-mouse medians, FDR-corrected) ---")
    rows_stat = []
    for area in sorted(df['area'].unique()):
        area_df = mouse_medians[mouse_medians['area'] == area]
        for opto_cond in ['opto_0.5s', 'opto_2s']:
            w2t  = area_df[area_df['condition'] == 'W2T'].set_index('mouse')
            opto = area_df[area_df['condition'] == opto_cond].set_index('mouse')
            common = sorted(set(w2t.index) & set(opto.index))
            n = len(common)
            if n < 2:
                rows_stat.append({'area': area, 'comparison': f'W2T vs {opto_cond}',
                                  'n': n, 'stat': np.nan, 'p': np.nan, 'r': np.nan,
                                  'w2t_med': np.nan, 'opto_med': np.nan})
                continue
            a = w2t.loc[common, 'latency'].values
            b = opto.loc[common, 'latency'].values
            stat_w, p = stats.wilcoxon(a, b)
            r = rank_biserial_r(a, b)
            rows_stat.append({'area': area, 'comparison': f'W2T vs {opto_cond}',
                               'n': n, 'stat': stat_w, 'p': p, 'r': r,
                               'w2t_med': np.median(a), 'opto_med': np.median(b)})

    stat_df = pd.DataFrame(rows_stat)
    valid_p = stat_df['p'].dropna()
    if len(valid_p) > 0:
        stat_df.loc[valid_p.index, 'p_fdr'] = fdr_bh(valid_p.values)
    else:
        stat_df['p_fdr'] = np.nan

    print(f"  {'Area':<6} {'Comparison':<18} {'n':>3}  {'W2T':>7}  {'Opto':>7}  "
          f"{'W':>6}  {'p':>7}  {'p_FDR':>7}  {'r':>5}  sig")
    print("  " + "-" * 75)
    for _, row in stat_df.iterrows():
        if np.isnan(row['p']):
            print(f"  {row['area']:<6} {row['comparison']:<18} {int(row['n']):>3}  — insufficient n —")
            continue
        p_fdr = row.get('p_fdr', np.nan)
        sig = '***' if p_fdr < 0.001 else '**' if p_fdr < 0.01 else '*' if p_fdr < 0.05 else 'ns'
        print(f"  {row['area']:<6} {row['comparison']:<18} {int(row['n']):>3}  "
              f"{row['w2t_med']:>7.3f}  {row['opto_med']:>7.3f}  "
              f"{row['stat']:>6.1f}  {row['p']:>7.4f}  {p_fdr:>7.4f}  "
              f"{row['r']:>5.2f}  {sig}")

    # ── LME: condition * area ─────────────────────────────────────────
    print("\n--- LME: latency ~ condition * area  |  random intercept: mouse ---")
    print("  (reference: condition=W2T)\n")
    try:
        model = smf.mixedlm(
            "latency ~ C(condition, Treatment('W2T')) * C(area)",
            df, groups=df['mouse']
        )
        result = model.fit(reml=True, method='lbfgs')
        fe, ci_lme, pvals = result.fe_params, result.conf_int(), result.pvalues
        print(f"  {'Parameter':<50} {'Coef':>7}  {'[0.025':>7}  {'0.975]':>7}  {'p':>7}")
        print("  " + "-" * 82)
        for param in fe.index:
            coef = fe[param]; lo, hi = ci_lme.loc[param]; p = pvals[param]
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            print(f"  {param[:49]:<50} {coef:>7.3f}  {lo:>7.3f}  {hi:>7.3f}  {p:>7.4f}  {sig}")
        print(f"\n  Random effect (mouse) variance: {result.cov_re.values[0][0]:.4f}")
        print(f"  Log-likelihood: {result.llf:.2f}")
    except Exception as e:
        print(f"  LME failed: {e}")

    # ── LME: genotype × condition interaction ─────────────────────────
    print("\n--- LME: Genotype × Condition interaction ---")
    print("  latency ~ condition * genotype  |  random intercept: mouse\n")
    opto_df = df[df['condition'] != 'W2T'].copy()
    if opto_df['mouse'].nunique() >= 2:
        try:
            model2 = smf.mixedlm(
                "latency ~ C(condition, Treatment('opto_0.5s')) * C(genotype, Treatment('ChR2'))",
                opto_df, groups=opto_df['mouse']
            )
            result2 = model2.fit(reml=True, method='lbfgs')
            fe2, ci2, pv2 = result2.fe_params, result2.conf_int(), result2.pvalues
            print(f"  {'Parameter':<55} {'Coef':>7}  {'[0.025':>7}  {'0.975]':>7}  {'p':>7}")
            print("  " + "-" * 85)
            for param in fe2.index:
                coef = fe2[param]; lo, hi = ci2.loc[param]; p = pv2[param]
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                print(f"  {param[:54]:<55} {coef:>7.3f}  {lo:>7.3f}  {hi:>7.3f}  {p:>7.4f}  {sig}")
            print(f"\n  Random effect (mouse) variance: {result2.cov_re.values[0][0]:.4f}")
        except Exception as e:
            print(f"  LME failed: {e}")
    else:
        print("  — insufficient mice —")

    print("=" * 70)


# ── Run all plots ─────────────────────────────────────────────────────────────
plot_latencies(mouse_sessions)
plot_grouped_latency_MLA_vs_SNA(mouse_sessions)
plot_all_areas_all_mice_latency(mouse_sessions)
plot_latency_all_areas_MLA_vs_SNA(mouse_sessions)
plot_delta_latency(mouse_sessions)
plot_latency_violins(mouse_sessions)
plot_latency_quantiles(mouse_sessions)
plot_exgaussian_fits(mouse_sessions)
run_statistics(mouse_sessions)

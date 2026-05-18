# -*- coding: utf-8 -*-
"""
Created on Wed Dec 3 16:04:44 2025

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
import statsmodels.formula.api as smf
from pathlib import Path

from bpod_utils import (load_mat, is_hit, extract_day_from_path, MIN_TRIALS,
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


# ── Session loaders ───────────────────────────────────────────────────────────

def load_hit_rate_from_session(filepath, trialtype_filter):
    if not os.path.exists(filepath):
        return np.nan
    data = load_mat(filepath)
    session_data = data.get('SessionData', None)
    if session_data is None:
        return np.nan
    trialtypes = session_data.get('TrialTypes', None)
    if trialtypes is None:
        return np.nan
    raw_events = session_data.get('RawEvents', None)
    if raw_events is None:
        return np.nan
    trials = raw_events.get('Trial', None) if isinstance(raw_events, dict) else getattr(raw_events, 'Trial', None)
    if trials is None:
        return np.nan
    if isinstance(trials, dict):
        trials = [trials]
        ntrials = 1
    else:
        ntrials = len(trials)
    if ntrials < MIN_TRIALS:
        print(f"Session {os.path.basename(filepath)} übersprungen – zu wenig Trials ({ntrials})")
        return np.nan
    hits = [1 if is_hit(trials[i]) else 0
            for i, tt in enumerate(trialtypes) if tt == trialtype_filter]
    return np.mean(hits) * 100 if hits else np.nan


# ── Per-mouse mean ± session values ──────────────────────────────────────────

def get_mouse_session_values(mouse_sessions, mouse, brain_area, trial_key, tt_num):
    """Return list of per-session hit-rates for one mouse/area/condition."""
    paths = [p for p in mouse_sessions.get(mouse, {}).get(brain_area, {}).get(trial_key, []) if p]
    return [v for v in (load_hit_rate_from_session(p, tt_num) for p in paths)
            if not np.isnan(v)]


def compute_group_stat(mouse_sessions, mice_list, brain_area, trial_key, trialtype_num):
    """
    Per-mouse mean, then group mean ± bootstrap CI across mouse means.
    Returns (mean, ci_lo, ci_hi, list_of_mouse_means, mouse_ids).
    """
    mouse_means, mouse_ids = [], []
    for mouse in mice_list:
        vals = get_mouse_session_values(mouse_sessions, mouse, brain_area, trial_key, trialtype_num)
        if vals:
            flag_outlier_sessions(vals, mouse_id=f'{mouse}/{brain_area}/{trial_key}')
            mouse_means.append(np.mean(vals))
            mouse_ids.append(mouse)
    if not mouse_means:
        return np.nan, np.nan, np.nan, [], []
    mean_val = np.mean(mouse_means)
    lo, hi = bootstrap_ci(mouse_means)
    return mean_val, lo, hi, mouse_means, mouse_ids


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _add_individual_points(ax, x_center, mouse_means, mouse_ids, jitter=0.04):
    """Overlay per-mouse points with connecting lines on a bar position."""
    rng = np.random.default_rng(0)
    for m, val in zip(mouse_ids, mouse_means):
        jit = rng.uniform(-jitter, jitter)
        ax.plot(x_center + jit, val,
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

def plot_hit_rates(mouse_sessions):
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
                    m = np.mean(vals)
                    lo, hi = bootstrap_ci(vals)
                else:
                    m, lo, hi = np.nan, np.nan, np.nan
                means.append(m)
                lo_list.append(lo)
                hi_list.append(hi)

            xerr_lo = [m - lo if not np.isnan(m) else 0 for m, lo in zip(means, lo_list)]
            xerr_hi = [hi - m if not np.isnan(m) else 0 for m, hi in zip(means, hi_list)]
            ax.bar(x + i * bar_width, means, width=bar_width, color=colors[i],
                   yerr=[xerr_lo, xerr_hi], capsize=4, label=tt_name.replace('_', ' '),
                   error_kw={'elinewidth': 1.2})

            # individual session dots
            for j, mouse in enumerate(mice):
                vals = get_mouse_session_values(mouse_sessions, mouse, brain_area, tt_name, tt_num)
                for v in vals:
                    ax.plot(x[j] + i * bar_width, v, 'k.', markersize=4,
                            zorder=5, alpha=0.6)

        ax.set_xticks(x + bar_width)
        ax.set_xticklabels(mice, rotation=15, ha='right', fontsize=8)
        ax.set_ylabel('Success Rate (%)')
        ax.set_title(f'Success Rate per mouse — {brain_area} OptoStim\n(bars = mean, dots = sessions, error = 95% bootstrap CI)')
        ax.legend(fontsize=8)
        ax.set_ylim(0, 110)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        _savefig('opto_hitrate')


# ── Plot 2: MLA vs SNA per area — spaghetti + bar ────────────────────────────

def plot_grouped_MLA_vs_SNA(mouse_sessions):
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
        tt_labels = [t[0].replace('_', ' ') for t in trial_types]

        for i, (tt_name, tt_num) in enumerate(trial_types):
            for grp_mice, cols, offset, label_sfx in [
                (mla_mice, colors_mla, -bar_width/2, 'ChR2-VGAT'),
                (sna_mice, colors_sna,  bar_width/2, 'WT'),
            ]:
                mean_v, lo, hi, mmeans, mids = compute_group_stat(
                    mouse_sessions, grp_mice, area, tt_name, tt_num)
                xpos = indices[i] + offset
                if not np.isnan(mean_v):
                    ax.bar(xpos, mean_v, bar_width, color=cols[i],
                           yerr=[[mean_v - lo], [hi - mean_v]],
                           capsize=4, error_kw={'elinewidth': 1.2},
                           label=f'{label_sfx} {tt_name.replace("_"," ")}' if i == 0 else '')
                    _add_individual_points(ax, xpos, mmeans, mids, jitter=0.05)
                    # spaghetti: connect same mouse across conditions
                    for m, val in zip(mids, mmeans):
                        ax.plot(xpos, val,
                                marker=MOUSE_MARKERS.get(m, 'o'),
                                color=MOUSE_COLORS.get(m, 'k'),
                                markersize=7, zorder=6, linestyle='none',
                                markeredgecolor='white', markeredgewidth=0.6)

        # draw within-group spaghetti lines across conditions
        for grp_mice, offset in [(mla_mice, -bar_width/2), (sna_mice, bar_width/2)]:
            for mouse in grp_mice:
                mouse_x, mouse_y = [], []
                for i, (tt_name, tt_num) in enumerate(trial_types):
                    vals = get_mouse_session_values(mouse_sessions, mouse, area, tt_name, tt_num)
                    if vals:
                        mouse_x.append(indices[i] + offset)
                        mouse_y.append(np.mean(vals))
                if len(mouse_x) > 1:
                    ax.plot(mouse_x, mouse_y,
                            color=MOUSE_COLORS.get(mouse, 'gray'),
                            linewidth=1, alpha=0.5, zorder=4)

        ax.set_xticks(indices)
        ax.set_xticklabels(tt_labels)
        ax.set_ylim(0, 110)
        ax.set_ylabel('Success Rate (%)')
        ax.set_title(f'Success Rate: ChR2-VGAT vs WT — {area}\n(bars = group mean, lines = individual mice, error = 95% bootstrap CI)')
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
        _savefig('opto_hitrate')


# ── Plot 3: All mice, all areas ───────────────────────────────────────────────

def plot_all_areas_all_mice_hitrate(mouse_sessions):
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
            # pool per-mouse means (not raw sessions) for group bar
            per_mouse = []
            for mouse in mice:
                vals = get_mouse_session_values(mouse_sessions, mouse, area, tt_name, tt_num)
                if vals:
                    per_mouse.append(np.mean(vals))
            if per_mouse:
                means.append(np.mean(per_mouse))
                lo, hi = bootstrap_ci(per_mouse)
                lo_list.append(lo); hi_list.append(hi)
            else:
                means.append(np.nan); lo_list.append(np.nan); hi_list.append(np.nan)

        xerr_lo = [m - lo if not np.isnan(m) else 0 for m, lo in zip(means, lo_list)]
        xerr_hi = [hi - m if not np.isnan(m) else 0 for m, hi in zip(means, hi_list)]
        ax.bar(x + i * bar_width, means, width=bar_width, color=colors[i],
               yerr=[xerr_lo, xerr_hi], capsize=4,
               label=tt_name.replace('_', ' '), error_kw={'elinewidth': 1.2})

        # individual per-mouse points
        for j, area in enumerate(brain_areas):
            for mouse in mice:
                vals = get_mouse_session_values(mouse_sessions, mouse, area, tt_name, tt_num)
                if vals:
                    ax.plot(x[j] + i * bar_width, np.mean(vals),
                            marker=MOUSE_MARKERS.get(mouse, 'o'),
                            color=MOUSE_COLORS.get(mouse, 'k'),
                            markersize=6, zorder=5, linestyle='none',
                            markeredgecolor='white', markeredgewidth=0.5)

    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(brain_areas)
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Success Rate across brain areas — all mice\n(bars = group mean of mouse means, markers = per-mouse mean, error = 95% bootstrap CI)')
    ax.set_ylim(0, 110)
    cond_legend = ax.legend(fontsize=8, loc='upper left', title='Condition', framealpha=0.8)
    ax.add_artist(cond_legend)
    ax.grid(axis='y', alpha=0.3)
    _spaghetti_legend(ax)
    plt.tight_layout()
    _savefig('opto_hitrate')


# ── Plot 4: ChR2 vs WT, all areas ─────────────────────────────────────────────

def plot_hitrate_all_areas_MLA_vs_SNA(mouse_sessions):
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
            means, lo_list, hi_list = [], [], []
            all_mmeans = []
            for area in brain_areas:
                mean_v, lo, hi, mmeans, _ = compute_group_stat(
                    mouse_sessions, grp_mice, area, tt_name, tt_num)
                means.append(mean_v); lo_list.append(lo); hi_list.append(hi)
                all_mmeans.append(mmeans)

            xpos = x + (i * 2 * bar_width) + grp_offset
            xerr_lo = [m - lo if not np.isnan(m) else 0 for m, lo in zip(means, lo_list)]
            xerr_hi = [hi - m if not np.isnan(m) else 0 for m, hi in zip(means, hi_list)]
            ax.bar(xpos, means, bar_width, color=cols[i],
                   yerr=[xerr_lo, xerr_hi], capsize=3,
                   label=f'{grp_label} {tt_name.replace("_"," ")}',
                   error_kw={'elinewidth': 1.0})

            for j, (area, mmeans_j) in enumerate(zip(brain_areas, all_mmeans)):
                for m_val in mmeans_j:
                    ax.plot(xpos[j], m_val, 'k.', markersize=5, zorder=5, alpha=0.7)

    ax.set_xticks(x + bar_width * 3)
    ax.set_xticklabels(brain_areas)
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Success Rate: ChR2-VGAT vs WT across all brain areas\n(error = 95% bootstrap CI, dots = per-mouse means)')
    ax.set_ylim(0, 115)
    ax.legend(fontsize=7, ncol=3)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    _savefig('opto_hitrate')


# ── Plot 5: Δ Hit-Rate (W2T − Opto) per mouse & area ─────────────────────────

def plot_delta_hitrate(mouse_sessions):
    """
    For each opto condition (0.5s, 2s): plot W2T_mean − Opto_mean per mouse per area.
    Positive = inhibition (hit rate drops with opto).
    Reference line at 0. ChR2 and WT coloured separately.
    """
    mice_all   = list(mouse_sessions.keys())
    brain_areas = sorted({area for m in mice_all for area in mouse_sessions[m].keys()})
    opto_conds  = [('opto_0.5s', 2, '0.5 s opto'), ('opto_2s', 2, '2 s opto')]

    for opto_key, tt_num, opto_label in opto_conds:
        fig, ax = plt.subplots(figsize=(max(6, len(brain_areas) * 1.8), 5))
        style_ax(ax)

        x = np.arange(len(brain_areas))
        jitter_rng = np.random.default_rng(1)

        delta_per_area = {area: {'vals': [], 'mice': [], 'genos': []}
                          for area in brain_areas}

        for mouse in mice_all:
            geno = 'WT' if mouse.startswith('SNA') else 'ChR2'
            for area in brain_areas:
                w2t_vals  = get_mouse_session_values(mouse_sessions, mouse, area, 'W2T',    1)
                opto_vals = get_mouse_session_values(mouse_sessions, mouse, area, opto_key, tt_num)
                if w2t_vals and opto_vals:
                    delta = np.mean(w2t_vals) - np.mean(opto_vals)
                    delta_per_area[area]['vals'].append(delta)
                    delta_per_area[area]['mice'].append(mouse)
                    delta_per_area[area]['genos'].append(geno)

        # group means + bootstrap CI per area
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

        # individual mouse points
        for j, area in enumerate(brain_areas):
            for val, mouse, geno in zip(delta_per_area[area]['vals'],
                                        delta_per_area[area]['mice'],
                                        delta_per_area[area]['genos']):
                jit = jitter_rng.uniform(-0.12, 0.12)
                ax.plot(j + jit, val,
                        marker=MOUSE_MARKERS.get(mouse, 'o'),
                        color=MOUSE_COLORS.get(mouse, 'k'),
                        markersize=9, zorder=5, linestyle='none',
                        markeredgecolor='white', markeredgewidth=0.7)

        ax.axhline(0, color='black', linewidth=1.2, linestyle='--', alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(brain_areas)
        ax.set_ylabel('Δ Success Rate: W2T − Opto (%)')
        ax.set_title(f'Inhibition Effect ({opto_label})\n'
                     f'Positive = opto suppresses performance\n'
                     f'(error = 95% bootstrap CI, markers = individual mice)')
        _spaghetti_legend(ax)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        _savefig('opto_hitrate')


# ── Statistics ────────────────────────────────────────────────────────────────

def build_hitrate_dataframe(mouse_sessions):
    """One row per session: mouse, genotype, area, condition, hitrate, is_blocked."""
    trial_type_map = {'W2T': 1, 'opto_0.5s': 2, 'opto_2s': 2}
    rows = []
    for mouse, areas in mouse_sessions.items():
        geno = 'WT' if mouse.startswith('SNA') else 'ChR2'
        for area, conds in areas.items():
            for cond_key, tt_num in trial_type_map.items():
                for path in conds.get(cond_key, []):
                    val = load_hit_rate_from_session(path, tt_num)
                    if not np.isnan(val):
                        rows.append({'mouse': mouse, 'genotype': geno,
                                     'area': area, 'condition': cond_key,
                                     'hitrate': val,
                                     'blocked': is_blocked_session(path)})
    return pd.DataFrame(rows)


def run_statistics(mouse_sessions):
    df = build_hitrate_dataframe(mouse_sessions)
    if df.empty:
        print("No data for statistics.")
        return

    print("\n" + "=" * 70)
    print("STATISTICS — Success Rate")
    print("=" * 70)

    # ── Descriptive ──────────────────────────────────────────────────
    print("\n--- Descriptive (per-mouse session means) ---")
    mouse_means = (df.groupby(['mouse', 'area', 'condition'])['hitrate']
                   .mean().reset_index())
    desc = (mouse_means.groupby(['area', 'condition'])['hitrate']
            .agg(n='count', mean='mean', std='std').round(2))
    print(desc.to_string())

    # ── Session QC ───────────────────────────────────────────────────
    print("\n--- Session QC (outlier flags, threshold = 2.5 SD within mouse) ---")
    any_flag = False
    for mouse in df['mouse'].unique():
        for area in df[df['mouse'] == mouse]['area'].unique():
            for cond in df['condition'].unique():
                sub = df[(df['mouse'] == mouse) & (df['area'] == area) &
                         (df['condition'] == cond)]['hitrate'].values
                flags = flag_outlier_sessions(sub, mouse_id=f'{mouse}/{area}/{cond}')
                if flags.any():
                    any_flag = True
    if not any_flag:
        print("  No outlier sessions detected.")

    # ── Wilcoxon + FDR + effect size ─────────────────────────────────
    print("\n--- Wilcoxon signed-rank: W2T vs opto (per-mouse means, FDR-corrected) ---")
    rows_stat = []
    for area in sorted(df['area'].unique()):
        area_df = mouse_means[mouse_means['area'] == area]
        for opto_cond in ['opto_0.5s', 'opto_2s']:
            w2t  = area_df[area_df['condition'] == 'W2T'].set_index('mouse')
            opto = area_df[area_df['condition'] == opto_cond].set_index('mouse')
            common = sorted(set(w2t.index) & set(opto.index))
            n = len(common)
            if n < 2:
                rows_stat.append({'area': area, 'comparison': f'W2T vs {opto_cond}',
                                  'n': n, 'stat': np.nan, 'p': np.nan, 'r': np.nan,
                                  'w2t_mean': np.nan, 'opto_mean': np.nan})
                continue
            a = w2t.loc[common, 'hitrate'].values
            b = opto.loc[common, 'hitrate'].values
            stat, p = stats.wilcoxon(a, b)
            r = rank_biserial_r(a, b)
            rows_stat.append({'area': area, 'comparison': f'W2T vs {opto_cond}',
                               'n': n, 'stat': stat, 'p': p, 'r': r,
                               'w2t_mean': a.mean(), 'opto_mean': b.mean()})

    stat_df = pd.DataFrame(rows_stat)
    valid_p = stat_df['p'].dropna()
    if len(valid_p) > 0:
        adj = fdr_bh(valid_p.values)
        stat_df.loc[valid_p.index, 'p_fdr'] = adj
    else:
        stat_df['p_fdr'] = np.nan

    print(f"  {'Area':<6} {'Comparison':<18} {'n':>3}  {'W2T':>6}  {'Opto':>6}  "
          f"{'W':>6}  {'p':>7}  {'p_FDR':>7}  {'r':>5}  sig")
    print("  " + "-" * 75)
    for _, row in stat_df.iterrows():
        if np.isnan(row['p']):
            print(f"  {row['area']:<6} {row['comparison']:<18} {int(row['n']):>3}  — insufficient n —")
            continue
        p_fdr = row.get('p_fdr', np.nan)
        sig = '***' if p_fdr < 0.001 else '**' if p_fdr < 0.01 else '*' if p_fdr < 0.05 else 'ns'
        print(f"  {row['area']:<6} {row['comparison']:<18} {int(row['n']):>3}  "
              f"{row['w2t_mean']:>6.1f}  {row['opto_mean']:>6.1f}  "
              f"{row['stat']:>6.1f}  {row['p']:>7.4f}  {p_fdr:>7.4f}  "
              f"{row['r']:>5.2f}  {sig}")

    # ── LME: condition * area, random intercept mouse ────────────────
    print("\n--- LME: hitrate ~ condition * area  |  random intercept: mouse ---")
    print("  (reference: condition=W2T)\n")
    try:
        model = smf.mixedlm(
            "hitrate ~ C(condition, Treatment('W2T')) * C(area)",
            df, groups=df['mouse']
        )
        result = model.fit(reml=True, method='lbfgs')
        fe = result.fe_params
        ci = result.conf_int()
        pvals = result.pvalues
        print(f"  {'Parameter':<50} {'Coef':>7}  {'[0.025':>7}  {'0.975]':>7}  {'p':>7}")
        print("  " + "-" * 82)
        for param in fe.index:
            coef = fe[param]
            lo, hi = ci.loc[param]
            p = pvals[param]
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            print(f"  {param[:49]:<50} {coef:>7.2f}  {lo:>7.2f}  {hi:>7.2f}  {p:>7.4f}  {sig}")
        print(f"\n  Random effect (mouse) variance: {result.cov_re.values[0][0]:.3f}")
        print(f"  Log-likelihood: {result.llf:.2f}")
    except Exception as e:
        print(f"  LME failed: {e}")

    # ── LME: genotype × opto interaction ─────────────────────────────
    print("\n--- LME: Genotype × Condition interaction ---")
    print("  hitrate ~ condition * genotype  |  random intercept: mouse\n")
    opto_df = df[df['condition'] != 'W2T'].copy()
    if opto_df['mouse'].nunique() >= 2:
        try:
            model2 = smf.mixedlm(
                "hitrate ~ C(condition, Treatment('opto_0.5s')) * C(genotype, Treatment('ChR2'))",
                opto_df, groups=opto_df['mouse']
            )
            result2 = model2.fit(reml=True, method='lbfgs')
            fe2 = result2.fe_params
            ci2 = result2.conf_int()
            pv2 = result2.pvalues
            print(f"  {'Parameter':<55} {'Coef':>7}  {'[0.025':>7}  {'0.975]':>7}  {'p':>7}")
            print("  " + "-" * 85)
            for param in fe2.index:
                coef = fe2[param]
                lo, hi = ci2.loc[param]
                p = pv2[param]
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                print(f"  {param[:54]:<55} {coef:>7.2f}  {lo:>7.2f}  {hi:>7.2f}  {p:>7.4f}  {sig}")
            print(f"\n  Random effect (mouse) variance: {result2.cov_re.values[0][0]:.3f}")
        except Exception as e:
            print(f"  LME failed: {e}")
    else:
        print("  — insufficient mice —")

    print("=" * 70)


# ── Run all plots ─────────────────────────────────────────────────────────────
plot_hit_rates(mouse_sessions)
plot_grouped_MLA_vs_SNA(mouse_sessions)
plot_all_areas_all_mice_hitrate(mouse_sessions)
plot_hitrate_all_areas_MLA_vs_SNA(mouse_sessions)
plot_delta_hitrate(mouse_sessions)
run_statistics(mouse_sessions)

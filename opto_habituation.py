# -*- coding: utf-8 -*-
"""
opto_habituation.py
===================
Compares opto trial performance between:
  - Random sessions  (W2T_Opto_*  folders, no _blocked suffix)
  - Blocked sessions (W2T_Opto_*_blocked folders)

Scientific question:
  Does blocking opto trials (predictable context) cause habituation / different
  suppression compared to randomly interleaved opto trials?

Detection logic:
  _blocked suffix in the filepath → blocked session
  no _blocked suffix              → random session

Metrics per session:
  - Hit rate (opto trials, TT=2)
  - Median response latency (opto trials, TT=2)
  - Hit rate (W2T trials, TT=1) — to check whether W2T baseline shifts too

Plots:
  1. Bar + individual points: random vs blocked per brain area (hit rate & latency)
  2. Within-mouse spaghetti: random vs blocked (paired, per area)
  3. Trial-by-trial opto hit-rate within session (first vs second half — within-session habituation)

Statistics:
  - Wilcoxon signed-rank (matched per mouse, random vs blocked)
  - LME: hitrate ~ session_type * area | random intercept: mouse
"""

import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
from collections import defaultdict
from scipy import stats
import statsmodels.formula.api as smf
from pathlib import Path

from bpod_utils import (load_mat, is_hit, latency as lat_fn, MIN_TRIALS,
                        fdr_bh, rank_biserial_r, bootstrap_ci,
                        is_blocked_session, extract_day_from_path)
from session_config import mouse_sessions
from plot_config import MOUSE_COLORS_SHORT as MOUSE_COLORS, MOUSE_MARKERS, OPTO_DUR_COLORS, SESSION_TYPE_COLORS, style_ax

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

def load_session_metrics(filepath):
    """
    Returns dict with:
      - hit_rate_opto   : % hits on TT=2 trials
      - latency_opto    : median latency on TT=2 hit trials (s)
      - hit_rate_w2t    : % hits on TT=1 trials
      - n_opto / n_w2t  : trial counts
      - first_half_opto : hit rate in first 50% of opto trials
      - second_half_opto: hit rate in second 50% of opto trials
      - blocked         : bool
    """
    if not os.path.exists(filepath):
        return None
    data = load_mat(filepath)
    sd = data.get('SessionData', None)
    if sd is None:
        return None
    trialtypes = sd.get('TrialTypes', None)
    raw = sd.get('RawEvents', None)
    if trialtypes is None or raw is None:
        return None
    trials = raw.get('Trial', None) if isinstance(raw, dict) else getattr(raw, 'Trial', None)
    if trials is None:
        return None
    if isinstance(trials, dict):
        trials = [trials]
    if len(trials) < MIN_TRIALS:
        return None

    opto_hits, opto_lats, opto_idx = [], [], []
    w2t_hits = []
    for i, trial in enumerate(trials):
        if i >= len(trialtypes):
            break
        tt = trialtypes[i]
        hit = is_hit(trial)
        if tt == 2:
            opto_hits.append(1 if hit else 0)
            opto_idx.append(i)
            if hit:
                l = lat_fn(trial)
                if not np.isnan(l):
                    opto_lats.append(l)
        elif tt == 1:
            w2t_hits.append(1 if hit else 0)

    if not opto_hits:
        return None

    n = len(opto_hits)
    half = n // 2
    first_half  = np.mean(opto_hits[:half]) * 100 if half > 0 else np.nan
    second_half = np.mean(opto_hits[half:]) * 100 if n - half > 0 else np.nan

    return {
        'hit_rate_opto':    np.mean(opto_hits) * 100,
        'latency_opto':     np.nanmedian(opto_lats) if opto_lats else np.nan,
        'hit_rate_w2t':     np.mean(w2t_hits) * 100 if w2t_hits else np.nan,
        'n_opto':           n,
        'n_w2t':            len(w2t_hits),
        'first_half_opto':  first_half,
        'second_half_opto': second_half,
        'blocked':          is_blocked_session(filepath),
    }


# ── Build DataFrame ───────────────────────────────────────────────────────────

def build_habituation_dataframe(mouse_sessions):
    """One row per session across all opto conditions."""
    opto_keys = ['opto_0.5s', 'opto_2s']
    rows = []
    for mouse, areas in mouse_sessions.items():
        geno = 'WT' if mouse.startswith('SNA') else 'ChR2'
        for area, conds in areas.items():
            seen_paths = set()
            for cond_key in opto_keys:
                for path in conds.get(cond_key, []):
                    if path in seen_paths:
                        continue
                    seen_paths.add(path)
                    m = load_session_metrics(path)
                    if m is None:
                        continue
                    row = {
                        'mouse':            mouse,
                        'genotype':         geno,
                        'area':             area,
                        'opto_duration':    cond_key,
                        'session_type':     'blocked' if m['blocked'] else 'random',
                        'hit_rate_opto':    m['hit_rate_opto'],
                        'latency_opto':     m['latency_opto'],
                        'hit_rate_w2t':     m['hit_rate_w2t'],
                        'n_opto':           m['n_opto'],
                        'first_half_opto':  m['first_half_opto'],
                        'second_half_opto': m['second_half_opto'],
                        'path':             path,
                    }
                    rows.append(row)
                    # Also aggregate into 'All' so every session appears there
                    if area != 'All':
                        rows.append({**row, 'area': 'All'})
    return pd.DataFrame(rows)


# ── Plots ─────────────────────────────────────────────────────────────────────

def _spaghetti_legend(ax):
    handles = [mlines.Line2D([], [], color=MOUSE_COLORS[m], marker=MOUSE_MARKERS[m],
                             linestyle='-', markersize=6, label=m)
               for m in MOUSE_COLORS]
    ax.legend(handles=handles, fontsize=7, title='Mouse', title_fontsize=7,
              loc='upper right', framealpha=0.7)


def _add_points_and_lines(ax, df_random, df_blocked, x_random, x_blocked,
                           col_key, jitter=0.04):
    """Plot per-mouse points + connecting line between random and blocked."""
    rng = np.random.default_rng(42)
    for mouse in df_random['mouse'].unique():
        r_row = df_random[df_random['mouse'] == mouse]
        b_row = df_blocked[df_blocked['mouse'] == mouse]
        if r_row.empty or b_row.empty:
            continue
        r_val = r_row[col_key].mean()
        b_val = b_row[col_key].mean()
        jit = rng.uniform(-jitter, jitter)
        xr = x_random + jit
        xb = x_blocked + jit
        ax.plot([xr, xb], [r_val, b_val],
                color=MOUSE_COLORS.get(mouse, 'gray'), linewidth=1.2, alpha=0.6, zorder=3)
        for xp, yp in [(xr, r_val), (xb, b_val)]:
            ax.plot(xp, yp, marker=MOUSE_MARKERS.get(mouse, 'o'),
                    color=MOUSE_COLORS.get(mouse, 'gray'), markersize=7,
                    zorder=4, linestyle='none',
                    markeredgecolor='white', markeredgewidth=0.5)


def plot_random_vs_blocked_bars(df, metric, ylabel, title_suffix):
    """Bar chart: random vs blocked per brain area, per opto duration."""
    brain_areas  = sorted(df['area'].unique())
    opto_durs    = sorted(df['opto_duration'].unique())
    n_areas      = len(brain_areas)
    n_durs       = len(opto_durs)
    bar_w        = 0.3
    group_gap    = 0.8
    colors_rand  = [OPTO_DUR_COLORS['opto_0.5s']['random'],  OPTO_DUR_COLORS['opto_2s']['random']]
    colors_block = [OPTO_DUR_COLORS['opto_0.5s']['blocked'], OPTO_DUR_COLORS['opto_2s']['blocked']]

    fig, axes = plt.subplots(1, n_areas, figsize=(max(5, n_areas * 4), 5), sharey=True)
    if n_areas == 1:
        axes = [axes]

    for ax, area in zip(axes, brain_areas):
        style_ax(ax)
        for d_idx, opto_dur in enumerate(opto_durs):
            sub = df[(df['area'] == area) & (df['opto_duration'] == opto_dur)]
            x_r = d_idx * group_gap
            x_b = x_r + bar_w

            for sess_type, xpos, col in [
                ('random',  x_r, colors_rand[d_idx]),
                ('blocked', x_b, colors_block[d_idx]),
            ]:
                vals_per_mouse = (sub[sub['session_type'] == sess_type]
                                  .groupby('mouse')[metric].mean())
                if vals_per_mouse.empty:
                    continue
                mean_v = vals_per_mouse.mean()
                lo, hi = bootstrap_ci(vals_per_mouse.values)
                ax.bar(xpos, mean_v, bar_w, color=col,
                       yerr=[[mean_v - lo], [hi - mean_v]], capsize=4,
                       error_kw={'elinewidth': 1.2},
                       label=f'{opto_dur.replace("_"," ")} {sess_type}' if area == brain_areas[0] else '')

            # spaghetti overlay
            sub_r = sub[sub['session_type'] == 'random']
            sub_b = sub[sub['session_type'] == 'blocked']
            _add_points_and_lines(ax, sub_r, sub_b, x_r, x_b, metric)

        ax.set_xticks([d * group_gap + bar_w/2 for d in range(n_durs)])
        ax.set_xticklabels([d.replace('_', ' ') for d in opto_durs])
        ax.set_title(area)
        ax.grid(axis='y', alpha=0.3)

    axes[0].set_ylabel(ylabel)
    fig.suptitle(f'{title_suffix}\nRandom (dark) vs Blocked (light) — lines = individual mice, error = 95% bootstrap CI',
                 fontsize=10)
    if n_areas > 0:
        axes[0].legend(fontsize=7, loc='upper right')
    plt.tight_layout()
    _savefig('opto_habituation')


def plot_within_session_habituation(df):
    """
    Within-session: first half vs second half of opto trials.
    Shows whether hit rate drops within a session (habituation to opto).
    """
    brain_areas = sorted(df['area'].unique())
    opto_durs   = sorted(df['opto_duration'].unique())
    colors      = SESSION_TYPE_COLORS
    session_types = ['random', 'blocked']

    for area in brain_areas:
        n_durs = len(opto_durs)
        fig, axes = plt.subplots(1, n_durs, figsize=(n_durs * 5, 4), sharey=True)
        if n_durs == 1:
            axes = [axes]
        fig.suptitle(f'Within-session habituation — {area}\n'
                     f'(first vs second half of opto trials)', fontsize=10)

        for ax, opto_dur in zip(axes, opto_durs):
            style_ax(ax)
            sub = df[(df['area'] == area) & (df['opto_duration'] == opto_dur)]
            x_pos = 0
            xticks, xlabels = [], []
            for sess_type in session_types:
                ss = sub[sub['session_type'] == sess_type]
                if ss.empty:
                    continue
                # per-mouse means of first/second half
                first_by_mouse  = ss.groupby('mouse')['first_half_opto'].mean()
                second_by_mouse = ss.groupby('mouse')['second_half_opto'].mean()
                common_mice = first_by_mouse.index.intersection(second_by_mouse.index)
                if common_mice.empty:
                    continue

                x_f, x_s = x_pos, x_pos + 0.5
                col = colors[sess_type]
                for xp, vals_m, label in [(x_f, first_by_mouse[common_mice], '1st half'),
                                           (x_s, second_by_mouse[common_mice], '2nd half')]:
                    mean_v = vals_m.mean()
                    lo, hi = bootstrap_ci(vals_m.values)
                    ax.bar(xp, mean_v, 0.4, color=col, alpha=(0.9 if label == '1st half' else 0.55),
                           yerr=[[mean_v - lo], [hi - mean_v]], capsize=3,
                           error_kw={'elinewidth': 1.0})
                    for mouse in common_mice:
                        ax.plot(xp, vals_m[mouse],
                                marker=MOUSE_MARKERS.get(mouse, 'o'),
                                color=MOUSE_COLORS.get(mouse, 'gray'),
                                markersize=6, zorder=5, linestyle='none',
                                markeredgecolor='white', markeredgewidth=0.5)

                # connect first→second per mouse
                for mouse in common_mice:
                    ax.plot([x_f, x_s],
                            [first_by_mouse[mouse], second_by_mouse[mouse]],
                            color=MOUSE_COLORS.get(mouse, 'gray'),
                            linewidth=1.2, alpha=0.6, zorder=3)

                xticks += [x_f, x_s]
                xlabels += [f'{sess_type[:4]}\n1st', f'{sess_type[:4]}\n2nd']
                x_pos += 1.4

            ax.set_xticks(xticks)
            ax.set_xticklabels(xlabels, fontsize=8)
            ax.set_title(opto_dur.replace('_', ' '))
            ax.grid(axis='y', alpha=0.3)

        axes[0].set_ylabel('Opto Success Rate (%)')
        _spaghetti_legend(axes[-1])
        plt.tight_layout()
        _savefig('opto_habituation')


# ── Statistics ────────────────────────────────────────────────────────────────

def run_habituation_statistics(df):
    # Exclude 'All' pseudo-area — those rows are duplicates of specific areas
    # and would inflate n and create pseudo-replication in statistical tests.
    df = df[df['area'] != 'All'].copy()

    print("\n" + "=" * 70)
    print("STATISTICS — Random vs Blocked Opto Sessions (Habituation)")
    print("=" * 70)

    # ── Descriptive ──────────────────────────────────────────────────
    print("\n--- Descriptive (per-mouse session means) ---")
    desc = (df.groupby(['area', 'opto_duration', 'session_type'])
            [['hit_rate_opto', 'latency_opto', 'hit_rate_w2t']]
            .agg(['mean', 'std', 'count']).round(2))
    print(desc.to_string())

    # ── Within-session habituation: first vs second half ─────────────
    print("\n--- Within-session: first half vs second half opto hit rate ---")
    print(f"  {'Area':<6} {'Opto dur':<12} {'Type':<8} {'n':>3}  "
          f"{'1st half':>8}  {'2nd half':>8}  {'W':>6}  {'p':>7}  sig")
    print("  " + "-" * 65)
    for area in sorted(df['area'].unique()):
        for opto_dur in sorted(df['opto_duration'].unique()):
            for stype in ['random', 'blocked']:
                sub = df[(df['area'] == area) & (df['opto_duration'] == opto_dur) &
                         (df['session_type'] == stype)]
                fm = sub.groupby('mouse')['first_half_opto'].mean()
                sm = sub.groupby('mouse')['second_half_opto'].mean()
                common = sorted(set(fm.index) & set(sm.index))
                n = len(common)
                if n < 2:
                    print(f"  {area:<6} {opto_dur:<12} {stype:<8} {n:>3}  — n<2 —")
                    continue
                a, b = fm[common].values, sm[common].values
                wstat, p = stats.wilcoxon(a, b)
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
                print(f"  {area:<6} {opto_dur:<12} {stype:<8} {n:>3}  "
                      f"{np.mean(a):>8.1f}  {np.mean(b):>8.1f}  "
                      f"{wstat:>6.1f}  {p:>7.4f}  {sig}")

    # ── Wilcoxon: random vs blocked (matched per mouse) ───────────────
    print("\n--- Wilcoxon: random vs blocked (per-mouse means, FDR-corrected) ---")
    rows_stat = []
    for area in sorted(df['area'].unique()):
        for opto_dur in sorted(df['opto_duration'].unique()):
            sub = df[(df['area'] == area) & (df['opto_duration'] == opto_dur)]
            for metric in ['hit_rate_opto', 'latency_opto']:
                r_m = sub[sub['session_type'] == 'random'].groupby('mouse')[metric].mean()
                b_m = sub[sub['session_type'] == 'blocked'].groupby('mouse')[metric].mean()
                common = sorted(set(r_m.index) & set(b_m.index))
                n = len(common)
                if n < 2:
                    rows_stat.append({'area': area, 'opto_dur': opto_dur,
                                      'metric': metric, 'n': n,
                                      'p': np.nan, 'r': np.nan,
                                      'rand_mean': np.nan, 'block_mean': np.nan})
                    continue
                a, b = r_m[common].values, b_m[common].values
                wstat, p = stats.wilcoxon(a, b)
                r = rank_biserial_r(a, b)
                rows_stat.append({'area': area, 'opto_dur': opto_dur,
                                  'metric': metric, 'n': n,
                                  'p': p, 'r': r,
                                  'rand_mean': np.mean(a), 'block_mean': np.mean(b)})

    stat_df = pd.DataFrame(rows_stat)
    valid_p = stat_df['p'].dropna()
    if len(valid_p) > 0:
        stat_df.loc[valid_p.index, 'p_fdr'] = fdr_bh(valid_p.values)
    else:
        stat_df['p_fdr'] = np.nan

    print(f"\n  {'Area':<6} {'Opto dur':<12} {'Metric':<16} {'n':>3}  "
          f"{'Random':>8}  {'Blocked':>8}  {'p':>7}  {'p_FDR':>7}  {'r':>5}  sig")
    print("  " + "-" * 80)
    for _, row in stat_df.iterrows():
        if np.isnan(row['p']):
            print(f"  {row['area']:<6} {row['opto_dur']:<12} {row['metric']:<16} "
                  f"{int(row['n']):>3}  — insufficient n —")
            continue
        p_fdr = row.get('p_fdr', np.nan)
        sig = '***' if p_fdr < 0.001 else '**' if p_fdr < 0.01 else '*' if p_fdr < 0.05 else 'ns'
        print(f"  {row['area']:<6} {row['opto_dur']:<12} {row['metric']:<16} "
              f"{int(row['n']):>3}  {row['rand_mean']:>8.2f}  {row['block_mean']:>8.2f}  "
              f"{row['p']:>7.4f}  {p_fdr:>7.4f}  {row['r']:>5.2f}  {sig}")

    # ── LME: session_type * area ──────────────────────────────────────
    for metric, label in [('hit_rate_opto', 'Success Rate (%)'), ('latency_opto', 'Latency (s)')]:
        print(f"\n--- LME: {label} ~ session_type * area  |  random intercept: mouse ---\n")
        sub_df = df[['mouse', 'area', 'session_type', metric]].dropna()
        if sub_df['mouse'].nunique() < 2:
            print("  — insufficient mice —")
            continue
        try:
            model = smf.mixedlm(
                f"{metric} ~ C(session_type, Treatment('random')) * C(area)",
                sub_df, groups=sub_df['mouse']
            )
            res = model.fit(reml=True, method='lbfgs')
            fe, ci_lme, pv = res.fe_params, res.conf_int(), res.pvalues
            print(f"  {'Parameter':<50} {'Coef':>7}  {'[0.025':>7}  {'0.975]':>7}  {'p':>7}")
            print("  " + "-" * 82)
            for param in fe.index:
                coef = fe[param]; lo, hi = ci_lme.loc[param]; p = pv[param]
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                print(f"  {param[:49]:<50} {coef:>7.2f}  {lo:>7.2f}  {hi:>7.2f}  {p:>7.4f}  {sig}")
            print(f"\n  Random effect (mouse) variance: {res.cov_re.values[0][0]:.3f}")
        except Exception as e:
            print(f"  LME failed: {e}")

    print("=" * 70)


# ── Run ───────────────────────────────────────────────────────────────────────

df = build_habituation_dataframe(mouse_sessions)

if df.empty:
    print("No opto session data found.")
else:
    print(f"Loaded {len(df)} opto sessions across "
          f"{df['mouse'].nunique()} mice, {df['area'].nunique()} areas.")
    print(f"Random sessions: {(df['session_type']=='random').sum()}  |  "
          f"Blocked sessions: {(df['session_type']=='blocked').sum()}")

    plot_random_vs_blocked_bars(df, 'hit_rate_opto', 'Opto Success Rate (%)',
                                'Opto Success Rate: Random vs Blocked')
    plot_random_vs_blocked_bars(df, 'latency_opto', 'Median Latency (s)',
                                'Opto Latency: Random vs Blocked')
    plot_random_vs_blocked_bars(df, 'hit_rate_w2t', 'W2T Success Rate (%)',
                                'W2T Baseline: Random vs Blocked sessions')
    plot_within_session_habituation(df)
    run_habituation_statistics(df)

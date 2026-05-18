# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 11:30:27 2025

@author: AG_Larkum
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats, optimize
from collections import defaultdict
from pathlib import Path

from bpod_utils import load_mat, is_hit, latency, extract_day_from_path, BASE_PATH
from session_config import sorted_mice as _sorted_mice, mice as _mice_cfg
from plot_config import MOUSE_COLORS, style_ax

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

# ── Learn-index constants (computed by calc_expert_def.py) ───────────────────
RT_NAIVE   = 1.7757
RT_EXPERT  = 0.50
SEM_NAIVE  = 0.0866
SEM_EXPERT = 0.0698

BINSIZE   = 10
threshold = 0.7


# ── Helper functions ──────────────────────────────────────────────────────────
def is_valid_training_session(filename, side):
    """Allow only proper W2T sessions and exclude October sessions (pilot/early)."""
    pattern = rf"_W2T_{side.split('_')[-1]}_\d{{8}}_\d{{6}}\.mat$"
    if not re.search(pattern, filename):
        return False
    match = re.search(r'_(\d{8})_', filename)
    if not match:
        return False
    return int(match.group(1)[4:6]) != 10  # exclude October


def interpolate_nans(array):
    array = np.array(array, dtype=float)
    return pd.Series(array).interpolate(method='linear', limit_direction='both').to_numpy()


def get_SEM(values, binsize=0):
    values = np.asarray(values)
    n_trials = len(values)
    if binsize <= 0:
        raise ValueError("Bitte binsize > 0 setzen!")
    sem_values, center_indices = [], []
    for i in range(n_trials - binsize + 1):
        window = values[i:i + binsize]
        valid = window[~np.isnan(window)]
        sem_values.append(np.std(valid, ddof=1) / np.sqrt(len(valid)) if len(valid) > 1 else np.nan)
        center_indices.append(i + binsize // 2)
    sem_full = np.full(n_trials, np.nan)
    for idx, s in zip(center_indices, sem_values):
        sem_full[idx] = s
    return sem_full


def get_hitrate(mouse, binsize=0):
    session_data = mouse.full_data
    n_trials = len(session_data)
    if binsize > 0:
        return [session_data.iloc[s:s + binsize]['success'].sum() / binsize
                for s in range(n_trials - binsize + 1)]
    else:
        hits = session_data['success'].sum()
        return [hits / n_trials if n_trials > 0 else 0]


def calc_LearningScore(mouse_data, binsize=10, plot=False):
    hit_rate = np.array(get_hitrate(mouse_data, binsize=binsize))
    n_trials = len(hit_rate)

    RT_trialwise = mouse_data.full_data['response_t'].copy()
    RT_trialwise[~mouse_data.full_data['success']] = np.nan

    rt_sem = get_SEM(RT_trialwise, binsize=binsize)
    rt_sem_interp = interpolate_nans(rt_sem)[:n_trials]
    response_time = np.asarray(RT_trialwise)[:n_trials]

    norm_hit_rate      = hit_rate
    norm_response_time = np.clip((RT_NAIVE - response_time) / (RT_NAIVE - RT_EXPERT), 0, 1)
    norm_rt_sem        = np.clip((SEM_NAIVE - rt_sem_interp) / (SEM_NAIVE - SEM_EXPERT), 0, 1)

    learning_score = (0.6 * norm_hit_rate +
                      0.35 * norm_response_time +
                      0.05 * norm_rt_sem)
    return interpolate_nans(learning_score)


def process_session_for_learning_vector(file_path, binsize=10):
    data = load_mat(file_path)
    trials = data['SessionData']['RawEvents']['Trial']
    if isinstance(trials, dict):
        trials = [trials]
    if len(trials) < 40:
        print(f"Session {os.path.basename(file_path)} übersprungen: nur {len(trials)} Trials")
        return np.array([])

    df = pd.DataFrame({
        'trialType':  ['W2T'] * len(trials),
        'success':    [is_hit(t) for t in trials],
        'response_t': [latency(t) for t in trials],
    })

    class MouseData:
        def __init__(self, df):
            self.full_data = df

    return calc_LearningScore(MouseData(df), binsize=binsize)


# ── Main loop ─────────────────────────────────────────────────────────────────
mouse_dirs = []
for _m in _sorted_mice:
    _bp = _mice_cfg[_m].get('base_path', BASE_PATH)
    _d  = os.path.join(_bp, _m)
    if os.path.isdir(_d):
        mouse_dirs.append(_d)

all_mice_data = []
all_days = set()

for mouse_dir in mouse_dirs:
    day_scores = defaultdict(list)
    found = False

    for side in ("W2T_right", "W2T_left"):
        # Support both nested (Training/task) and flat (task) folder structures
        session_dir = os.path.join(mouse_dir, f"Training/{side}/Session Data")
        if not os.path.exists(session_dir):
            session_dir = os.path.join(mouse_dir, f"{side}/Session Data")
        if not os.path.exists(session_dir):
            continue
        found = True
        mat_files = sorted([os.path.join(session_dir, f)
                            for f in os.listdir(session_dir)
                            if f.endswith('.mat') and is_valid_training_session(f, side)])

        for fpath in mat_files:
            print(f"{os.path.basename(mouse_dir)} | {side} | Verarbeite {os.path.basename(fpath)}")
            try:
                learning_score = process_session_for_learning_vector(fpath, binsize=BINSIZE)
                if len(learning_score) > 0:
                    avg_score = np.nanmean(learning_score)
                    day = extract_day_from_path(fpath)
                    if day is not None:
                        day_scores[day].append(avg_score)
                        all_days.add(day)
            except Exception as e:
                print(f"Fehler in {os.path.basename(fpath)}: {e}")

    if not found:
        print(f"Achtung: Weder W2T_right noch W2T_left für {os.path.basename(mouse_dir)} gefunden.")
        continue

    all_mice_data.append((day_scores, os.path.basename(mouse_dir)))

# Use relative day index per mouse (calendar dates ignored for alignment).
# Each mouse's sorted days → positions 0, 1, 2, …
per_mouse_vals = []
for day_scores, _ in all_mice_data:
    sorted_days = sorted(day_scores.keys())
    per_mouse_vals.append([np.nanmean(day_scores[d]) for d in sorted_days])

max_days = max((len(v) for v in per_mouse_vals), default=0)
score_matrix = np.full((len(all_mice_data), max_days), np.nan)
for i_mouse, vals in enumerate(per_mouse_vals):
    score_matrix[i_mouse, :len(vals)] = vals

all_days = list(range(max_days))   # 0-indexed, used as x_days below

mean_scores = np.nanmean(score_matrix, axis=0)
sem_scores  = stats.sem(score_matrix, axis=0, nan_policy='omit')

learning_session = next(
    (i + 1 for i, score in enumerate(mean_scores) if score >= threshold),
    None
)

# ── Sigmoid fit ──────────────────────────────────────────────────────────────
def sigmoid(x, L, x0, k, b):
    """4-parameter logistic: L = max, x0 = inflection, k = slope, b = min."""
    return b + (L - b) / (1 + np.exp(-k * (x - x0)))


def fit_sigmoid(x, y):
    """Fit sigmoid to learning curve. Returns (params, x_fine, y_fine, success)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = ~np.isnan(y)
    if valid.sum() < 5:
        return None, None, None, False
    try:
        p0 = [np.nanmax(y), x[valid][len(x[valid])//2], 0.5, np.nanmin(y)]
        bounds = ([0, x.min()-1, 0, 0], [1.5, x.max()+1, 10, 1])
        popt, _ = optimize.curve_fit(sigmoid, x[valid], y[valid],
                                     p0=p0, bounds=bounds, maxfev=5000)
        x_fine = np.linspace(x.min(), x.max(), 300)
        y_fine = sigmoid(x_fine, *popt)
        return popt, x_fine, y_fine, True
    except Exception:
        return None, None, None, False


x_days = np.arange(len(all_days))

fig, ax = plt.subplots(figsize=(14, 6))
style_ax(ax)

# Group mean ± SEM
ax.plot(x_days, mean_scores, color='black', linewidth=2.5, marker='o',
        markersize=6, label='Group Mean', zorder=6)
ax.fill_between(x_days, mean_scores - sem_scores, mean_scores + sem_scores,
                color='black', alpha=0.15, label='SEM')

# Group sigmoid fit
popt_grp, x_fine_grp, y_fine_grp, ok_grp = fit_sigmoid(x_days, mean_scores)
if ok_grp:
    inflection_day = popt_grp[1]  # x0
    ax.plot(x_fine_grp, y_fine_grp, color='navy', linewidth=2.5,
            linestyle='--', label=f'Sigmoid fit (inflection day {inflection_day:.1f})', zorder=7)
    ax.axvline(inflection_day, color='navy', linestyle=':', linewidth=1.5, alpha=0.6)

if learning_session is not None:
    ax.axvline(x=learning_session - 1, color='red', linestyle=':', linewidth=2,
               label=f'Learn Threshold (Day {learning_session})')
    ax.scatter(learning_session - 1, mean_scores[learning_session - 1],
               color='red', zorder=10, s=80)

ax.axhline(threshold, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xticks(x_days)
ax.set_xticklabels(np.arange(1, len(all_days) + 1))
ax.set_xlabel("Day")
ax.set_ylabel("Learnscore")
ax.set_title("Learnscore Mean ± SEM + Sigmoid Fit")
y_lo = max(0.0, float(np.nanmin(mean_scores - sem_scores)) - 0.05)
y_hi = min(1.0, float(np.nanmax(mean_scores + sem_scores)) + 0.05)
ax.set_ylim(y_lo, y_hi)
ax.grid(True, alpha=0.3)
ax.legend(loc='upper left', fontsize=8, framealpha=0.8)
plt.tight_layout()
_savefig('learn_index_curve')

# Heatmap: mice × days
fig, ax = plt.subplots(figsize=(max(8, len(all_days) * 0.6), 2.5))
im = ax.imshow(score_matrix, aspect='auto', cmap='RdYlGn',
               vmin=0, vmax=1, interpolation='nearest')
ax.set_yticks(range(len(all_mice_data)))
ax.set_yticklabels([m[1].split('_(')[0] for m in all_mice_data], fontsize=8)
ax.set_xticks(range(len(all_days)))
ax.set_xticklabels(range(1, len(all_days) + 1), fontsize=7)
ax.set_xlabel("Day")
ax.set_title("Heatmap — Learnscore per Mouse × Day")
plt.colorbar(im, ax=ax, label='Learnscore')
plt.tight_layout()
_savefig('learn_index_heatmap')

if learning_session is not None:
    print(f"Lernzeitpunkt (Gruppenmittel) erreicht an Tag {learning_session} "
          f"mit Score {mean_scores[learning_session - 1]:.2f}")
else:
    print("Kein Lernzeitpunkt (Score >= Schwelle) im Gruppendurchschnitt erreicht.")

# ── Per-mouse summary ─────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("MICE-SUMMARY (Learn Day = first Tag >= 0.7)")
print("=" * 50)

learn_days = []
max_scores = []
for i_mouse, (day_scores, mouse_id) in enumerate(all_mice_data):
    mouse_scores_full = score_matrix[i_mouse, :]  # already correctly keyed by day index
    mouse_scores = [v for v in mouse_scores_full if not np.isnan(v)]
    learn_day = next((i + 1 for i, s in enumerate(mouse_scores) if s >= 0.75), None)
    popt_m, _, _, ok_m = fit_sigmoid(x_days, mouse_scores_full)
    inflection = f"{popt_m[1]:.1f}" if ok_m else "—"
    learn_days.append(learn_day)
    max_scores.append(np.nanmax(mouse_scores) if mouse_scores else np.nan)
    print(f"{mouse_id}: Learn Day {learn_day or 'None'} | Sigmoid inflection: Day {inflection} "
          f"| Max Score: {max_scores[-1]:.3f}")

valid_learn_days = [d for d in learn_days if d is not None]
if valid_learn_days:
    print(f"\nGRUPPENSTATISTIK:")
    print(f"Mean Learn Day: {np.nanmean(valid_learn_days):.1f} ± "
          f"{np.nanstd(valid_learn_days):.1f} (N={len(valid_learn_days)}/4 Mice)")
else:
    print("\nKeine Maus hat Schwelle erreicht")
print("=" * 50)

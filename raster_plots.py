# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 16:09:06 2025

@author: Larkum_Practical_01
"""
"""
Rasterplots der Latenz (Response time) für W2T- und W2T_Opto-Trials
für alle Mäuse und Sessions (Training + Expert).
Hit/Miss-Bewertung und Latenzdefinition wie in latency_mean-SEM.py.
"""

import os
import re
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde, exponnorm
from pathlib import Path

from bpod_utils import (load_mat, is_hit, latency, extract_sort_key,
                        extract_day_from_filename, BASE_PATH, MIN_TRIALS,
                        build_session_path)
from session_config import sorted_mice
from plot_config import TRIAL_COLORS, MOUSE_COLORS

_OUTDIR = Path(os.environ.get('BPOD_OUTDIR',
               str(Path(__file__).parent / 'figures' / 'bpod' / 'default')))
_OUTDIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# 2) Tages-Helper
# ---------------------------------------------------------------------

def day_index_within_mouse(session_files):
    """liefert Dict: fullpath -> (day_str, day_idx) für diese Maus."""
    days = []
    for p in session_files:
        d = extract_day_from_filename(os.path.basename(p))
        if d is not None:
            days.append(d)

    unique_days = sorted(set(days))
    mapping = {}
    for p in session_files:
        d = extract_day_from_filename(os.path.basename(p))
        if d is None:
            continue
        idx = unique_days.index(d) + 1  # Tag 1,2,...
        mapping[p] = (d, idx)
    return mapping

def build_day_map_for_paths(paths):
    """für Expert: Dict path -> (day_str, idx) nur aus dieser Pfadliste."""
    days = []
    for p in paths:
        d = extract_day_from_filename(os.path.basename(p))
        if d is not None:
            days.append(d)
    unique_days = sorted(set(days))
    mapping = {}
    for p in paths:
        d = extract_day_from_filename(os.path.basename(p))
        if d is None:
            continue
        idx = unique_days.index(d) + 1
        mapping[p] = (d, idx)
    return mapping

# ---------------------------------------------------------------------
# 3) Maus-Setup (Training) + Expert-Pfade
# ---------------------------------------------------------------------

from session_config import mice

# Expert-Folder (nur opto_0.5s / opto_2s, kein W2T-Key)
mouse_sessions = {
    'MLA-026805': {
        'S1': {
            'opto_0.5s': ['D:/bpod/MLA-026805_(2)/Expert/W2T_Opto_right/Session Data/MLA-026805_(2)_W2T_Opto_right_20250916_132254.mat'],
        },
        'M1': {
            'opto_0.5s': ['D:/bpod/MLA-026805_(2)/Expert/W2T_Opto_right/Session Data/MLA-026805_(2)_W2T_Opto_right_20250916_125033.mat', 
                          'D:/bpod/MLA-026805_(2)/Expert/W2T_Opto_right/Session Data/MLA-026805_(2)_W2T_Opto_right_20250916_125633.mat', 
                          'D:/bpod/MLA-026805_(2)/Expert/W2T_Opto_right_blocked/Session Data/MLA-026805_(2)_W2T_Opto_right_blocked_20250917_142117.mat', 
                          'D:/bpod/MLA-026805_(2)/Expert/W2T_Opto_right/Session Data/MLA-026805_(2)_W2T_Opto_right_20250917_144321.mat'],
        },
    },
    'MLA-026806': {
        'S1': {
            'opto_0.5s': ['D:/bpod/MLA-026806_(3)/Expert/W2T_Opto_right/Session Data/MLA-026806_(3)_W2T_Opto_right_20250917_131243.mat', 
                          'D:/bpod/MLA-026806_(3)/Expert/W2T_Opto_right_blocked/Session Data/MLA-026806_(3)_W2T_Opto_right_blocked_20250917_124955.mat'],
            'opto_2s': ['D:/bpod/MLA-026806_(3)/Expert/W2T_Opto_right/Session Data/MLA-026806_(3)_W2T_Opto_right_20250922_153745.mat', 
                        'D:/bpod/MLA-026806_(3)/Expert/W2T_Opto_right_blocked/Session Data/MLA-026806_(3)_W2T_Opto_right_blocked_20250922_160545.mat']
        },
        'M1': {
            'opto_0.5s': ['D:/bpod/MLA-026806_(3)/Expert/W2T_Opto_right/Session Data/MLA-026806_(3)_W2T_Opto_right_20250916_145114.mat', 
                          'D:/bpod/MLA-026806_(3)/Expert/W2T_Opto_right/Session Data/MLA-026806_(3)_W2T_Opto_right_20250916_152034.mat'],
            'opto_2s': ['D:/bpod/MLA-026806_(3)/Expert/W2T_Opto_right/Session Data/MLA-026806_(3)_W2T_Opto_right_20250922_153745.mat',
                        'D:/bpod/MLA-026806_(3)/Expert/W2T_Opto_right_blocked/Session Data/MLA-026806_(3)_W2T_Opto_right_blocked_20250922_160545.mat']
        },
        'M2': {
            'opto_0.5s': ['D:/bpod/MLA-026806_(3)/Expert/W2T_Opto_right_blocked/Session Data/MLA-026806_(3)_W2T_Opto_right_blocked_20250918_132651.mat'],
            'opto_2s': ['D:/bpod/MLA-026806_(3)/Expert/W2T_Opto_right/Session Data/MLA-026806_(3)_W2T_Opto_right_20250924_135305.mat', 
                        'D:/bpod/MLA-026806_(3)/Expert/W2T_Opto_right_blocked/Session Data/MLA-026806_(3)_W2T_Opto_right_blocked_20250924_132852.mat']
        },
        'mPFC': {
            'opto_0.5s': ['D:/bpod/MLA-026806_(3)/Expert/W2T_Opto_right/Session Data/MLA-026806_(3)_W2T_Opto_right_20250919_134643.mat', 
                          'D:/bpod/MLA-026806_(3)/Expert/W2T_Opto_right_blocked/Session Data/MLA-026806_(3)_W2T_Opto_right_blocked_20250919_132132.mat'],
        }
    },
    'MLA-026807': {
        'S1': {
            'opto_0.5s': ['D:/bpod/MLA-026807_(4)/Expert/W2T_Opto_left/Session Data/MLA-026807_(4)_W2T_Opto_left_20250918_163451.mat', 
                          'D:/bpod/MLA-026807_(4)/Expert/W2T_Opto_left_blocked/Session Data/MLA-026807_(4)_W2T_Opto_left_blocked_20250918_161128.mat', 
                          'D:/bpod/MLA-026807_(4)/Expert/W2T_Opto_left_blocked/Session Data/MLA-026807_(4)_W2T_Opto_left_blocked_20250925_164056.mat', 
                          'D:/bpod/MLA-026807_(4)/Expert/W2T_Opto_left_blocked/Session Data/MLA-026807_(4)_W2T_Opto_left_blocked_20250925_164329.mat'],
            'opto_2s': ['D:/bpod/MLA-026807_(4)/Expert/W2T_Opto_left/Session Data/MLA-026807_(4)_W2T_Opto_left_20250923_152241.mat', 
                        'D:/bpod/MLA-026807_(4)/Expert/W2T_Opto_left_blocked/Session Data/MLA-026807_(4)_W2T_Opto_left_blocked_20250923_150023.mat']
        },
        'M1': {
            'opto_0.5s': ['D:/bpod/MLA-026807_(4)/Expert/W2T_Opto_left/Session Data/MLA-026807_(4)_W2T_Opto_left_20250916_172933.mat', 
                          'D:/bpod/MLA-026807_(4)/Expert/W2T_Opto_left_blocked/Session Data/MLA-026807_(4)_W2T_Opto_left_blocked_20250916_170507.mat'],
            'opto_2s': ['D:/bpod/MLA-026807_(4)/Expert/W2T_Opto_left/Session Data/MLA-026807_(4)_W2T_Opto_left_20250922_191941.mat', 
                        'D:/bpod/MLA-026807_(4)/Expert/W2T_Opto_left_blocked/Session Data/MLA-026807_(4)_W2T_Opto_left_blocked_20250922_185914.mat']
        },
        'M2': {
            'opto_0.5s': ['D:/bpod/MLA-026807_(4)/Expert/W2T_Opto_left/Session Data/MLA-026807_(4)_W2T_Opto_left_20250917_161142.mat', 
                          'D:/bpod/MLA-026807_(4)/Expert/W2T_Opto_left_blocked/Session Data/MLA-026807_(4)_W2T_Opto_left_blocked_20250917_155023.mat'],
            'opto_2s': ['D:/bpod/MLA-026807_(4)/Expert/W2T_Opto_left/Session Data/MLA-026807_(4)_W2T_Opto_left_20250924_161436.mat', 
                        'D:/bpod/MLA-026807_(4)/Expert/W2T_Opto_left_blocked/Session Data/MLA-026807_(4)_W2T_Opto_left_blocked_20250924_155127.mat']
        },
        'mPFC': {
            'opto_0.5s': ['D:/bpod/MLA-026807_(4)/Expert/W2T_Opto_left/Session Data/MLA-026807_(4)_W2T_Opto_left_20250919_122818.mat', 
                          'D:/bpod/MLA-026807_(4)/Expert/W2T_Opto_left_blocked/Session Data/MLA-026807_(4)_W2T_Opto_left_blocked_20250919_120444.mat'],
            'opto_2s': ['D:/bpod/MLA-026807_(4)/Expert/W2T_Opto_left/Session Data/MLA-026807_(4)_W2T_Opto_left_20250925_171037.mat']
        }
    },
    'SNA-145894': {
        'S1': {
            'opto_0.5s': ['D:/bpod/SNA-145894_(1)/Expert/W2T_Opto_left/Session Data/SNA-145894_(1)_W2T_Opto_left_20250917_180415.mat', 
                          'D:/bpod/SNA-145894_(1)/Expert/W2T_Opto_left_blocked/Session Data/SNA-145894_(1)_W2T_Opto_left_blocked_20250917_174027.mat'],
            'opto_2s': ['D:/bpod/SNA-145894_(1)/Expert/W2T_Opto_left/Session Data/SNA-145894_(1)_W2T_Opto_left_20250923_164205.mat', 
                        'D:/bpod/SNA-145894_(1)/Expert/W2T_Opto_left_blocked/Session Data/SNA-145894_(1)_W2T_Opto_left_blocked_20250923_160816.mat', 
                        'D:/bpod/SNA-145894_(1)/Expert/W2T_Opto_left_blocked/Session Data/SNA-145894_(1)_W2T_Opto_left_blocked_20250923_161449.mat', 
                        'D:/bpod/SNA-145894_(1)/Expert/W2T_Opto_left_blocked/Session Data/SNA-145894_(1)_W2T_Opto_left_blocked_20250923_162024.mat']
        },
        'M1': {
            'opto_0.5s': ['D:/bpod/SNA-145894_(1)/Expert/W2T_Opto_left/Session Data/SNA-145894_(1)_W2T_Opto_left_20250916_191935.mat', 
                          #'D:/bpod/SNA-145894_(1)/Expert/W2T_Opto_left_blocked/Session Data/SNA-145894_(1)_W2T_Opto_left_blocked_20250916_185353.mat' 
                          ],
            'opto_2s': ['D:/bpod/SNA-145894_(1)/Expert/W2T_Opto_left/Session Data/SNA-145894_(1)_W2T_Opto_left_20250922_172307.mat', 
                        'D:/bpod/SNA-145894_(1)/Expert/W2T_Opto_left_blocked/Session Data/SNA-145894_(1)_W2T_Opto_left_blocked_20250922_165834.mat']
        },
        'M2': {
            'opto_0.5s': ['D:/bpod/SNA-145894_(1)/Expert/W2T_Opto_left/Session Data/SNA-145894_(1)_W2T_Opto_left_20250918_175954.mat', 
                          'D:/bpod/SNA-145894_(1)/Expert/W2T_Opto_left_blocked/Session Data/SNA-145894_(1)_W2T_Opto_left_blocked_20250918_173555.mat'],
            'opto_2s': ['D:/bpod/SNA-145894_(1)/Expert/W2T_Opto_left/Session Data/SNA-145894_(1)_W2T_Opto_left_20250924_171953.mat', 
                        'D:/bpod/SNA-145894_(1)/Expert/W2T_Opto_left_blocked/Session Data/SNA-145894_(1)_W2T_Opto_left_blocked_20250924_165518.mat']
        },
        'mPFC': {
            'opto_0.5s': ['D:/bpod/SNA-145894_(1)/Expert/W2T_Opto_left/Session Data/SNA-145894_(1)_W2T_Opto_left_20250919_151616.mat', 
                          'D:/bpod/SNA-145894_(1)/Expert/W2T_Opto_left_blocked/Session Data/SNA-145894_(1)_W2T_Opto_left_blocked_20250919_145147.mat'],
            'opto_2s': ['D:/bpod/SNA-145894_(1)/Expert/W2T_Opto_left/Session Data/SNA-145894_(1)_W2T_Opto_left_20250925_182049.mat', 
                        'D:/bpod/SNA-145894_(1)/Expert/W2T_Opto_left/Session Data/SNA-145894_(1)_W2T_Opto_left_20250925_185943.mat']
        }
    }
}

# Allow runner to inject a different mouse_sessions dict via env var
_ms_override = os.environ.get('BPOD_RASTER_SESSIONS_JSON')
if _ms_override:
    mouse_sessions = json.loads(_ms_override)

# ---------------------------------------------------------------------
# 4) Training-Sessions finden
# ---------------------------------------------------------------------

def find_training_sessions(mouse_name):
    tasks = mice[mouse_name]['Training']
    _mbase = mice[mouse_name].get('base_path', BASE_PATH)
    all_files = []
    for task in tasks:
        data_path = build_session_path(_mbase, mouse_name, 'Training', task)
        if not os.path.exists(data_path):
            continue
        files = [os.path.join(data_path, f)
                 for f in os.listdir(data_path)
                 if f.endswith('.mat')]
        files.sort(key=lambda f: extract_sort_key(os.path.basename(f)))
        all_files.extend(files)
    return all_files

# ---------------------------------------------------------------------
# 5) Training: Latenzen + Farben
# ---------------------------------------------------------------------

def get_trials_training(mat_path, max_latency=2.0):
    data = load_mat(mat_path)
    session = data['SessionData']
    trials = session['RawEvents']['Trial']
    if isinstance(trials, dict):
        trials = [trials]

    latencies = []
    colors = []

    for trial in trials:
        hit_flag = is_hit(trial)
        lat = latency(trial, max_latency=max_latency)

        if hit_flag:
            if np.isnan(lat):
                continue
            x_val = lat
            color = TRIAL_COLORS['w2t_hit']
        else:
            x_val = max_latency
            color = TRIAL_COLORS['w2t_miss']

        latencies.append(x_val)
        colors.append(color)

    return np.array(latencies), colors

# ---------------------------------------------------------------------
# 6) Training-Raster pro Maus
# ---------------------------------------------------------------------

def plot_raster_for_mouse(mouse_name, max_latency=2.0, save=False):
    session_files = find_training_sessions(mouse_name)
    day_map = day_index_within_mouse(session_files)

    for sess_path in session_files:
        latencies, colors = get_trials_training(sess_path, max_latency=max_latency)
        if len(latencies) == 0:
            continue

        fig, (ax_raster, ax_psth) = plt.subplots(2, 1, figsize=(4, 7),
                                                sharex=True,
                                                gridspec_kw={'height_ratios': [3, 1]})
        fig.patch.set_facecolor('white')

        # Rasterplot
        rt_list = [[v] for v in latencies]
        offsets = np.arange(len(latencies))
        ax_raster.eventplot(rt_list, lineoffsets=offsets,
                            linewidths=4.0, colors=colors)
        ax_raster.set_xlim([-0.2, max_latency + 0.05])
        ax_raster.set_ylim([len(latencies) + 5, -5])
        ax_raster.set_ylabel('Trial #')
        ax_raster.set_xticks(np.arange(0, max_latency + 0.01, 0.5))

        day_str, day_idx = day_map.get(sess_path, ('', ''))
        pretty_date = f"{day_str[6:8]}.{day_str[4:6]}.{day_str[0:4]}" if day_str else ""
        fig.suptitle(f"{mouse_name} - Novice", y=0.98)
        ax_raster.set_title(f"Day {day_idx} ({pretty_date})", fontsize=8)

        hit_patch  = mpatches.Patch(color=TRIAL_COLORS['w2t_hit'],  label='Hit')
        miss_patch = mpatches.Patch(color=TRIAL_COLORS['w2t_miss'], label='Miss')
        ax_raster.legend(handles=[hit_patch, miss_patch],
                         loc='upper right', frameon=False, fontsize=7)

        # Lower panel: KDE of hit latencies
        HIT_COLOR = TRIAL_COLORS['w2t_hit']
        hit_latencies = np.array([lat for lat, c in zip(latencies, colors)
                                  if c == HIT_COLOR])
        x_psth = np.linspace(0.01, max_latency, 200)
        if len(hit_latencies) > 1:
            kde = gaussian_kde(hit_latencies, bw_method='scott')
            ax_psth.fill_between(x_psth, kde(x_psth), alpha=0.2, color=HIT_COLOR)
            ax_psth.plot(x_psth, kde(x_psth), color=HIT_COLOR, linewidth=2,
                         label=f'KDE (n={len(hit_latencies)})')
            # quantile lines
            for q, ls in [(0.25, ':'), (0.50, '--'), (0.75, ':')]:
                qv = np.quantile(hit_latencies, q)
                ax_psth.axvline(qv, color=HIT_COLOR, linestyle=ls,
                                linewidth=1.2, alpha=0.8)
            ax_psth.axvline(np.median(hit_latencies), color=HIT_COLOR,
                            linestyle='--', linewidth=2.0, label='median')
            # ex-Gaussian fit (needs ≥10 hits)
            lats_fit = hit_latencies[(hit_latencies > 0) & (hit_latencies < max_latency)]
            if len(lats_fit) >= 10:
                try:
                    K, mu, sigma = exponnorm.fit(lats_fit, floc=None)
                    pdf_exg = exponnorm.pdf(x_psth, K, loc=mu, scale=sigma)
                    tau = K * sigma
                    ax_psth.plot(x_psth, pdf_exg, color=HIT_COLOR, linewidth=1.8,
                                 linestyle='--', label=f'ex-G mu={mu:.2f} tau={tau:.2f}')
                except Exception:
                    pass
        ax_psth.set_xlim([0, max_latency])
        ax_psth.set_xlabel('Latency (s)')
        ax_psth.set_ylabel('Density')
        ax_psth.legend(fontsize=7, frameon=False)
        ax_psth.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        if save:
            sess_base = os.path.basename(sess_path).replace('.mat', '')
            outname = str(_OUTDIR / f"{mouse_name}_Training_{sess_base}_raster_psth.png")
            plt.savefig(outname, dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

# ---------------------------------------------------------------------
# 7) Expert: Klassifikation (Hit/Miss/OptoHit/OptoMiss)
# ---------------------------------------------------------------------

def classify_trial_expert(tt, trial, max_latency=2.0):
    """
    tt: TrialType (1=W2T, 2=Opto)
    W2T-Hit     -> grün
    W2T-Miss    -> rot
    Opto-Hit    -> yellow
    Opto-Miss   -> hellblau
    """
    hit_flag = is_hit(trial)
    lat = latency(trial, max_latency=max_latency)

    if hit_flag:
        if np.isnan(lat):
            return None, None
        x_val = lat
        color = TRIAL_COLORS['w2t_hit'] if tt == 1 else TRIAL_COLORS['opto_hit']
    else:
        x_val = max_latency
        color = TRIAL_COLORS['w2t_miss'] if tt == 1 else TRIAL_COLORS['opto_miss']

    return x_val, color

# ---------------------------------------------------------------------
# 8) Expert-Raster aus mouse_sessions
# ---------------------------------------------------------------------

def plot_expert_raster_from_mouse_sessions(mouse_sessions, max_latency=2.0, save=False):

    for mouse, areas in mouse_sessions.items():
        for area, conds in areas.items():
            for cond_key, paths in conds.items():
                paths = [p for p in paths if os.path.exists(p)]
                if not paths:
                    continue
                n_sessions = len(paths)
                fig, axs = plt.subplots(
                    2, n_sessions,
                    figsize=(4 * n_sessions, 7),
                    sharex='col', sharey='row',
                    gridspec_kw={'height_ratios': [3, 1]}
                )
                if n_sessions == 1:
                    axs = axs[:, np.newaxis]

                nice_cond = '0.5s OptoStim' if cond_key == 'opto_0.5s' else '2s OptoStim'
                fig.patch.set_facecolor('white')
                fig.suptitle(f"{mouse} | {nice_cond} | {area}", y=0.98)
                day_map = build_day_map_for_paths(paths)

                # Legend: W2T green/red, Opto blue/orange
                w2t_hit_p   = mpatches.Patch(color=TRIAL_COLORS['w2t_hit'],  label='W2T Hit')
                w2t_miss_p  = mpatches.Patch(color=TRIAL_COLORS['w2t_miss'], label='W2T Miss')
                opto_hit_p  = mpatches.Patch(color=TRIAL_COLORS['opto_hit'], label='Opto Hit')
                opto_miss_p = mpatches.Patch(color=TRIAL_COLORS['opto_miss'],label='Opto Miss')
                legend_handles = [w2t_hit_p, w2t_miss_p, opto_hit_p, opto_miss_p]

                for idx, sess_path in enumerate(paths):
                    ax_raster = axs[0, idx]
                    ax_psth   = axs[1, idx]

                    data = load_mat(sess_path)
                    session = data['SessionData']
                    trial_types = np.atleast_1d(session['TrialTypes'])
                    trials = session['RawEvents']['Trial']
                    if isinstance(trials, dict):
                        trials = [trials]

                    latencies = []
                    colors    = []
                    kinds     = []   # 'w2t_hit', 'w2t_miss', 'opto_hit', 'opto_miss'

                    for i, trial in enumerate(trials):
                        if i >= len(trial_types):
                            break
                        tt = trial_types[i]          # 1 = W2T, 2 = Opto
                        x_val, _ = classify_trial_expert(tt, trial, max_latency=max_latency)
                        if x_val is None:
                            continue

                        if tt == 1:  # W2T
                            if is_hit(trial):
                                color_plot = TRIAL_COLORS['w2t_hit']
                                kinds.append('w2t_hit')
                            else:
                                color_plot = TRIAL_COLORS['w2t_miss']
                                kinds.append('w2t_miss')
                        else:        # Opto
                            if is_hit(trial):
                                color_plot = TRIAL_COLORS['opto_hit']
                                kinds.append('opto_hit')
                            else:
                                color_plot = TRIAL_COLORS['opto_miss']
                                kinds.append('opto_miss')

                        latencies.append(x_val)
                        colors.append(color_plot)

                    if len(latencies) == 0:
                        continue

                    # Raster: nur Balken (wie Training)
                    rt_list = [[v] for v in latencies]
                    offsets = np.arange(len(latencies))
                    ax_raster.eventplot(rt_list, lineoffsets=offsets,
                                        linewidths=4.0, colors=colors)

                    ax_raster.set_xlim([-0.2, max_latency + 0.05])
                    ax_raster.set_ylim([len(latencies) + 5, -5])
                    ax_raster.set_ylabel('Trial #')
                    ax_raster.set_xticks(np.arange(0, max_latency + 0.01, 0.5))

                    day_str, day_idx = day_map.get(sess_path, ('', ''))
                    pretty_date = f"{day_str[6:8]}.{day_str[4:6]}.{day_str[0:4]}" if day_str else ""
                    ax_raster.set_title(f"Day {day_idx} ({pretty_date})", fontsize=8)

                    # PSTH: KDE + quantile lines + ex-Gaussian for W2T-Hit and Opto-Hit
                    w2t_hit_lat  = np.array([lat for lat, kind in zip(latencies, kinds)
                                             if kind == 'w2t_hit'])
                    opto_hit_lat = np.array([lat for lat, kind in zip(latencies, kinds)
                                             if kind == 'opto_hit'])
                    x_psth = np.linspace(0.01, max_latency, 200)

                    for hit_lats, col, lbl in [
                        (w2t_hit_lat,  TRIAL_COLORS['w2t_hit'],  'W2T Hit'),
                        (opto_hit_lat, TRIAL_COLORS['opto_hit'], 'Opto Hit'),
                    ]:
                        if len(hit_lats) < 2:
                            continue
                        kde_h = gaussian_kde(hit_lats, bw_method='scott')
                        ax_psth.fill_between(x_psth, kde_h(x_psth), alpha=0.2, color=col)
                        ax_psth.plot(x_psth, kde_h(x_psth), color=col,
                                     linewidth=1.8, label=f'{lbl} KDE')
                        # quantile lines (Q25, Q50, Q75)
                        for q, ls in [(0.25, ':'), (0.50, '--'), (0.75, ':')]:
                            ax_psth.axvline(np.quantile(hit_lats, q), color=col,
                                            linestyle=ls, linewidth=1.0, alpha=0.7)
                        # ex-Gaussian fit
                        lats_fit = hit_lats[(hit_lats > 0) & (hit_lats < max_latency)]
                        if len(lats_fit) >= 10:
                            try:
                                K, mu, sigma = exponnorm.fit(lats_fit, floc=None)
                                tau = K * sigma
                                ax_psth.plot(x_psth,
                                             exponnorm.pdf(x_psth, K, loc=mu, scale=sigma),
                                             color=col, linestyle='--', linewidth=1.5,
                                             label=f'ex-G µ={mu:.2f} τ={tau:.2f}')
                            except Exception:
                                pass

                    ax_psth.set_xlim([0, max_latency])
                    ax_psth.set_xlabel('Latency (s)')
                    if idx == 0:
                        ax_psth.set_ylabel('Density')
                        ax_psth.legend(fontsize=6, frameon=False)
                        ax_psth.grid(True, axis='y', alpha=0.3)

                axs[0, 0].legend(handles=legend_handles,
                                 loc='upper right', ncol=2,
                                 frameon=False, fontsize=7)

                plt.tight_layout()
                if save:
                    outname = str(_OUTDIR / f"{mouse}_{area}_{cond_key}_expert_raster_psth.png")
                    plt.savefig(outname, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                else:
                    plt.show()

# ---------------------------------------------------------------------
# 9) Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # Training-Raster
    for mouse_name in sorted_mice:
        plot_raster_for_mouse(mouse_name, max_latency=2.0, save=True)

    # Expert-Raster
    plot_expert_raster_from_mouse_sessions(mouse_sessions, max_latency=2.0, save=True)
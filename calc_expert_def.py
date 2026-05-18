# -*- coding: utf-8 -*-
"""
Created on Fri Jan  9 12:02:46 2026

@author: Lea

Normierungsparameter für Learn Index berechnen.
Läuft über alle Training-Sessions aller Mäuse.
"""

import os
import numpy as np
import re
from scipy import stats

from bpod_utils import load_mat, latency, BASE_PATH, MIN_TRIALS

mouselist = ['SNA-145894_(1)', 'MLA-026805_(2)', 'MLA-026806_(3)', 'MLA-026807_(4)']
mice_tasks = {
    'SNA-145894_(1)': ['W2T_left'],
    'MLA-026805_(2)': ['W2T_right'],
    'MLA-026806_(3)': ['W2T_right'],
    'MLA-026807_(4)': ['W2T_left'],
}


def get_SEM(values, binsize=10):
    values = np.asarray(values)
    n_trials = len(values)
    sem_values = []
    center_indices = []
    for i in range(n_trials - binsize + 1):
        window = values[i:i + binsize]
        valid = window[~np.isnan(window)]
        if len(valid) > 1:
            sem = np.std(valid, ddof=1) / np.sqrt(len(valid))
        else:
            sem = np.nan
        sem_values.append(sem)
        center_indices.append(i + binsize // 2)
    sem_full = np.full(n_trials, np.nan)
    for idx, sem in zip(center_indices, sem_values):
        sem_full[idx] = sem
    return sem_full


# Collect all trial-wise latencies from Training
all_latencies = []

for mouse in mouselist:
    for task in mice_tasks[mouse]:
        data_path = os.path.join(BASE_PATH, mouse, 'Training', task, 'Session Data')
        if not os.path.exists(data_path):
            continue
        session_files = sorted(
            [f for f in os.listdir(data_path) if f.endswith('.mat')],
            key=lambda f: re.search(r'_(\d{8})_(\d{6})', f).group()
                          if re.search(r'_(\d{8})_(\d{6})', f) else f
        )

        for session_file in session_files:
            full_path = os.path.join(data_path, session_file)
            data = load_mat(full_path)
            try:
                ntrials = data['SessionData']['nTrials']
            except Exception:
                continue
            if ntrials < MIN_TRIALS:
                continue

            trials = data['SessionData']['RawEvents']['Trial']
            if isinstance(trials, dict):
                trials = [trials]

            session_latencies = [lat for trial in trials
                                 for lat in [latency(trial)] if not np.isnan(lat)]
            if session_latencies:
                all_latencies.extend(session_latencies)
                print(f"{mouse} | {session_file}: {len(session_latencies)} gültige Latencies hinzugefügt")

all_latencies = np.array(all_latencies)

print(f"\n=== NORMIERUNGSPARAMETER AUS {len(all_latencies)} TRIALS ===")
print(f"RT_naive  (95. Perzentil): {np.percentile(all_latencies, 95):.3f} s")
print(f"RT_expert (10. Perzentil): {np.percentile(all_latencies, 10):.3f} s")
print(f"RT_mean: {np.mean(all_latencies):.3f} s ± {np.std(all_latencies):.3f} s")

# SEM parameter: simulate SEMs across all latencies (single pooled distribution)
print("\n=== RT-SEM PARAMETER ===")
n_sims = 1000
all_sems = np.array([
    np.std(np.random.choice(all_latencies, size=50, replace=True)) /
    np.sqrt(50)
    for _ in range(n_sims)
])

print(f"SEM_naive  (90. Perzentil): {np.percentile(all_sems, 90):.3f} s")
print(f"SEM_expert (10. Perzentil): {np.percentile(all_sems, 10):.3f} s")

print("\n=== FÜR LEARN INDEX EINSETZEN ===")
print("RT_NAIVE =",  np.percentile(all_latencies, 95))
print("RT_EXPERT =", np.percentile(all_latencies, 10))
print("SEM_NAIVE =", np.percentile(all_sems, 90))
print("SEM_EXPERT =", np.percentile(all_sems, 10))

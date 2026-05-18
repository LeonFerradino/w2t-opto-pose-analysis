# Optogenetic and Video-Based Dissection of Cortical Contributions to a Learned Active Touch Task in Mice

Analysis code for the B.Sc. thesis *"Optogenetic and Video-Based Dissection of
Motor, Sensory, and Prefrontal Cortical Contributions to Learned Active Touch
Task in Mice."*

This repository contains the scripts used to produce the behavioral and
video-based pose figures of the thesis. The thesis PDF is included in the
repository.

> **Note on neural validation (Section 4.2):** The neural-validation figures
> are based on analysis scripts written by **Jelte De Vries**. That code is
> **not included in this repository**, as it is his work; only the behavioral
> and pose analysis code is provided here.

---

## Overview

Mice learn a whisker-based detection task, and the necessity of individual
cortical areas for **learning** versus **expert-level execution** is probed with
optogenetic inactivation. This repository covers two of the three analysis
readouts of the thesis:

1. **Behavior** — operant performance (hit rate, response latency, learning
   trajectory) from Bpod, and how optogenetic inhibition of each cortical area
   changes performance.
2. **Pose / kinematics** — markerless video tracking (DLC / SLEAP) of body and
   whisker movement to characterize learned movement stereotypy and how
   optogenetic inhibition reshapes task-related kinematics.

The third readout, neural validation, confirmed effective optogenetic silencing
via PSTHs, rasters, and population responses — see the note above on code
authorship.

---

## The behavioral task

The task is a head-fixed, whisker-based **go-detection task** ("whisker-to-target",
W2T):

- Each trial begins with a **2 s audio cue**, during which the mouse can strike a
  sensor with its whisker to obtain a water reward.
- The cue is followed by a **6 s inter-trial interval**, then the next trial.
- ~250 trials per session.
- Every trial is a stimulus (go) trial — there are no catch / stimulus-absent
  trials, so every response is a **hit** or a **miss**.

Optogenetic conditions compare baseline trials (W2T) against trials with
cortical inhibition delivered at different timings (e.g. 0.5 s and 2 s windows),
across four targeted cortical areas (M1, M2, S1, mPFC).

**Animals:** 3 VGAT-ChR2 mice (ChR2-expressing, prefixed `MLA-`) and 1 wildtype,
light-only control mouse (`SNA-145894`). The wildtype control is analyzed
separately and is never pooled with the ChR2 group, as it serves to flag
light-only artifacts.

---

## Quick start

```bash
conda env create -f environment.yml
```

This creates the analysis environment with all required packages. Activate it,
then run any of the figure scripts below.

---

## Figure map

Scripts mapped directly to the thesis figures and methods they produce.

### Behavioral analysis — Section 4.1

| Script | Produces |
|--------|----------|
| `hitrate_mean+SEM.py` | Novice success rate over training |
| `latency_mean+SEM.py` | Novice response latency over training |
| `learn_index_mean+SEM.py` | Composite LearnScore with sigmoid fit |
| `calc_expert_def.py` | LearnScore reference constants (naive / expert RT and SEM) |
| `raster_plots.py` | Single-mouse latency rasters and ex-Gaussian fits |
| `opto_hitrate.py` | Optogenetic effects on hit rate (per-mouse means, delta plots) |
| `opto_latency.py` | Optogenetic effects on response latency |
| `opto_habituation.py` | Random vs. blocked optogenetic session comparison |
| `phaseB_figures.py` | Re-rendered, annotated versions of the above (per-mouse stats, layout fixes) |

### Pose analysis — Section 4.3

| Script | Produces |
|--------|----------|
| `pose_sync_qc.py` | Bpod–camera frame synchronization QC |
| `pose_example_traces.py` | Example single-trial pose traces |
| `pose_trial_aligned.py` | HIT vs. Miss trial-aligned kinematics |
| `pose_learning_kinematics.py` | Learning-related kinematics (stereotypy, PCA, scatter) |
| `pose_figure2_rerender.py` | Re-rendered example/learning panels with the angle-wrap fix |
| `pose_opto_summary_v3.py` | Optogenetic kinematics heatmap (final version, angle-wrap fix applied) |

### Shared modules

| Module | Role |
|--------|------|
| `bpod_utils.py` | Bpod data loading, hit classification, latency, bootstrap CI, BH-FDR, rank-biserial *r* |
| `session_config.py` | Mouse session paths, mouse ordering |
| `plot_config.py` | Shared colors, markers, axis styling |
| `pose_config.py` | Keypoint specifications, mouse colors, confidence thresholds |
| `pose_utils.py` | Pose data loading, trial slicing, baseline subtraction, keypoint extraction |

---

## Methods summary

- **Circular baseline subtraction** (angle-wrap fix for whisker-angle keypoints):
  `dev = ((trace - bl_mean + 180) % 360) - 180`
- **Bootstrap confidence intervals:** 5000 resamples.
- **Linear mixed-effects models:** `statsmodels` `mixedlm`, REML, random
  intercept per mouse.
- **Per-mouse trial-level tests:** Mann-Whitney U (baseline vs. optogenetic
  trials).
- **Effect size:** rank-biserial *r* (unpaired, baseline vs. optogenetic HIT
  trials).
- **Multiple comparisons:** Benjamini-Hochberg FDR.
- **Outlier detection:** Iglewicz-Hoaglin modified Z-score (threshold 3.5).
- **Pose tracking confidence thresholds:** DLC 0.8, SLEAP 0.6.

---

## Data availability

Raw behavioral and video data are **not included** in this repository. The
scripts expect locally available Bpod session files and tracked pose data, with
paths defined in the `*_config.py` modules. Pose tracking (DLC / SLEAP) is
upstream of this repository; the scripts here operate on already-tracked pose
data.

---

## Author

B.Sc. thesis project. Neural-validation analysis (Section 4.2) is based on code
by Jelte De Vries and is not part of this repository.

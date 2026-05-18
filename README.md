# w2t-opto-pose-analysis
B.Sc. thesis code — optogenetic and video-based (DLC/SLEAP) dissection of motor, sensory, and prefrontal cortical contributions to a learned whisker-based active touch task (w2t) in mice

Overview
Mice learn a whisker-based detection task, and the necessity of individual
cortical areas for learning versus expert-level execution is probed with
optogenetic inactivation. This repository covers two of the three analysis
readouts of the thesis:

Behavior — operant performance (hit rate, response latency, learning
trajectory) from Bpod, and how optogenetic inhibition of each cortical area
changes performance.
Pose / kinematics — markerless video tracking (DLC / SLEAP) of body and
whisker movement to characterize learned movement stereotypy and how
optogenetic inhibition reshapes task-related kinematics.

The third readout, neural validation, confirmed effective optogenetic silencing
via PSTHs, rasters, and population responses — see the note above on code
authorship.

The behavioral task
The task is a head-fixed, whisker-based go-detection task ("whisker-to-target",
W2T):

Each trial begins with a 2 s audio cue, during which the mouse can strike a
sensor with its whisker to obtain a water reward.
The cue is followed by a 6 s inter-trial interval, then the next trial.
~250 trials per session.
Every trial is a stimulus (go) trial — there are no catch / stimulus-absent
trials, so every response is a hit or a miss.

Optogenetic conditions compare baseline trials (W2T) against trials with
cortical inhibition delivered at different timings (e.g. 0.5 s and 2 s windows),
across four targeted cortical areas (M1, M2, S1, mPFC).
Animals: 3 VGAT-ChR2 mice (ChR2-expressing, prefixed MLA-) and 1 wildtype,
light-only control mouse (SNA-145894). The wildtype control is analyzed
separately and is never pooled with the ChR2 group, as it serves to flag
light-only artifacts.

Quick start
bashconda env create -f environment.yml
This creates the analysis environment with all required packages. Activate it,
then run any of the figure scripts below.

Figure map
Scripts mapped directly to the thesis figures and methods they produce.
Behavioral analysis — Section 4.1
ScriptProduceshitrate_mean+SEM.pyNovice success rate over traininglatency_mean+SEM.pyNovice response latency over traininglearn_index_mean+SEM.pyComposite LearnScore with sigmoid fitcalc_expert_def.pyLearnScore reference constants (naive / expert RT and SEM)raster_plots.pySingle-mouse latency rasters and ex-Gaussian fitsopto_hitrate.pyOptogenetic effects on hit rate (per-mouse means, delta plots)opto_latency.pyOptogenetic effects on response latencyopto_habituation.pyRandom vs. blocked optogenetic session comparisonphaseB_figures.pyRe-rendered, annotated versions of the above (per-mouse stats, layout fixes)
Pose analysis — Section 4.3
ScriptProducespose_sync_qc.pyBpod–camera frame synchronization QCpose_example_traces.pyExample single-trial pose tracespose_trial_aligned.pyHIT vs. Miss trial-aligned kinematicspose_learning_kinematics.pyLearning-related kinematics (stereotypy, PCA, scatter)pose_figure2_rerender.pyRe-rendered example/learning panels with the angle-wrap fixpose_opto_summary_v3.pyOptogenetic kinematics heatmap (final version, angle-wrap fix applied)
Shared modules
ModuleRolebpod_utils.pyBpod data loading, hit classification, latency, bootstrap CI, BH-FDR, rank-biserial rsession_config.pyMouse session paths, mouse orderingplot_config.pyShared colors, markers, axis stylingpose_config.pyKeypoint specifications, mouse colors, confidence thresholdspose_utils.pyPose data loading, trial slicing, baseline subtraction, keypoint extraction

Methods summary

Circular baseline subtraction (angle-wrap fix for whisker-angle keypoints):
dev = ((trace - bl_mean + 180) % 360) - 180
Bootstrap confidence intervals: 5000 resamples.
Linear mixed-effects models: statsmodels mixedlm, REML, random
intercept per mouse.
Per-mouse trial-level tests: Mann-Whitney U (baseline vs. optogenetic
trials).
Effect size: rank-biserial r (unpaired, baseline vs. optogenetic HIT
trials).
Multiple comparisons: Benjamini-Hochberg FDR.
Outlier detection: Iglewicz-Hoaglin modified Z-score (threshold 3.5).
Pose tracking confidence thresholds: DLC 0.8, SLEAP 0.6.


Data availability
Raw behavioral and video data are not included in this repository. The
scripts expect locally available Bpod session files and tracked pose data, with
paths defined in the *_config.py modules. Pose tracking (DLC / SLEAP) is
upstream of this repository; the scripts here operate on already-tracked pose
data.

Author
B.Sc. thesis project.

# -*- coding: utf-8 -*-
"""
plot_config.py
Central style configuration shared across all plotting scripts.
Import from here to keep colours, markers, and font sizes consistent.
"""

# ── Mouse identity colours and markers ───────────────────────────────────────
# Long-format keys  →  match sorted_mice / folder names (used in hitrate,
#                       latency, learn_index scripts)
MOUSE_COLORS = {
    'SNA-145894_(1)': '#d62728',   # red    – wild-type
    'MLA-026805_(2)': '#1f77b4',   # blue
    'MLA-026806_(3)': '#ff7f0e',   # orange
    'MLA-026807_(4)': '#2ca02c',   # green
}

# Short-format keys  →  match mouse_sessions dict keys (used in opto scripts)
MOUSE_COLORS_SHORT = {
    'SNA-145894': '#d62728',
    'MLA-026805': '#1f77b4',
    'MLA-026806': '#ff7f0e',
    'MLA-026807': '#2ca02c',
}

MOUSE_MARKERS = {
    'SNA-145894': 'D',
    'MLA-026805': 'o',
    'MLA-026806': 's',
    'MLA-026807': '^',
}

# ── Training-phase colours ────────────────────────────────────────────────────
PHASE_COLORS = {
    'Training': '#1f77b4',   # blue  – novice
    'Expert':   '#2ca02c',   # green – expert
}

# ── Raster / trial-type colours ───────────────────────────────────────────────
# All four are taken from matplotlib's standard categorical palette so they are
# maximally distinguishable on a white background.
TRIAL_COLORS = {
    'w2t_hit':  '#2ca02c',   # green  – W2T hit
    'w2t_miss': '#d62728',   # red    – W2T miss
    'opto_hit': '#1f77b4',   # blue   – optogenetic hit  (replaces yellow)
    'opto_miss':'#ff7f0e',   # orange – optogenetic miss (replaces light-blue)
}

# ── Condition colours for bar / line plots ────────────────────────────────────
COND_COLORS = {
    'W2T':      '#555555',   # dark grey – baseline (W2T control)
    'opto_0.5s':'#ff7f0e',   # orange    – short inhibition
    'opto_2s':  '#d62728',   # red       – long inhibition
}

# ── Habituation: opto duration × session type ─────────────────────────────────
# Dark = random sessions, light = blocked sessions (same hue = same duration)
OPTO_DUR_COLORS = {
    'opto_0.5s': {'random': '#ff7f0e', 'blocked': '#ffbb78'},   # orange family
    'opto_2s':   {'random': '#d62728', 'blocked': '#ff9896'},   # red family
}

# Within-session habituation: random vs blocked session type
SESSION_TYPE_COLORS = {
    'random':  '#ff7f0e',   # orange
    'blocked': '#9467bd',   # purple – clearly distinct from orange
}

# ── Global font sizes ─────────────────────────────────────────────────────────
FS_TITLE  = 10
FS_LABEL  =  9
FS_TICK   =  8
FS_LEGEND =  7


# ── Axes style helper ─────────────────────────────────────────────────────────
def style_ax(ax):
    """Remove top and right spines for cleaner, publication-ready plots."""
    ax.spines[['top', 'right']].set_visible(False)

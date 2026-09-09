# -*- coding: utf-8 -*-
"""
Onset goal-directedness ("directional vector") analysis for the Squircle project.

Operationalises the idea from the 06-08-2026 meeting: at trial onset, does the
animal move toward the REWARDED well rather than toward the field of non-rewarded
("unfilled") wells? The non-rewarded wells provide a within-trial, geometry-aware
chance baseline, so a goal-directedness score reads as "how much more goal-aligned
than a randomly chosen well was this animal's first move".

Recommended use: keep as its own module and call gd.*  ->  import squircle.goal_directedness as gd
Alternative: paste BLOCK 1 into squircle/tools.py (call st.compute_goal_directedness)
             and BLOCK 2 into squircle/visualization.py (call sv.*).

Efficiency convention elsewhere in the pipeline is real/ideal (>=1, lower better);
this module is separate and reports goal-directedness on its own scales (see docstrings).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from squircle.design import well_locations_cm, fs
import squircle.tools as st  # used only by BLOCK 2 (build/plot)


# ======================================================================
# BLOCK 1  ->  squircle/tools.py   (st.compute_goal_directedness)
# ======================================================================

def _heading_index(x, y, window, window_unit, fs=fs):
    """First trajectory index reaching `window` cm of path (or `window` s of time)."""
    if window_unit == 's':
        return int(min(round(window * fs), len(x) - 1))
    if window_unit == 'cm':
        seg = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
        cum = np.concatenate(([0.0], np.cumsum(seg)))
        return int(min(np.searchsorted(cum, window), len(x) - 1))
    raise ValueError("window_unit must be 'cm' or 's'")


def compute_goal_directedness(df, rewarded_well,
                              window=5.0, window_unit='cm', fs=fs,
                              include_center=False, min_displacement=1.0):
    """
    Quantify onset goal-directedness for a single trial.

    The initial movement vector (start -> point reached after `window` cm of path,
    or `window` s of time) is compared to the direction of the rewarded well and of
    every non-rewarded well. Non-rewarded wells act as a chance distribution.

    Parameters
    ----------
    df : DataFrame
        Trial trajectory in cm ('x','y'), i.e. load_trials() tuple index 15.
    rewarded_well : int
        1-based rewarded well number (tuple index 8).
    window : float
        Size of the onset segment. Default 5.
    window_unit : {'cm','s'}
        Path length in cm (recommended: decouples heading from locomotor speed,
        which is confounded with age) or time in seconds.
    include_center : bool
        Keep the centre well (#17) in the null set. Default False.
    min_displacement : float
        Minimum start->heading displacement (cm) needed to define a direction.
        Below this the animal barely moved -> returns None (censored, not dropped).

    Returns
    -------
    dict or None
        heading_error_deg : angle 0-180 between first move and straight-to-goal
                            (0 = perfect, 90 = chance).
        directedness      : cos(heading_error), in [-1, 1] (1 = perfect, 0 = chance).
        goal_rank_pct     : fraction of non-rewarded wells worse-aligned than the
                            goal (0.5 = chance, 1 = goal best-aligned of all wells).
        heading_z         : (mean null angle - goal angle) / sd null angle.
        approach_t0/_tw   : (mean distance to non-rewarded wells) - (distance to
                            goal) at onset / after the window. >0 = nearer the goal
                            than the average well.
        approach_delta    : approach_tw - approach_t0 (net differential approach).
        frames_used       : trajectory index used as the heading point.
    """
    if df is None or getattr(df, "empty", True) or len(df) < 2:
        return None

    x = df['x'].to_numpy(dtype=float)
    y = df['y'].to_numpy(dtype=float)
    if np.isnan(x[0]) or np.isnan(y[0]):
        return None

    start = np.array([x[0], y[0]])
    k = _heading_index(x, y, window, window_unit, fs)
    head = np.array([x[k], y[k]])
    h = head - start
    if np.linalg.norm(h) < min_displacement:
        return None  # no interpretable initial direction

    wells = np.asarray(well_locations_cm, dtype=float)      # (17, 2)
    g = rewarded_well - 1
    idx = np.arange(len(wells))
    null_mask = idx != g
    if not include_center:
        null_mask &= idx != (len(wells) - 1)                # drop centre well

    # angle between the heading vector and each start->well vector
    v = wells - start                                       # (17, 2)
    nv = np.linalg.norm(v, axis=1)
    valid = nv > 1e-6
    cos = np.clip((v @ h) / (nv * np.linalg.norm(h) + 1e-12), -1.0, 1.0)
    ang = np.degrees(np.arccos(cos))                        # (17,)

    goal_ang = ang[g]
    null_ang = ang[null_mask & valid]
    if null_ang.size == 0:
        return None

    goal_rank_pct = float(np.mean(null_ang > goal_ang))
    heading_z = float((np.mean(null_ang) - goal_ang) / (np.std(null_ang) + 1e-12))
    directedness = float(np.cos(np.radians(goal_ang)))

    # two-snapshot distance version (matches the T=0 / T=0+window sketch)
    d0 = nv
    dt = np.linalg.norm(wells - head, axis=1)
    approach_t0 = float(np.mean(d0[null_mask]) - d0[g])
    approach_tw = float(np.mean(dt[null_mask]) - dt[g])

    return {
        'heading_error_deg': float(goal_ang),
        'directedness': directedness,
        'goal_rank_pct': goal_rank_pct,
        'heading_z': heading_z,
        'approach_t0': approach_t0,
        'approach_tw': approach_tw,
        'approach_delta': approach_tw - approach_t0,
        'frames_used': int(k),
    }


# ======================================================================
# BLOCK 2  ->  squircle/visualization.py   (sv.build_* / sv.plot_*)
# ======================================================================

def build_goal_directedness_df(trials, drop_rotation=True, **kwargs):
    """
    Vectorise compute_goal_directedness over a list of load_trials() tuples.
    Extra kwargs (window, window_unit, include_center, min_displacement) pass through.
    Returns a tidy DataFrame ready for plotting or a GLMM.
    """
    rows = []
    for t in trials:
        if drop_rotation and t[7] != 0:      # skip rotated probes (goal moved)
            continue
        m = compute_goal_directedness(t[15], t[8], **kwargs)
        if m is None:
            continue
        rows.append({
            'Subject': t[0], 'Sex': t[2], 'SessionType': t[3], 'Env': t[4],
            'Age': t[6], 'TrialNumber': t[10], 'Baited': t[16], **m
        })
    return pd.DataFrame(rows)


def plot_goal_directedness(trials, metric='goal_rank_pct', x='TrialNumber',
                           split_session=True, window=5.0, window_unit='cm',
                           figsize=(12, 6), title=None):
    """
    Plot onset goal-directedness against trial number or age.

    metric : 'goal_rank_pct' | 'heading_error_deg' | 'directedness'
             | 'heading_z' | 'approach_delta'
    x      : 'TrialNumber' or 'Age'
    split_session : colour Training vs Probe separately (the key contrast).
    """
    df = build_goal_directedness_df(trials, window=window, window_unit=window_unit)
    if df.empty:
        print("No trials with a definable heading direction.")
        return None

    chance = {'goal_rank_pct': 0.5, 'heading_error_deg': 90.0,
              'directedness': 0.0, 'heading_z': 0.0,
              'approach_delta': 0.0}.get(metric)

    sns.set_style('white')
    fig, ax = plt.subplots(figsize=figsize)
    hue = 'SessionType' if split_session else None
    sns.lineplot(data=df, x=x, y=metric, hue=hue, marker='o',
                 errorbar=('ci', 95), palette='Set2', ax=ax)

    if chance is not None:
        ax.axhline(chance, ls='--', color='red', lw=1, label='Chance')

    ax.set_xlabel('Trial number' if x == 'TrialNumber' else 'Age (days)')
    ax.set_ylabel(metric.replace('_', ' '))
    ax.set_title(title or f"Onset goal-directedness ({metric}) vs {x.lower()}")
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    return fig

# -*- coding: utf-8 -*-
"""
Distance-to-target onset goal-directedness visualisations for the Squircle project.

Implements the two-snapshot idea from the meeting: compare the animal's distance to
the TARGET well at T=0 (start) with its distance once it has travelled `window` cm of
path into the arena. The reduction is a goal-directedness signal; the non-rewarded
("unfilled") wells provide the chance baseline.

Three functions:
    plot_onset_distance_arena  - single-trial illustration on the real square/circle
    plot_onset_distance_grid   - a grid of such illustrations (whole session at a glance)
    plot_onset_distance_paired - before/after summary, target line vs unfilled-well line

Use as its own module (import squircle.goal_directedness_arena as gda) or paste into
squircle/visualization.py (call sv.*). Trial tuples are the current 17-tuple from
load_trials():  env = t[4], rewarded_well = t[8], df = t[15].

NOTE ON UNITS
  `window` is ARC LENGTH along the trajectory (path travelled), in cm.
  Distance-to-target at each snapshot is STRAIGHT-LINE Euclidean to the target-well centre.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from squircle.design import (well_locations_cm, circle_radius_cm,
                             square_side_cm)

_SQUARE_PAD = 7  # matches square_pad used elsewhere in visualization.py


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _path_point_index(x, y, window):
    """First trajectory index at which cumulative path length reaches `window` cm."""
    seg = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    cum = np.concatenate(([0.0], np.cumsum(seg)))
    return int(min(np.searchsorted(cum, window), len(x) - 1))


def _onset_distances(df, rewarded_well, window):
    """
    Distances to the target well and (mean) to the unfilled wells at T=0 and at
    the `window`-cm point. Returns None if the trajectory is unusable.
    """
    if df is None or getattr(df, 'empty', True) or len(df) < 2:
        return None
    x = df['x'].to_numpy(dtype=float)
    y = df['y'].to_numpy(dtype=float)
    if np.isnan(x[0]) or np.isnan(y[0]):
        return None

    k = _path_point_index(x, y, window)
    start = np.array([x[0], y[0]])
    point = np.array([x[k], y[k]])

    wells = np.asarray(well_locations_cm, dtype=float)
    g = rewarded_well - 1
    others = np.arange(len(wells)) != g

    d0 = np.linalg.norm(wells - start, axis=1)
    dw = np.linalg.norm(wells - point, axis=1)
    return {
        'd0_goal': float(d0[g]),
        'dw_goal': float(dw[g]),
        'd0_others': float(d0[others].mean()),
        'dw_others': float(dw[others].mean()),
        'k': k,
    }


# ----------------------------------------------------------------------
# Tier 1 - single-trial arena illustration
# ----------------------------------------------------------------------

def plot_onset_distance_arena(trial, window=5.0, ax=None,
                              show_full_path=True, annotate=True, title=None):
    """
    Illustrate the two-snapshot distance-to-target measure for one trial on the
    real arena. Draws boundary, all wells, the target, the start (T=0), the point
    reached after `window` cm of path, the first-`window`-cm path segment, and the
    two straight-line distances to target.

    `trial` : one 17-tuple from load_trials().
    """
    
    name = trial[0]
    kind = trial[3]
    env = trial[4]
    age = trial[6]
    rewarded_well = trial[8]
    tnum = trial[10]
    df = trial[15]


    if df is None or getattr(df, 'empty', True) or len(df) < 2:
        print(f"No usable trajectory for {name} trial {tnum}.")
        return None

    x = df['x'].to_numpy(dtype=float)
    y = df['y'].to_numpy(dtype=float)
    k = _path_point_index(x, y, window)

    sx, sy = x[0], y[0]
    px, py = x[k], y[k]
    tx, ty = well_locations_cm[rewarded_well - 1]
    d0 = float(np.hypot(tx - sx, ty - sy))
    dw = float(np.hypot(tx - px, ty - py))

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(8, 8))
    sns.set_style('white')

    # boundary
    if env == 'Circle':
        ax.add_patch(plt.Circle((0, 0), circle_radius_cm,
                                color='black', fill=False, lw=1.5))
        lim = circle_radius_cm + 6
    elif env == 'Square':
        side = square_side_cm + _SQUARE_PAD
        ax.add_patch(plt.Rectangle((-side / 2, -side / 2), side, side,
                                   color='black', fill=False, lw=1.5))
        lim = side / 2 + 6
    else:
        raise ValueError('Environment type not correct!')

    # wells and target
    ax.scatter(*zip(*well_locations_cm), color='gray', s=50, zorder=2, label='Wells')
    ax.scatter([tx], [ty], color='red', s=180, zorder=3, label='Target well')

    # trajectory
    if show_full_path:
        ax.plot(x, y, color='lightsteelblue', lw=1, alpha=0.7, zorder=2)
    ax.plot(x[:k + 1], y[:k + 1], color='navy', lw=2.5, zorder=4,
            label=f'First {window:.0f} cm of path')

    # snapshot markers
    ax.scatter([sx], [sy], marker='*', s=260, color='lime',
               edgecolor='black', lw=1.5, zorder=5, label='Start (T=0)')
    ax.scatter([px], [py], marker='D', s=110, color='orange',
               edgecolor='black', lw=1.2, zorder=5, label=f'{window:.0f} cm point')

    # straight-line distances to target
    ax.plot([sx, tx], [sy, ty], ls='--', color='gray', lw=1.2, zorder=3)
    ax.plot([px, tx], [py, ty], ls='--', color='darkorange', lw=1.4, zorder=3)
    if annotate:
        ax.annotate(f'{d0:.1f} cm', ((sx + tx) / 2, (sy + ty) / 2),
                    color='gray', fontsize=9)
        ax.annotate(f'{dw:.1f} cm', ((px + tx) / 2, (py + ty) / 2),
                    color='darkorange', fontsize=9)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')
    ax.set_title(title or
                 f"{name}  P{age}  trial {tnum} ({kind}, {env})\n"
                 f"dist to target: {d0:.1f} \u2192 {dw:.1f} cm  (\u0394 = {d0 - dw:+.1f})")

    if created:
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9)
        plt.tight_layout()
        plt.show()
    return ax


def plot_onset_distance_grid(trials, window=5.0, ncols=3, max_trials=9, figsize=None):
    """Grid of single-trial arena illustrations (first `max_trials` usable trials)."""
    usable = [t for t in trials
              if t[15] is not None and not getattr(t[15], 'empty', True)][:max_trials]
    if not usable:
        print('No usable trials.')
        return None

    n = len(usable)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=figsize or (5 * ncols, 5 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for ax, t in zip(axes, usable):
        plot_onset_distance_arena(t, window=window, ax=ax, annotate=False)
    for ax in axes[n:]:
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    return fig


# ----------------------------------------------------------------------
# Tier 2 - paired before/after summary with unfilled-well chance reference
# ----------------------------------------------------------------------

def plot_onset_distance_paired(trials, window=5.0, split_session=True,
                               drop_rotation=True, figsize=(8, 6), title=None):
    """
    Mean distance to the TARGET well at T=0 vs after `window` cm, with the mean
    distance to the UNFILLED wells overlaid as the chance reference. The target line
    dropping more steeply than the unfilled-well line is the goal-directedness signal.

    split_session : plot Training and Probe as separate colours.
    """
    rows = []
    for t in trials:
        if drop_rotation and t[7] != 0:      # skip rotated probes (target moved)
            continue
        dd = _onset_distances(t[15], t[8], window)
        if dd is None:
            continue
        rows.append({'SessionType': t[3], **dd})

    D = pd.DataFrame(rows)
    if D.empty:
        print('No usable trials.')
        return None

    sns.set_style('white')
    fig, ax = plt.subplots(figsize=figsize)
    xpos = [0, 1]
    palette = {'Training': 'tab:blue', 'Probe': 'tab:green', 'All': 'tab:blue'}
    sessions = list(D['SessionType'].unique()) if split_session else ['All']

    for s in sessions:
        sub = D if s == 'All' else D[D['SessionType'] == s]
        c = palette.get(s, 'gray')
        goal = [sub['d0_goal'].mean(), sub['dw_goal'].mean()]
        gerr = [sub['d0_goal'].sem(), sub['dw_goal'].sem()]
        other = [sub['d0_others'].mean(), sub['dw_others'].mean()]
        ax.errorbar(xpos, goal, yerr=gerr, marker='o', color=c, lw=2.5,
                    capsize=4, label=f'{s} \u2014 target')
        ax.plot(xpos, other, marker='o', color=c, lw=1.2, ls=':',
                alpha=0.7, label=f'{s} \u2014 unfilled wells (chance)')

    ax.set_xticks(xpos)
    ax.set_xticklabels(['T=0', f'{window:.0f} cm in'])
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylabel('Distance to well (cm)')
    ax.set_title(title or
                 f'Onset approach: distance to target vs unfilled wells '
                 f'(first {window:.0f} cm of path)')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    plt.tight_layout()
    plt.show()
    return fig

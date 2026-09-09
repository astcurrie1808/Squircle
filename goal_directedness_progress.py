# -*- coding: utf-8 -*-
"""
Path-progress goal-directedness for the Squircle project.

Operationalises the "20% of the way in" idea: measure how far the animal IS from
the rewarded well at trial onset, measure it again once the animal has walked a
given fraction of its path, and subtract. A larger reduction = the early part of
the trajectory was aimed at the goal.

TERMINOLOGY (this trips people up, so it is spelled out)
    Both measurements are the straight-line Euclidean distance from WHERE THE
    ANIMAL IS to the centre of the target well. They are not "ideal distances" in
    the sense that term carries elsewhere in this pipeline.

    Elsewhere, "ideal distance" (_ideal_distance in visualization.py, and the
    denominator inside st.calculate_efficiency) means one specific quantity: the
    straight line from the START position to the reward. It is a fixed property of
    the trial, evaluated once at frame 0.

    Here, d0 happens to coincide with that (it is the same start-to-reward line),
    but d_at does not - it is the start-to-reward line's counterpart measured from
    the animal's position at the snapshot. So:

        d0   = |animal at frame 0        - target well|      (== the ideal distance)
        d_at = |animal at the 20% mark   - target well|      (NOT an ideal distance)
        dD   = d_at - d0                                     (negative = closer)

Relation to the other two modules
    goal_directedness.py        - ANGULAR onset measure (heading vs straight-to-goal)
                                  over an absolute window (default 5 cm of path).
    goal_directedness_arena.py  - DISTANCE two-snapshot measure, absolute window,
                                  single-trial arena illustrations.
    this module                 - DISTANCE measure at a RELATIVE point along the
                                  path, plus the full approach profile.

WHERE THE SECOND MEASUREMENT IS TAKEN  (parameters `frac` and `frac_of`)
    The snapshot is taken once the animal has walked `s` cm. `frac` sets the
    fraction; `frac_of` says a fraction OF WHAT:

        frac_of='total_path'      s = frac * L    L = total path walked in the trial
        frac_of='start_distance'  s = frac * d0   d0 = start-to-reward straight line

    frac=0.2, frac_of='total_path' (the defaults) == "20% of the total distance
    traversed", which is the paradigm as specified. Everything reported for
    litters 9-11 used these defaults.

    frac_of='start_distance' exists only as a sensitivity check, because the two
    can disagree. Under 'total_path', distance to goal cannot fall faster than the
    animal walks, so
            d0 - d_at <= frac * L = frac * efficiency * d0
    i.e. the ceiling on the RAW scores (g, delta_cm, dD) grows with inefficiency:
    a dead-straight trial scores exactly `frac`, a meandering one has room to score
    higher. So do not read raw dD as "more goal-directed" across trials of differing
    efficiency without checking. g_rank_pct is not affected in the same way (see
    CHANCE BASELINE) and is the score to report. Re-running an analysis under
    'start_distance' shows whether a result depends on that choice.

CHANCE BASELINE
    The same trajectory is re-scored against every non-rewarded well on the SAME
    RING as the target (wells 1-8 outer, 9-16 inner, 17 centre; see design.py).
    Ring-matching keeps the geometry comparable. This yields g_rank_pct
    (0.5 = chance) and g_z, which are safe to compare across Circle and Square.

    Under frac_of='total_path' all wells are scored at the SAME physical point (the
    frac*L position), so the inefficiency ceiling applies to every well at once and
    largely cancels in the ranking. That is what makes g_rank_pct the robust
    readout of this paradigm: it asks "having walked 20% of its path, was the
    animal disproportionately nearer the TARGET than the other wells it could
    equally have been heading for" - a question a long, wandering path does not
    answer in the affirmative by accident.

Both arenas are convex and all wells are interior, so the straight line from any
position to any well is always traversable - no obstacle correction is needed.

Usage:  import squircle.goal_directedness_progress as gdp
"""

import os
import json
from io import StringIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from scipy import stats as _sps

from squircle.design import well_locations_cm
import squircle.tools as st

# Ring membership, 0-based, matching design.well_locations
_OUTER = np.arange(0, 8)
_INNER = np.arange(8, 16)
_CENTRE = 16


# ======================================================================
# Loading  (so these functions can be called the same way as sv.plot_*)
# ======================================================================

def load_litters(working_folder, litters=(9, 10, 11), **filters):
    """
    Load every trial from litter_<n>_processed_cm.json for the given litters.

    Mirrors how visualization.py's plot_* functions take (working_folder, litters=...),
    including skipping missing litter files silently. Extra kwargs go to
    st.load_trials (name, kind, environment, date, number, rotation, baited); with
    no kind filter, Training AND Probe trials are returned together.
    """
    trials = []
    for n in litters:
        path = os.path.join(working_folder, f'litter_{n}_processed_cm.json')
        if not os.path.exists(path):
            continue
        selected, _ = st.load_trials(path, **filters)
        trials.extend(selected)
    return trials


def _as_trials(source, litters=(9, 10, 11)):
    """
    Accept either a working_folder path (load it) or an already-loaded trial list.
    Lets every plot below be called both ways:
        gdp.plot_progress_panels(working_folder, litters=(9, 10, 11))
        gdp.plot_progress_panels(trials)
    """
    if isinstance(source, (str, os.PathLike)):
        return load_litters(source, litters)
    return source


# ======================================================================
# Geometry helpers
# ======================================================================

def _clean_xy(df):
    """Finite (x, y) arrays from a trial DataFrame, NaN frames dropped."""
    if df is None or getattr(df, 'empty', True) or len(df) < 2:
        return None, None
    x = df['x'].to_numpy(dtype=float)
    y = df['y'].to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return None, None
    return x[ok], y[ok]


def _cumpath(x, y):
    """Cumulative arc length along the trajectory, same length as x."""
    return np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))))


def _point_at(x, y, cum, s):
    """
    Position after `s` cm of path, linearly interpolated along arc length
    (not snapped to the nearest frame - at 30 fps a pup covers several cm
    between samples). Clipped to the end of the trajectory.
    """
    s = float(np.clip(s, 0.0, cum[-1]))
    return np.interp(s, cum, x), np.interp(s, cum, y)


def _ring_of(g):
    """Indices of the wells sharing a ring with 0-based well index `g`, `g` excluded."""
    if g in _OUTER:
        ring = _OUTER
    elif g in _INNER:
        ring = _INNER
    else:
        return np.array([], dtype=int)      # centre well has no ring-mates
    return ring[ring != g]


# ======================================================================
# Core measure
# ======================================================================

def compute_progress_approach(df, rewarded_well, frac=0.2, frac_of='total_path',
                              min_d0=3.0, min_path=2.0):
    """
    How much of its starting distance to the target the animal has closed by the
    time it has walked a given fraction of its path.

    Both distances are straight-line animal-to-target-well distances (see the
    TERMINOLOGY note in the module docstring - only the frame-0 one is an "ideal
    distance" in this pipeline's sense of the term).

    Parameters
    ----------
    df : DataFrame
        Trial trajectory in cm ('x','y'); load_trials() tuple index 15.
    rewarded_well : int
        1-based rewarded well number (tuple index 8).
    frac : float
        How far along to take the snapshot, as a fraction. 0.2 = the "20%".
    frac_of : {'total_path','start_distance'}
        A fraction of WHAT. Default 'total_path' = frac * L, the paradigm as
        specified. 'start_distance' = frac * d0, a sensitivity check.
        See the module docstring.
    min_d0 : float
        Skip trials starting closer than this (cm) to the target - the ratio blows up.
    min_path : float
        Skip trials whose whole trajectory is shorter than this (cm).

    Returns
    -------
    dict or None
        d0             : distance from the animal's START position to the target, cm
        d_at           : distance from the animal's position AT THE SNAPSHOT to the
                         target, cm
        dD             : d_at - d0, cm. NEGATIVE = moved closer to the goal.
                         Matches the pipeline's "lower is better" convention
                         (st.calculate_efficiency reports real/ideal, lower better).
        delta_cm       : -dD, i.e. d0 - d_at. Positive = closed distance.
                         Same number, opposite sign; both kept so the sign is
                         explicit at every call site rather than assumed.
        g              : delta_cm / d0, the unitless version (start-distance corrected).
        g_rank_pct     : fraction of ring-matched non-target wells approached LESS
                         than the target (0.5 = chance, 1.0 = target best of all).
        g_z            : (g - mean null g) / sd null g.
        g_null_mean    : mean g over the ring-matched null wells.
        total_path     : whole-trial path length, cm.
        efficiency     : total_path / d0 (matches st.calculate_efficiency).
        s_used         : cm of path actually walked to reach the snapshot.
    """
    if frac_of not in ('total_path', 'start_distance'):
        raise ValueError("frac_of must be 'total_path' or 'start_distance'")

    x, y = _clean_xy(df)
    if x is None:
        return None

    cum = _cumpath(x, y)
    total_path = float(cum[-1])
    if total_path < min_path:
        return None

    wells = np.asarray(well_locations_cm, dtype=float)
    g_idx = rewarded_well - 1
    start = np.array([x[0], y[0]])

    d0_all = np.linalg.norm(wells - start, axis=1)
    d0 = float(d0_all[g_idx])
    if d0 < min_d0:
        return None

    def _closed(well_i):
        """
        Fraction of initial distance to `well_i` closed by its own snapshot.
        With frac_of='start_distance' each well gets its own snapshot at frac*d0_i, so the null
        wells are scored on equal terms rather than at the target's snapshot.
        Returns None when the start is too close to that well for a stable ratio.
        """
        di0 = float(d0_all[well_i])
        if di0 < min_d0:
            return None
        s = frac * di0 if frac_of == 'start_distance' else frac * total_path
        px, py = _point_at(x, y, cum, s)
        di = float(np.hypot(wells[well_i, 0] - px, wells[well_i, 1] - py))
        return (di0 - di) / di0, di, s, di - di0

    goal = _closed(g_idx)
    if goal is None:
        return None
    g_val, d_at, s_used, _ = goal

    ring = [r for r in map(_closed, _ring_of(g_idx)) if r is not None]
    null = np.array([r[0] for r in ring], dtype=float)
    null_dD = np.array([r[3] for r in ring], dtype=float)
    null = null[np.isfinite(null)]
    null_dD = null_dD[np.isfinite(null_dD)]

    if null.size:
        g_rank_pct = float(np.mean(null < g_val))
        g_null_mean = float(np.mean(null))
        g_z = float((g_val - g_null_mean) / (np.std(null) + 1e-12))
    else:
        g_rank_pct = g_null_mean = g_z = np.nan

    return {
        'd0': d0,
        'd_at': d_at,
        'dD': d_at - d0,          # negative = moved closer
        'delta_cm': d0 - d_at,    # positive = moved closer
        # Chance level for dD, in cm: the same movement scored against the wells
        # on the target's ring. NOT zero - arena geometry makes the expected
        # change non-zero, and it differs between Circle and Square. Compare dD
        # to THIS, not to the zero line.
        'dD_null': float(np.mean(null_dD)) if null_dD.size else np.nan,
        'g': g_val,
        'g_rank_pct': g_rank_pct,
        'g_z': g_z,
        'g_null_mean': g_null_mean,
        'total_path': total_path,
        'efficiency': total_path / d0,
        's_used': s_used,
    }


def approach_profile(df, rewarded_well, grid=None, normalise=True):
    """
    Distance to the target well as a function of path progress.

    Returns (grid, d) where `grid` is fraction of total path walked (0 -> 1) and
    `d` is the straight-line distance to the target at each point, divided by the
    onset distance when normalise=True (so it starts at 1 and ends near 0).

    This is the curve your single 20% number is one slice of.
    """
    if grid is None:
        grid = np.linspace(0.0, 1.0, 101)

    x, y = _clean_xy(df)
    if x is None:
        return grid, np.full_like(grid, np.nan, dtype=float)

    cum = _cumpath(x, y)
    if cum[-1] <= 0:
        return grid, np.full_like(grid, np.nan, dtype=float)

    tx, ty = well_locations_cm[rewarded_well - 1]
    px = np.interp(grid * cum[-1], cum, x)
    py = np.interp(grid * cum[-1], cum, y)
    d = np.hypot(tx - px, ty - py)
    return grid, (d / d[0] if (normalise and d[0] > 0) else d)


# ======================================================================
# Tidy frames
# ======================================================================

def _agg_kwargs(agg='mean', errorbar=None):
    """
    seaborn estimator/errorbar for a central-tendency choice.

    agg='mean'   -> mean with a bootstrapped 95% CI  (default)
    agg='median' -> median with the interquartile range, matching the
                    "pooled median + IQR" convention in visualization.py

    A median is the better summary for dD: the per-trial distribution has a long
    tail (single trials reach -30 cm), so a point built from ~7 trials can be
    dragged far by one of them. Note the IQR is a SPREAD, not a confidence
    interval - it will look wider than the CI and does not shrink with n.
    """
    if agg == 'median':
        return dict(estimator='median', errorbar=errorbar or ('pi', 50))
    if agg == 'mean':
        return dict(estimator='mean', errorbar=errorbar or ('ci', 95))
    raise ValueError("agg must be 'mean' or 'median'")


def _tick_every_integer(ax, values, max_ticks=32):
    """
    Label EVERY whole number spanned by `values` (1, 2, 3, ... 20) rather than
    letting matplotlib choose its own spacing, which lands on every 3rd tick.
    Falls back to the automatic integer locator if that would crowd the axis
    with more than `max_ticks` labels.
    """
    v = pd.to_numeric(pd.Series(list(values)), errors='coerce').dropna()
    if v.empty:
        return
    lo, hi = int(np.floor(v.min())), int(np.ceil(v.max()))
    if hi - lo + 1 <= max_ticks:
        ax.set_xticks(np.arange(lo, hi + 1))
    else:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def _whole_numbers(series):
    """
    Cast a column to int when every value is whole. Trial numbers arrive as
    float64 (the source CSV parses them that way), which makes axes render
    2.5 / 7.5 tick labels and writes '1.0' into exported CSVs. Values that are
    genuinely fractional are left alone rather than silently rounded.
    """
    s = pd.to_numeric(series, errors='coerce')
    if s.notna().all() and (s == s.round()).all():
        return s.astype(int)
    return series


def _completed(trial):
    """Mirror of visualization._is_completed, without the circular import."""
    if len(trial) < 18:
        return True
    return str(trial[17]).strip().lower() in ('completed', 'complete', '1', 'true')


def _session_key(t):
    """(subject, kind, env, date, rotation) - identifies one session."""
    return (t[0], t[3], t[4], t[5], t[7])


def _first_trial_baited_map(trials):
    """
    {session key: baiting (0/1) of that session's trial 1}.

    Matches the rule in visualization.plot_efficiency_across_trials: the session's
    trial NUMBER 1 decides, and a session with no trial 1 gets no entry (so it is
    excluded whenever the filter is active). Note trials with no tracking data are
    already dropped by load_trials, so a session whose trial 1 was untracked lands
    in that excluded group.
    """
    return {_session_key(t): int(t[16]) for t in trials if t[10] == 1}


def build_progress_df(trials, litters=(9, 10, 11), drop_rotation=True,
                      completed_only=False, first_trial_baited=None, **kwargs):
    """
    Vectorise compute_progress_approach over load_trials() tuples.
    Extra kwargs (frac, frac_of, min_d0, min_path) pass through.

    completed_only : keep only trials whose logged Completed status marks a success.
        In litters 9-11 this is 2993/2996 trials, so it changes nothing there - but
        set it True when you need "successful trials" to be literally true in the
        methods section.
    first_trial_baited : int or None
        Split by whether the SESSION's first trial was baited, as in
        sv.plot_efficiency_across_trials. 0 keeps only sessions whose trial 1 was
        UNbaited, 1 only those baited, None (default) keeps all. The whole
        session's trials are kept - only the session's inclusion depends on it.
    """
    trials = _as_trials(trials, litters)
    fb_map = _first_trial_baited_map(trials) if first_trial_baited is not None else None
    rows = []
    for t in trials:
        if drop_rotation and t[7] != 0:          # rotated probes: goal moved
            continue
        if completed_only and not _completed(t):
            continue
        if fb_map is not None and fb_map.get(_session_key(t)) != first_trial_baited:
            continue
        m = compute_progress_approach(t[15], t[8], **kwargs)
        if m is None:
            continue
        rows.append({
            'Subject': t[0], 'Sex': t[2], 'SessionType': t[3], 'Env': t[4],
            'Date': t[5], 'Age': t[6], 'TrialNumber': t[10], 'Baited': t[16],
            'Completed': _completed(t), **m
        })
    D = pd.DataFrame(rows)
    if not D.empty:
        D['TrialNumber'] = _whole_numbers(D['TrialNumber'])
    return D


def build_profile_df(trials, litters=(9, 10, 11), drop_rotation=True, n_points=101):
    """Long-form approach profiles, one row per (trial, progress point)."""
    trials = _as_trials(trials, litters)
    grid = np.linspace(0.0, 1.0, n_points)
    rows = []
    for i, t in enumerate(trials):
        if drop_rotation and t[7] != 0:
            continue
        _, d = approach_profile(t[15], t[8], grid=grid)
        if not np.isfinite(d).any():
            continue
        rows.append(pd.DataFrame({
            'TrialUID': i, 'Subject': t[0], 'SessionType': t[3], 'Env': t[4],
            'Age': t[6], 'TrialNumber': t[10], 'Baited': t[16],
            'progress': grid, 'rel_distance': d,
        }))
    if not rows:
        return pd.DataFrame()
    P = pd.concat(rows, ignore_index=True)
    P['TrialNumber'] = _whole_numbers(P['TrialNumber'])
    return P


# ======================================================================
# Plots
# ======================================================================

def plot_approach_profile(trials, litters=(9, 10, 11), hue='Env', frac=0.2,
                          drop_rotation=True, agg='mean', errorbar=None,
                          figsize=(9, 6), title=None):
    """
    Mean +/- 95% CI distance-to-target (normalised to onset) against percentage of
    path walked. Goal-directed trials drop steeply early; exploratory trials show a
    flat shoulder then a late plunge. Vertical line marks the `frac` snapshot.

    hue : 'Env' | 'SessionType' | 'Baited' | None
    """
    P = build_profile_df(trials, litters=litters, drop_rotation=drop_rotation)
    if P.empty:
        print('No usable trials.')
        return None

    sns.set_style('white')
    fig, ax = plt.subplots(figsize=figsize)
    sns.lineplot(data=P, x='progress', y='rel_distance', hue=hue,
                 palette='Set2', ax=ax, **_agg_kwargs(agg, errorbar))

    ax.axvline(frac, ls='--', color='red', lw=1)
    ax.annotate(f'{frac:.0%} of path', (frac, 1.0), xytext=(4, -4),
                textcoords='offset points', color='red', fontsize=9, va='top')
    # A dead-straight run would fall along this line.
    ax.plot([0, 1], [1, 0], ls=':', color='gray', lw=1, label='Perfectly direct')

    ax.set_xlabel('Path travelled (fraction of trial total)')
    ax.set_ylabel('Distance to target / onset distance')
    ax.set_title(title or 'Approach profile: distance to target vs path progress')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    plt.tight_layout()
    plt.show()
    return fig


def plot_progress_delta(trials, litters=(9, 10, 11), metric='g_rank_pct', x='TrialNumber',
                        hue='SessionType', frac=0.2, frac_of='total_path',
                        drop_rotation=True, completed_only=False,
                        first_trial_baited=None, agg='mean', errorbar=None,
                        figsize=(11, 6), title=None):
    """
    The summary panel: goal-directedness against trial number or age.

    NOTE ON x='TrialNumber' (litters 9-11): within-session trial number carries no
    learning signal in this dataset - path efficiency is flat across trials 1-20
    (Spearman rho = +0.009, p = 0.68), as are path length and start distance. The
    developmental effect lives ACROSS days: dD vs Age gives rho = -0.34, p = 4e-4.
    So x='Age' is the informative axis here; x='TrialNumber' is a control showing
    the within-session null.

    metric : 'g_rank_pct' - vs ring-matched wells, 0.5 = chance  <- report this one
             'dD'         - the subtraction in cm, NEGATIVE = moved closer
             'delta_cm'   - same, sign flipped: POSITIVE = moved closer
             'g'          - dD normalised by d0 (raw; ceiling grows with
                            inefficiency under frac_of='total_path')
             'g_z'        - z relative to the ring-matched null
    x      : 'TrialNumber' | 'Age'
    hue    : 'SessionType' | 'Env' | 'Baited' | None
    """
    D = build_progress_df(trials, litters=litters, drop_rotation=drop_rotation,
                          completed_only=completed_only,
                          first_trial_baited=first_trial_baited,
                          frac=frac, frac_of=frac_of)
    if D.empty:
        print('No usable trials.')
        return None
    D = add_age_band(D)          # so hue='Band' works here too, as in the panels

    # Chance is only 0 for the metrics that are already scored against the null
    # wells. For dD/delta_cm (raw cm) zero means "no change in distance", which is
    # NOT chance: arena geometry makes the expected change non-zero and it differs
    # between Circle and Square, so the empirical null is used instead.
    if metric in ('dD', 'delta_cm'):
        chance = float(D['dD_null'].mean())
        if metric == 'delta_cm':
            chance = -chance
        chance_label = 'Chance (ring-matched wells)'
    else:
        chance = {'g': 0.0, 'g_rank_pct': 0.5, 'g_z': 0.0}.get(metric)
        chance_label = 'Chance'
    ceiling = frac if (metric == 'g' and frac_of == 'start_distance') else None

    sns.set_style('white')
    fig, ax = plt.subplots(figsize=figsize)
    sns.lineplot(data=D, x=x, y=metric, hue=hue, marker='o',
                 palette='Set2', ax=ax, **_agg_kwargs(agg, errorbar))

    if chance is not None:
        ax.axhline(chance, ls='--', color='red', lw=1,
                   label=f'{chance_label} = {chance:+.2f}'
                         if metric in ('dD', 'delta_cm') else chance_label)
    if metric in ('dD', 'delta_cm'):
        ax.axhline(0.0, ls='-', color='0.75', lw=1, zorder=0,
                   label='No change in distance')
    if ceiling is not None:
        ax.axhline(ceiling, ls=':', color='green', lw=1,
                   label=f'Ceiling (perfectly direct = {ceiling:.2f})')

    # dD is "lower is better", so invert the axis to keep goal-directedness upward
    if metric == 'dD':
        ax.invert_yaxis()

    ax.set_xlabel('Trial number' if x == 'TrialNumber' else 'Age (days)')
    _tick_every_integer(ax, D[x])                # label every 1, 2, 3 ... 20
    ax.set_ylabel({'g': f'Fraction of onset distance closed by {frac:.0%}',
                   'dD': f'ΔD at the {frac:.0%} mark (cm)   ← closer to goal',
                   'delta_cm': f'Distance closed by the {frac:.0%} mark (cm)',
                   'g_rank_pct': 'Target rank vs ring-matched wells',
                   'g_z': 'z vs ring-matched wells'}.get(metric, metric))
    ax.set_title(title or
                 f'Path-progress goal-directedness ({metric}, '
                 f'frac={frac:.0%}, frac_of={frac_of}) vs {x.lower()}')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    plt.tight_layout()
    plt.show()
    return fig


AGE_BANDS = ((17, 20, 'Pre-weaning (P17-20)'),
             (21, 24, 'Peri-weaning (P21-24)'),
             (25, 32, 'Post-weaning (P25-32)'))   # matches visualization.py


def add_age_band(D, age_bands=AGE_BANDS):
    """Add a 'Band' column using the pipeline's standard age bands."""
    D = D.copy()
    D['Band'] = pd.cut(D['Age'],
                       [age_bands[0][0] - 1] + [b[1] for b in age_bands],
                       labels=[b[2] for b in age_bands])
    return D


def plot_progress_panels(trials, litters=(9, 10, 11),
                         metrics=('dD', 'g_rank_pct'), x='TrialNumber',
                         session_types=('Training', 'Probe'),
                         age_bands=AGE_BANDS, frac=0.2, frac_of='total_path',
                         completed_only=True, drop_rotation=True,
                         first_trial_baited=None, agg='mean', errorbar=None,
                         figsize=None, suptitle=None):
    """
    The recommended main figure: both candidate x-axes in one panel.

    Columns = session type (Training | Probe), rows = metric, one line per age
    band. Putting trial number on x and age in the hue answers "does it improve
    within a session" and "does it improve across development" simultaneously,
    and mirrors sv.plot_efficiency_across_trials(band_by=...).

    Plotting dD and g_rank_pct as two ROWS is deliberate. dD is the absolute
    distance closed; g_rank_pct is the same movement scored against the other
    wells on the target's ring. An age shift in the top row WITHOUT one in the
    bottom row means the animals close more ground with age but not more ground
    TOWARD THE TARGET specifically - i.e. general locomotor development rather
    than sharpening goal-directedness. The two rows together are the result;
    either alone is a misleading figure.
    """
    D = add_age_band(build_progress_df(trials, litters=litters,
                                       first_trial_baited=first_trial_baited,
                                       drop_rotation=drop_rotation,
                                       completed_only=completed_only,
                                       frac=frac, frac_of=frac_of), age_bands)
    if D.empty:
        print('No usable trials.')
        return None

    sns.set_style('white')
    nr, nc = len(metrics), len(session_types)
    fig, axes = plt.subplots(nr, nc, squeeze=False,
                             figsize=figsize or (6.2 * nc, 4.4 * nr))
    chance = {'g': 0.0, 'dD': 0.0, 'delta_cm': 0.0, 'g_rank_pct': 0.5, 'g_z': 0.0}
    ylab = {'dD': 'ΔD at the 20% mark (cm)',
            'delta_cm': 'Distance closed (cm)',
            'g': f'Fraction of onset distance closed by {frac:.0%}',
            'g_rank_pct': 'Target rank vs ring-matched wells',
            'g_z': 'z vs ring-matched wells'}

    for i, met in enumerate(metrics):
        for j, s in enumerate(session_types):
            ax = axes[i][j]
            sub = D[D.SessionType == s]
            if sub.empty:
                ax.axis('off')
                continue
            sns.lineplot(data=sub, x=x, y=met, hue='Band', marker='o',
                         palette='viridis', ax=ax,
                         legend=(i == 0 and j == nc - 1),
                         **_agg_kwargs(agg, errorbar))
            if chance.get(met) is not None:
                ax.axhline(chance[met], ls='--', color='red', lw=1)
            if met == 'dD':
                ax.invert_yaxis()          # keep goal-directedness upward
            ax.set_title(f'{s}' if i == 0 else '')
            ax.set_ylabel(ylab.get(met, met) if j == 0 else '')
            ax.set_xlabel('Trial number' if x == 'TrialNumber' else 'Age (days)')
            _tick_every_integer(ax, sub[x])

    axes[0][nc - 1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    fig.suptitle(suptitle or
                 f'Goal-directedness at the {frac:.0%} path mark '
                 f'(↑ = more goal-directed in both rows)', y=1.01)
    plt.tight_layout()
    plt.show()
    return fig


def plot_progress_vs_efficiency(trials, litters=(9, 10, 11), frac=0.2,
                                frac_of='total_path', hue='Env',
                                metric='g', drop_rotation=True, figsize=(7, 6)):
    """
    Diagnostic: how much of the measure is path efficiency in disguise?

    The dashed guide is the algebraic ceiling frac * efficiency. Raw g under
    frac_of='total_path' is bounded by it, so points hugging that line are ceiling-limited
    rather than informative.

    Run it for metric='g' and metric='g_rank_pct' and compare the Spearman rho in
    each title. Some correlation with efficiency is expected either way and is not
    itself a problem - a rat that heads straight for the well is both efficient and
    goal-directed. What you are checking is whether g_rank_pct still separates your
    groups once efficiency is in the model (see the mixedlm in the driver cells).
    """
    D = build_progress_df(trials, litters=litters, drop_rotation=drop_rotation,
                          frac=frac, frac_of=frac_of)
    if D.empty:
        print('No usable trials.')
        return None

    sns.set_style('white')
    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(data=D, x='efficiency', y=metric, hue=hue, alpha=0.6,
                    palette='Set2', ax=ax)

    e = np.linspace(1, max(2.0, D['efficiency'].quantile(0.99)), 100)
    if metric == 'g':
        ax.plot(e, np.minimum(frac * e, 1.0), ls='--', color='gray', lw=1.2,
                label='Algebraic ceiling (frac x efficiency)')
        ax.axhline(frac, ls=':', color='green', lw=1,
                   label=f'Perfectly direct = {frac:.2f}')
    elif metric == 'g_rank_pct':
        ax.axhline(0.5, ls='--', color='red', lw=1, label='Chance')

    r = D[['efficiency', metric]].corr(method='spearman').iloc[0, 1]
    ax.set_xlabel('Path efficiency (real / ideal)')
    ax.set_ylabel({'g': f'Fraction of onset distance closed by {frac:.0%}',
                   'g_rank_pct': 'Target rank vs ring-matched wells'}.get(metric, metric))
    ax.set_title(f'{metric} vs efficiency (frac_of={frac_of})   Spearman rho = {r:+.2f}')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()
    return fig


def compare_trial_blocks(trials, litters=(9, 10, 11), metric='dD',
                         blocks=((1, 1, 'Trial 1'),
                                 (2, 3, 'Trials 2-3'),
                                 (4, 20, 'Trials 4-20')),
                         session='Training', frac=0.2, frac_of='total_path',
                         completed_only=True, first_trial_baited=None,
                         figsize=(8, 6), title=None):
    """
    Test for a STEP in goal-directedness across the first few trials of a session,
    rather than a gradual slope.

    Why this and not the slope: the rewarded well CHANGES between sessions in this
    paradigm (e.g. L9 Banjo's 7 training sessions use wells 1, 2, 6, 8, 12, 13), so
    every session's trial 1 is naive with respect to that session's target. If the
    animals then solve it in one or two trials and plateau, the learning is a step
    at trial 1 -> 2, and a linear fit across trials 1-20 is nearly blind to it.
    plot_progress_delta(x='TrialNumber') tests the slope; this tests the step.

    Animal is the unit of analysis: each animal contributes one mean per block, so
    the paired tests are not inflated by trial count.

    Returns (fig, table, tests) — table is per-animal block means, tests is a dict
    of Friedman across blocks plus pairwise Wilcoxon.
    """
    D = build_progress_df(trials, litters=litters, completed_only=completed_only,
                          first_trial_baited=first_trial_baited,
                          frac=frac, frac_of=frac_of)
    if D.empty:
        print('No usable trials.')
        return None
    if session is not None:
        D = D[D.SessionType == session]

    labels = [b[2] for b in blocks]
    D = D.copy()
    D['Block'] = pd.Series(None, index=D.index, dtype='object')   # object, not float
    for lo, hi, lab in blocks:
        D.loc[D.TrialNumber.between(lo, hi), 'Block'] = lab
    D = D.dropna(subset=['Block'])

    table = (D.groupby(['Subject', 'Block'])[metric].mean()
               .unstack().reindex(columns=labels))
    complete = table.dropna()

    tests = {'n_animals_complete': len(complete), 'n_trials': len(D)}
    if len(complete) > 2:
        tests['friedman_p'] = float(
            _sps.friedmanchisquare(*[complete[c] for c in labels]).pvalue)
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            pair = table[[labels[i], labels[j]]].dropna()
            if len(pair) > 2:
                tests[f'{labels[i]} vs {labels[j]}'] = (
                    len(pair),
                    float(_sps.wilcoxon(pair.iloc[:, 0], pair.iloc[:, 1]).pvalue))

    sns.set_style('white')
    fig, ax = plt.subplots(figsize=figsize)
    for _, row in table.iterrows():                       # one faint line per animal
        ax.plot(labels, row.values, color='gray', alpha=0.45, lw=1,
                marker='o', ms=4, zorder=1)
    ax.errorbar(labels, table.mean(), yerr=table.sem(), color='tab:blue', lw=2.5,
                marker='o', ms=9, capsize=5, zorder=3, label='Mean ± SEM')

    null = D['dD_null'].mean() if metric in ('dD', 'delta_cm') else \
        {'g': 0.0, 'g_rank_pct': 0.5, 'g_z': 0.0}.get(metric)
    if null is not None:
        if metric == 'delta_cm':
            null = -null
        ax.axhline(null, ls='--', color='red', lw=1,
                   label=f'Chance = {null:+.2f}' if metric in ('dD', 'delta_cm')
                   else 'Chance')
    if metric == 'dD':
        ax.invert_yaxis()                                 # keep "better" upward

    ax.set_ylabel({'dD': f'ΔD at the {frac:.0%} mark (cm)   ← closer to goal',
                   'g_rank_pct': 'Target rank vs ring-matched wells'}.get(metric, metric))
    ax.set_title(title or
                 f'{session or "All"}: is there a step after trial 1?  '
                 f'(n = {len(complete)} animals with all blocks)')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()

    print(f'\nper-animal {metric} by block (n trials = {len(D)}):')
    print(table.round(3).to_string())
    print('\ntests:')
    for k, v in tests.items():
        print(f'  {k}: {v}')
    return fig, table, tests


def plot_progress_arena(trial, frac=0.2, frac_of='total_path', **kwargs):
    """
    Single-trial arena illustration at the `frac` snapshot, reusing the drawing
    code in goal_directedness_arena (which takes an absolute path window in cm).

    Shows the real Circle/Square boundary, all 17 wells, the target, the start
    (T=0), the point reached after frac of the path, the path itself, and the two
    straight-line distances to the target with their values in cm.
    """
    import squircle.goal_directedness_arena as gda

    m = compute_progress_approach(trial[15], trial[8], frac=frac, frac_of=frac_of)
    if m is None:
        print(f'No usable trajectory for {trial[0]} trial {trial[10]}.')
        return None
    return gda.plot_onset_distance_arena(trial, window=m['s_used'], **kwargs)


def plot_progress_arena_by_day(trials, litters=(9, 10, 11), subject=None,
                               trial_number=1, session='Training', env=None,
                               per_figure=3, ncols=3, frac=0.2,
                               frac_of='total_path', figsize=None):
    """
    One figure per block of session-days, instead of a single tall grid.

    With per_figure=3 an animal that ran 7 training sessions gives three figures:
    days 1-3, days 4-6, and day 7 on its own. The last block is however many days
    are left over - it is not padded.

    "Day" is the session's rank in that animal's own chronological order for the
    given session type (day 1 = its earliest), matching how `days=` works in
    visualization.plot_efficiency_across_trials. It is NOT the calendar date.

    subject : one name, a list of names, or None for every animal present
        (one set of figures per animal - that can be a lot of windows).
    trial_number : default 1, the naive trial of each session.

    Returns a list of figures, in order.
    """
    trials = _as_trials(trials, litters)
    pool = [t for t in trials
            if t[7] == 0
            and (session is None or t[3] == session)
            and (env is None or t[4] == env)
            and (trial_number is None or int(t[10]) == int(trial_number))]
    if not pool:
        print('No usable trials.')
        return []

    if subject is None:
        names = sorted({t[0] for t in pool})
    elif isinstance(subject, str):
        names = [subject]
    else:
        names = list(subject)

    figs = []
    for name in names:
        mine = [t for t in pool if t[0] == name]
        if not mine:
            print(f'No trials for {name}.')
            continue
        # rank this animal's sessions chronologically -> day index 1..N
        days = sorted({(t[6], t[5]) for t in mine})
        day_of = {key: i + 1 for i, key in enumerate(days)}

        for start in range(0, len(days), per_figure):
            block = days[start:start + per_figure]
            nums = [day_of[k] for k in block]
            chunk = [t for t in mine if (t[6], t[5]) in set(block)]
            label = (f'day {nums[0]}' if len(nums) == 1
                     else f'days {nums[0]}-{nums[-1]}')
            f = plot_progress_arena_grid(
                chunk, frac=frac, frac_of=frac_of, select='all',
                ncols=min(ncols, len(chunk)), figsize=figsize,
                suptitle=(f'{name} — {label} — trial {trial_number} '
                          f'({session or "all sessions"}), '
                          f'distance to target at start vs at the {frac:.0%} mark'))
            if f is not None:
                figs.append(f)
    return figs


def plot_progress_arena_grid(trials, litters=(9, 10, 11), frac=0.2,
                             frac_of='total_path', env=None, session=None,
                             subject=None, trial_number=None,
                             select='all', n=None, ncols=3, seed=0,
                             annotate=True, figsize=None, suptitle=None):
    """
    Grid of arena illustrations, each drawn at its OWN frac-of-path point.

    Not the same as gda.plot_onset_distance_grid, which applies one absolute
    window (in cm) to every trial. Here each trial's snapshot is frac * its own
    path length, which is what the dD measure actually uses.

    select : how to choose which trials to show
        'all'    - (default) every matching trial, in chronological order (age,
                   then date). With trial_number=1 and one subject this gives one
                   panel per session day, which is what you want for "did this
                   animal look goal-directed on its naive trial, day by day".
        'span'   - sorted by dD and sampled evenly, so the grid spans the range
                   from strongly goal-directed (dD very negative, top-left) to
                   strongly away (dD positive, bottom-right). Good for seeing what
                   the measure responds to, but the panels are NOT in time order -
                   do not read them as a sequence.
        'best'   - the n most negative dD (clearest approaches)
        'worst'  - the n most positive dD
        'random' - a random sample (seed fixed for reproducibility)
        'first'  - the first n usable trials, in load order
    env     : 'Circle' | 'Square' | None (both)
    session : 'Training' | 'Probe' | None (both)
    subject : one animal's name, or None for all
    trial_number : keep only this within-session trial number (e.g. 1 to look
        at naive trials only), or None for all
    """
    import squircle.goal_directedness_arena as gda

    trials = _as_trials(trials, litters)
    rows = []
    for t in trials:
        if t[7] != 0:
            continue
        if env is not None and t[4] != env:
            continue
        if session is not None and t[3] != session:
            continue
        if subject is not None and t[0] != subject:
            continue
        if trial_number is not None and int(t[10]) != int(trial_number):
            continue
        m = compute_progress_approach(t[15], t[8], frac=frac, frac_of=frac_of)
        if m is None:
            continue
        rows.append((m['dD'], m['s_used'], t, m))
    if not rows:
        print('No usable trials.')
        return None

    rows.sort(key=lambda r: r[0])                     # most negative dD first
    if n is None and select not in ('all', 'chronological'):
        n = 9
    if select == 'span':
        idx = np.unique(np.linspace(0, len(rows) - 1, min(n, len(rows))).astype(int))
        chosen = [rows[i] for i in idx]
    elif select == 'best':
        chosen = rows[:n]
    elif select == 'worst':
        chosen = rows[-n:][::-1]
    elif select == 'random':
        rng = np.random.default_rng(seed)
        chosen = [rows[i] for i in
                  sorted(rng.choice(len(rows), min(n, len(rows)), replace=False))]
    elif select == 'first':
        chosen = rows[:n]
    elif select in ('all', 'chronological'):
        # every matching trial, in the order it was run (age, then date)
        chosen = sorted(rows, key=lambda r: (r[2][6], r[2][5], r[2][0]))
        if n is not None:
            chosen = chosen[:n]
    else:
        raise ValueError("select must be 'all','span','best','worst',"
                         " 'random' or 'first'")

    nrows = int(np.ceil(len(chosen) / ncols))
    # Panels are drawn with equal aspect, so a near-square panel keeps the arena
    # as large as possible; extra height would just become whitespace.
    # layout='constrained' (not tight_layout) is what actually keeps 3-line titles,
    # the suptitle and the figure legend from colliding at any figure size.
    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             figsize=figsize or (4.7 * ncols, 5.0 * nrows),
                             layout='constrained')
    axes = axes.flatten()
    for k, (ax, (dD, s_used, t, m)) in enumerate(zip(axes, chosen)):
        # annotate=False: gda writes both cm values at the LINE MIDPOINTS, which
        # collide whenever start and snapshot are near each other. The numbers go
        # in the title instead, where they are always readable.
        gda.plot_onset_distance_arena(t, window=s_used, ax=ax, annotate=False)
        # gda labels its artists with the ABSOLUTE window ("First 162 cm of path"),
        # which is meaningless here because the window differs per panel. Relabel
        # them with the fraction, which is the same for every panel.
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if 'of path' in lab:
                h.set_label(f'First {frac:.0%} of path')
            elif 'point' in lab:
                h.set_label(f'{frac:.0%} point')
        ax.set_title(f"{t[0]}  P{t[6]}  trial {int(t[10])}  ({t[4]})\n"
                     f"{m['d0']:.1f} → {m['d_at']:.1f} cm   ΔD = {dD:+.1f}\n"
                     f"walked {s_used:.1f} cm",
                     fontsize=9, linespacing=1.4)
        # axis labels only on the outer edge - they are identical everywhere
        if k % ncols:
            ax.set_ylabel('')
        if k < len(chosen) - ncols:
            ax.set_xlabel('')
        ax.tick_params(labelsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes[len(chosen):]:
        ax.axis('off')

    fig.suptitle(suptitle or
                 f'Distance to target at start vs at the {frac:.0%} path mark'
                 + (f'  —  {env}' if env else ''), fontsize=13)
    # one shared legend rather than one per panel (they are all identical).
    # 'outside lower center' is reserved space under constrained layout, so it
    # cannot overlap the bottom row; fall back for older matplotlib.
    if handles:
        try:
            fig.legend(handles, labels, loc='outside lower center',
                       ncol=min(len(handles), 4), fontsize=9, frameon=False)
        except ValueError:
            fig.legend(handles, labels, loc='lower center',
                       ncol=min(len(handles), 4), fontsize=9, frameon=False)
    plt.show()
    return fig


# ======================================================================
# Day-1 within-session learning speed  (three parallel views)
# ======================================================================
"""
Speed of learning WITHIN a single session, on day 1 only.

Motivation and the caveat that shaped it
    plot_progress_delta's docstring records that, POOLED ACROSS ALL DAYS,
    within-session trial number carries no learning signal in litters 9-11
    (efficiency vs trial: Spearman rho = +0.009, p = 0.68). That pooled null is
    exactly why day 1 is the interesting slice: from day 2 on the animals are
    already trained, so their day-2 trial-1 is already good and there is no
    within-session slope left to see - averaging days 2-7 (flat) into day 1
    washes out any day-1 effect. Day 1 is the only session in which a
    within-session learning slope can exist, and the day-1 efficiency curve for
    the older bands already shows it: a steep drop over the first few trials to a
    floor.

    Two honest caveats travel with every figure here:
      * N per band is small (entry banding: 5 / 6 / 4 for litters 9/10/11), so
        the between-band comparison is descriptive, not powered.
      * Probe day-1 comes AFTER that day's 20 training trials, so it is NOT a
        naive session - the naivety story is strongest for day-1 TRAINING. Probe
        is shown for completeness.

Three views of the same question, so they can be judged against one another
    (1) plot_day1_slopes   - one learning-RATE slope per animal (Theil-Sen robust
                             + OLS; efficiency vs Trial and vs log Trial).
    (2) plot_day1_block    - early-vs-late block drop per animal, agg(T1-3) minus
                             agg(T8-10). The most assumption-free view.
    (3) fit_day1_mixedlm   - mixed-effects model with a Band x trial interaction;
                             the interaction IS the "do bands learn at different
                             rates" test, with partial pooling across animals.

    compare_day1_learning_speed runs all three and prints one combined table.

Efficiency is real/ideal (>= 1, right-skewed, heavy-tailed), so the slope and
model views default to fitting log(efficiency): a straight line in log space is a
constant fractional change per trial, which is the natural "rate" here and tames
the outliers. The block view uses raw efficiency by default (a median of three
trials is already robust) but takes response='logEff' if you want it in log space
too. Lower efficiency = more direct, so a NEGATIVE slope / POSITIVE block drop =
learning.
"""


# One fixed band -> colour map, matching the published efficiency figure
# (visualization.plot_efficiency_across_trials: green / blue / orange) so all the
# day-1 views are colour-consistent with each other AND with that plot.
BAND_PALETTE = {AGE_BANDS[0][2]: 'tab:green',
                AGE_BANDS[1][2]: 'tab:blue',
                AGE_BANDS[2][2]: 'tab:orange'}


def _date_key(d):
    """Ordinal sort key for the JSON's 'dd_mm_yy' date strings (mirrors visualization)."""
    dd, mm, yy = d.split('_')
    return (int(yy), int(mm), int(dd))


def _band_of(age, age_bands):
    for lo, hi, label in age_bands:
        if lo <= age <= hi:
            return label
    return None


def _ideal_distance_cm(df, rewarded_well):
    """Start-to-reward straight-line distance, the denominator of efficiency."""
    start_x, start_y = df.iloc[0]['x'], df.iloc[0]['y']
    rx, ry = well_locations_cm[rewarded_well - 1]
    return float(np.hypot(rx - start_x, ry - start_y))


def build_day1_efficiency_df(working_folder, litters=(9, 10, 11),
                             age_bands=AGE_BANDS,
                             session_types=('Training', 'Probe'),
                             trials=range(1, 11), day=1,
                             min_ideal_cm=8.0, exclude_rotation=True):
    """
    Tidy per-trial path efficiency for one ordinal session-day - the substrate for
    the three day-1 learning-speed views.

    Mirrors the extraction inside visualization.plot_efficiency_across_trials
    (entry-age banding, min_ideal_cm start-distance gate, real/ideal efficiency via
    st.calculate_efficiency, rotated probes excluded) but returns a DataFrame and
    does not plot, so all three views read the same numbers behind the published
    efficiency figure.

    Parameters
    ----------
    trials : iterable of int
        Within-session trial numbers to keep (default 1-10: the older bands floor
        by ~trial 4, so 1-10 captures the drop without diluting it with the long
        flat tail; pass range(1, 21) for the full training session).
    day : int
        Which ordinal session-day (1 = each animal's earliest session of that
        type). Day 1 is the naive session for Training.

    Returns
    -------
    DataFrame with columns Subject, Band, Session, Trial, Efficiency, logEff,
    logTrial. 'Band' is fixed per animal from the age at its first Training session
    (entry banding, N = 5/6/4), stored as an ordered Categorical in age order.
    """
    trial_set = {int(t) for t in trials}
    band_order = [b[2] for b in age_bands]
    rows = []
    for litter in litters:
        path = os.path.join(working_folder, f"litter_{litter}_processed_cm.json")
        if not os.path.exists(path):
            continue
        with open(path, 'r') as f:
            data = json.load(f)

        for subject in data:
            train = [s for s in subject['Sessions'] if s['Type'] == 'Training']
            entry_band = (_band_of(min(train, key=lambda s: _date_key(s['Date']))['Age'],
                                   age_bands) if train else None)
            if entry_band is None:
                continue

            for stype in session_types:
                sess = [s for s in subject['Sessions'] if s['Type'] == stype]
                if stype == 'Probe' and exclude_rotation:
                    sess = [s for s in sess if s.get('Rotation', 0) == 0]
                sess = sorted(sess, key=lambda s: _date_key(s['Date']))
                if len(sess) < day:
                    continue
                s = sess[day - 1]
                well = s['Rewarded well']

                for trial in s['Trials']:
                    if int(trial['Number']) not in trial_set:
                        continue
                    sf, ef = trial['Start frame'], trial['End frame']
                    if pd.isna(sf) or pd.isna(ef) or ef <= sf:
                        continue
                    df = pd.read_json(StringIO(trial['Position']))
                    if df.empty:
                        continue
                    ideal = _ideal_distance_cm(df, well)
                    if not np.isfinite(ideal) or ideal < min_ideal_cm:
                        continue
                    rows.append({
                        'Subject': subject['Name'], 'Band': entry_band,
                        'Session': stype, 'Trial': int(trial['Number']),
                        'Efficiency': float(st.calculate_efficiency(df, well)),
                    })

    D = pd.DataFrame(rows)
    if not D.empty:
        D['logEff'] = np.log(D['Efficiency'])
        D['logTrial'] = np.log(D['Trial'])
        D['Band'] = pd.Categorical(D['Band'], categories=band_order, ordered=True)
    return D


# ----------------------------------------------------------------------
# View 1: per-animal slope
# ----------------------------------------------------------------------

def compute_day1_slopes(D, predictors=('Trial', 'logTrial'), response='logEff'):
    """
    One learning-rate slope per (Subject, Session), by Theil-Sen (robust to the
    efficiency outliers) and OLS, against each predictor.

    NEGATIVE slope = efficiency falls across trials = learning. Needs >= 3 trials
    with spread in the predictor for a slope.

    Returns long DataFrame: Subject, Band, Session, Predictor, Method, Slope, n.
    """
    recs = []
    for (subj, band, sess), g in D.groupby(['Subject', 'Band', 'Session'],
                                           observed=True):
        g = g.dropna(subset=[response])
        for pred in predictors:
            x = g[pred].to_numpy(float)
            y = g[response].to_numpy(float)
            if len(x) < 3 or np.ptp(x) == 0:
                continue
            ts = _sps.theilslopes(y, x)
            ols = _sps.linregress(x, y)
            base = {'Subject': subj, 'Band': band, 'Session': sess,
                    'Predictor': pred, 'n': int(len(x))}
            recs.append({**base, 'Method': 'TheilSen', 'Slope': float(ts[0])})
            recs.append({**base, 'Method': 'OLS', 'Slope': float(ols.slope)})
    return pd.DataFrame(recs)


def _strip_by_band(ax, sub, order, ycol='Slope'):
    """Per-animal dots + a black median bar per band on one axis."""
    sns.stripplot(data=sub, x='Band', y=ycol, order=order, hue='Band',
                  hue_order=order, palette=BAND_PALETTE, size=9, alpha=0.85,
                  jitter=0.18, ax=ax, legend=False)
    meds = sub.groupby('Band', observed=True)[ycol].median()
    for xi, b in enumerate(order):
        if b in meds.index and np.isfinite(meds[b]):
            ax.hlines(meds[b], xi - 0.32, xi + 0.32, color='k', lw=2.2, zorder=5)


def plot_day1_slopes(working_folder, response='logEff', method='TheilSen',
                     session_types=('Training', 'Probe'), age_bands=AGE_BANDS,
                     figsize=None, title=None, **build_kw):
    """
    View 1 - per-animal within-session learning rate on day 1.

    Rows = predictor (efficiency vs Trial; vs log Trial, which weights the early
    trials where the drop lives). Columns = session type. One dot per animal, black
    bar = per-band median, red line = 0 (no within-session change). Points below 0
    = that animal got more efficient across the session.

    method : 'TheilSen' (robust, headline) or 'OLS'.
    Extra **build_kw pass to build_day1_efficiency_df (litters, trials, day, ...).

    Returns (fig, slopes_df) - slopes_df carries BOTH methods for the table.
    """
    D = build_day1_efficiency_df(working_folder, session_types=session_types,
                                 age_bands=age_bands, **build_kw)
    if D.empty:
        print('No usable trials.')
        return None, None
    S = compute_day1_slopes(D, response=response)
    Sm = S[S.Method == method]
    order = [b[2] for b in age_bands]
    preds = ['Trial', 'logTrial']

    sns.set_style('whitegrid')
    nr, nc = len(preds), len(session_types)
    fig, axes = plt.subplots(nr, nc, squeeze=False,
                             figsize=figsize or (5.6 * nc, 4.2 * nr))
    for i, pred in enumerate(preds):
        for j, sess in enumerate(session_types):
            ax = axes[i][j]
            sub = Sm[(Sm.Predictor == pred) & (Sm.Session == sess)]
            if sub.empty:
                ax.axis('off')
                continue
            _strip_by_band(ax, sub, order)
            ax.axhline(0, ls='--', color='red', lw=1)
            note = '  (post-training, not naive)' if sess == 'Probe' else ''
            ax.set_title(f'{sess} — vs {pred}{note}', fontsize=10)
            ax.set_ylabel(f'{method} slope  d({response})/d({pred})\n← faster learning'
                          if j == 0 else '')
            ax.set_xlabel('')
            ax.tick_params(axis='x', rotation=12)

    fig.suptitle(title or
                 'Day-1 within-session learning rate — per-animal slopes '
                 f'({method}; negative = learning; N=5/6/4)', y=1.02)
    plt.tight_layout()
    plt.show()
    return fig, S


# ----------------------------------------------------------------------
# View 2: early-vs-late block
# ----------------------------------------------------------------------

def compute_day1_block(D, early=(1, 2, 3), late=(8, 9, 10), agg='median',
                       response='Efficiency'):
    """
    Per (Subject, Session): agg(early trials) - agg(late trials) on day 1.

    POSITIVE = efficiency fell from early to late = learning. The most
    assumption-free view - no model, no single noisy baseline trial.

    Returns Subject, Band, Session, Early, Late, Drop.
    """
    aggf = np.median if agg == 'median' else np.mean
    early, late = set(early), set(late)
    recs = []
    for (subj, band, sess), g in D.groupby(['Subject', 'Band', 'Session'],
                                           observed=True):
        e = g.loc[g.Trial.isin(early), response]
        l = g.loc[g.Trial.isin(late), response]
        if e.empty or l.empty:
            continue
        ev, lv = float(aggf(e)), float(aggf(l))
        recs.append({'Subject': subj, 'Band': band, 'Session': sess,
                     'Early': ev, 'Late': lv, 'Drop': ev - lv})
    return pd.DataFrame(recs)


def plot_day1_block(working_folder, early=(1, 2, 3), late=(8, 9, 10),
                    agg='median', response='Efficiency',
                    session_types=('Training', 'Probe'), age_bands=AGE_BANDS,
                    figsize=None, title=None, **build_kw):
    """
    View 2 - early-vs-late efficiency drop per animal on day 1.

    One dot per animal = agg(T1-3) - agg(T8-10); black bar = per-band median; red
    line = 0 (no change). Points above 0 = that animal ended the session more
    efficient than it started. Extra **build_kw pass to build_day1_efficiency_df.

    Returns (fig, block_df).
    """
    D = build_day1_efficiency_df(working_folder, session_types=session_types,
                                 age_bands=age_bands, **build_kw)
    if D.empty:
        print('No usable trials.')
        return None, None
    B = compute_day1_block(D, early=early, late=late, agg=agg, response=response)
    if B.empty:
        print('No animal had both early and late trials.')
        return None, B
    order = [b[2] for b in age_bands]

    sns.set_style('whitegrid')
    nc = len(session_types)
    fig, axes = plt.subplots(1, nc, squeeze=False, figsize=figsize or (5.6 * nc, 4.4))
    for j, sess in enumerate(session_types):
        ax = axes[0][j]
        sub = B[B.Session == sess]
        if sub.empty:
            ax.axis('off')
            continue
        _strip_by_band(ax, sub, order, ycol='Drop')
        ax.axhline(0, ls='--', color='red', lw=1)
        note = '  (post-training, not naive)' if sess == 'Probe' else ''
        ax.set_title(f'{sess}{note}', fontsize=10)
        ax.set_ylabel(f'{agg}(T{min(early)}-{max(early)}) − {agg}(T{min(late)}-{max(late)})'
                      f'  [{response}]\n↑ = more learning' if j == 0 else '')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=12)

    e0, e1, l0, l1 = min(early), max(early), min(late), max(late)
    fig.suptitle(title or
                 f'Day-1 early-vs-late drop — T{e0}-{e1} minus T{l0}-{l1} '
                 f'({agg} {response}; ↑ = learning; N=5/6/4)', y=1.03)
    plt.tight_layout()
    plt.show()
    return fig, B


# ----------------------------------------------------------------------
# View 3: mixed-effects model
# ----------------------------------------------------------------------

def fit_day1_mixedlm(D, session='Training', predictor='logTrial', response='logEff',
                     age_bands=AGE_BANDS):
    """
    Mixed-effects model of day-1 within-session learning, with a band x trial
    interaction and partial pooling across animals.

        response ~ predictor * C(Band)   + random (intercept [+ slope]) | Subject

    The predictor:C(Band) interaction terms ARE the between-band difference in
    learning rate, relative to the reference (first / youngest) band. A random
    slope is attempted first; with N = 4-6 animals per band it often fails to
    converge, in which case this falls back to a random intercept and says so.

    Returns (result, coef_table, kind) where coef_table is a tidy DataFrame of the
    fixed effects (estimate, 95% CI, p) and `kind` names which random structure was
    actually fitted. Returns (None, None, reason) when the model cannot be fit.
    """
    try:
        import statsmodels.formula.api as smf
    except Exception as exc:                       # pragma: no cover
        return None, None, f'statsmodels unavailable ({exc})'

    d = D[D.Session == session].dropna(subset=[response, predictor]).copy()
    if d.empty or d['Band'].nunique() < 2:
        return None, None, 'not enough data / bands'
    # reference = youngest band present, so interaction signs read "older vs young"
    present = [b[2] for b in age_bands if b[2] in set(d['Band'].astype(str))]
    d['Band'] = pd.Categorical(d['Band'].astype(str), categories=present, ordered=True)
    d = d.rename(columns={response: '_y', predictor: '_x'})

    formula = '_y ~ _x * C(Band)'
    kind = 'random slope + intercept'
    try:
        res = smf.mixedlm(formula, d, groups=d['Subject'],
                          re_formula='~_x').fit(reml=True, method='lbfgs')
        if not res.converged:
            raise RuntimeError('did not converge')
    except Exception:
        kind = 'random intercept only (random-slope fit failed)'
        res = smf.mixedlm(formula, d, groups=d['Subject']).fit(reml=True,
                                                               method='lbfgs')

    ci = res.conf_int()
    table = pd.DataFrame({
        'estimate': res.fe_params,
        'ci_low': ci.loc[res.fe_params.index, 0],
        'ci_high': ci.loc[res.fe_params.index, 1],
        'p': res.pvalues.loc[res.fe_params.index],
    })
    table.attrs['reference_band'] = present[0]
    table.attrs['predictor'] = predictor
    return res, table, kind


def plot_day1_mixedlm(working_folder, session='Training', predictor='logTrial',
                      response='logEff', age_bands=AGE_BANDS, figsize=(8, 5.5),
                      title=None, **build_kw):
    """
    View 3 - fit the mixed model on day-1 `session` data and draw the fitted
    per-band mean lines (fixed effects) over the observed per-band trial medians.

    Diverging fitted slopes = bands learn at different within-session rates. The
    printed coef_table gives the interaction estimates and p-values behind the
    picture. Extra **build_kw pass to build_day1_efficiency_df.

    Returns (fig, result, coef_table, kind).
    """
    D = build_day1_efficiency_df(working_folder, session_types=(session,),
                                 age_bands=age_bands, **build_kw)
    if D.empty:
        print('No usable trials.')
        return None, None, None, 'no data'
    res, table, kind = fit_day1_mixedlm(D, session=session, predictor=predictor,
                                        response=response, age_bands=age_bands)
    if res is None:
        print(f'Mixed model not fit: {kind}')
        return None, None, None, kind

    print(f'\nDay-1 {session} mixed model  ({kind})')
    print(f"  response={response}  predictor={predictor}  "
          f"reference band={table.attrs['reference_band']}")
    with pd.option_context('display.float_format', lambda v: f'{v:+.4f}'):
        print(table.to_string())

    d = D[D.Session == session].dropna(subset=[response, predictor]).copy()
    present = [b[2] for b in age_bands if b[2] in set(d['Band'].astype(str))]
    palette = BAND_PALETTE

    grid = np.sort(d[predictor].unique())
    sns.set_style('whitegrid')
    fig, ax = plt.subplots(figsize=figsize)
    for band in present:
        gb = d[d['Band'].astype(str) == band]
        med = gb.groupby('Trial')[response].median()
        ax.plot(np.log(med.index) if predictor == 'logTrial' else med.index,
                med.values, 'o', color=palette[band], alpha=0.5, ms=6)
        # fitted fixed-effects line, built from the coef table (see _mixedlm_line)
        ax.plot(grid, _mixedlm_line(res, table, band, grid, predictor),
                '-', color=palette[band], lw=2.2, label=band)

    xlab = 'log(trial)' if predictor == 'logTrial' else 'trial'
    ax.set_xlabel(f'Within-session {xlab}')
    ax.set_ylabel(f'{response}  (lower = more efficient)')
    ax.set_title(title or
                 f'Day-1 {session}: fitted per-band learning lines ({kind})')
    ax.legend(title='Age band', fontsize=8)
    plt.tight_layout()
    plt.show()
    return fig, res, table, kind


def _mixedlm_line(res, table, band, grid, predictor):
    """Fixed-effects fitted line for one band from the coef table (robust to the
    C(Band) dummy naming, so it does not depend on res.predict's design handling)."""
    fe = table['estimate']
    intercept = fe.get('Intercept', 0.0)
    slope = fe.get('_x', 0.0)
    # additive band effect + interaction, matched by substring of the dummy name
    for name, val in fe.items():
        if f'C(Band)[T.{band}]' == name:
            intercept += val
        elif name.startswith('_x:') and name.endswith(f'C(Band)[T.{band}]'):
            slope += val
    return intercept + slope * grid


# ----------------------------------------------------------------------
# Wrapper: all three, one combined table
# ----------------------------------------------------------------------

def compare_day1_learning_speed(working_folder, session_types=('Training', 'Probe'),
                                age_bands=AGE_BANDS, show=True, **build_kw):
    """
    Run all three day-1 views and return everything, printing one combined per-band
    summary table so the metrics sit side by side for you to choose between.

    Combined table (per Band x Session): median Theil-Sen slope vs log-trial,
    median early-vs-late Drop, and n animals. The mixed-model interaction estimates
    are printed separately by plot_day1_mixedlm (Training and Probe), since they are
    contrasts against a reference band rather than a per-band number.

    Returns dict with keys: 'data', 'slopes', 'block', 'mixedlm', 'summary', 'figs'.
    """
    D = build_day1_efficiency_df(working_folder, session_types=session_types,
                                 age_bands=age_bands, **build_kw)
    if D.empty:
        print('No usable trials.')
        return None

    slopes = compute_day1_slopes(D)
    block = compute_day1_block(D)

    order = [b[2] for b in age_bands]
    ts_log = (slopes[(slopes.Method == 'TheilSen') & (slopes.Predictor == 'logTrial')]
              .groupby(['Session', 'Band'], observed=True)['Slope'].median())
    drop = block.groupby(['Session', 'Band'], observed=True)['Drop'].median()
    n = D.groupby(['Session', 'Band'], observed=True)['Subject'].nunique()
    summary = (pd.concat({'median_TS_logtrial_slope': ts_log,
                          'median_block_drop': drop, 'n_animals': n}, axis=1)
               .reset_index())
    summary['Band'] = pd.Categorical(summary['Band'], categories=order, ordered=True)
    summary = summary.sort_values(['Session', 'Band']).reset_index(drop=True)

    print('\n=== Day-1 within-session learning speed: combined summary ===')
    print('(slope negative = learning; block drop positive = learning; N per band '
          'small, descriptive)')
    with pd.option_context('display.float_format', lambda v: f'{v:+.4f}'):
        print(summary.to_string(index=False))

    figs, mm = {}, {}
    if show:
        figs['slopes'], _ = plot_day1_slopes(working_folder, age_bands=age_bands,
                                             session_types=session_types, **build_kw)
        figs['block'], _ = plot_day1_block(working_folder, age_bands=age_bands,
                                           session_types=session_types, **build_kw)
        for sess in session_types:
            figs[f'mixedlm_{sess}'], res, tab, kind = plot_day1_mixedlm(
                working_folder, session=sess, age_bands=age_bands, **build_kw)
            mm[sess] = {'result': res, 'table': tab, 'kind': kind}

    return {'data': D, 'slopes': slopes, 'block': block, 'mixedlm': mm,
            'summary': summary, 'figs': figs}

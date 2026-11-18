# -*- coding: utf-8 -*-
"""
Idleness analysis for the Squircle project.

Measures how much of each trial the rat spends idle (not progressing),
following the established open-field / exploratory-behavior literature:

Speed thresholding (method A)
    Hen et al. (2004), J Neurosci Methods 133:161-172: smooth the centroid
    path, then classify frames as moving/not-moving from instantaneous speed.
    Noldus' EthoVision-style dual threshold (1.75 / 2.00 cm/s hysteresis)
    prevents flip-flopping at the cutoff.

Data-driven threshold (Drai, Benjamini & Golani (2000), J Neurosci Methods
    96:119-131): instead of a fixed cutoff, the log-speed distribution is
    bimodal (a slow "stopping" mode and a fast "progression" mode); the mode
    boundary - estimated here by a two-component Gaussian mixture EM on
    log(speed) - is the threshold. `drai_benjamini_threshold()` gives the
    per-trial (or pooled) data-derived cutoff, which answers "where should my
    threshold be for juvenile rats" without importing a literature constant.

Dispersion bounding circle (method B)
    Sliding-window dispersion: a rat that grooms or turns its head stays
    inside a small circle even though its centroid speed spikes. Every window
    of `window_s` seconds whose positions all fall within `radius_cm` of the
    window mean is marked stationary; overlapping/touching marked windows
    merge into one bout, so the window length is also the minimum bout
    duration. Radius should be set from measured tracking noise
    (dispersion of clearly-still stretches), e.g. ~2x that noise.

Outputs
    idle_proportion          - per-trial proportion of frames idle/stationary
    plot_speed_threshold     - log-speed histogram + fitted mixture + threshold
    plot_trial_idleness      - diagnostic: speed trace, idle spans, trajectory
    plot_idle_proportion_by_band - THE summary figure: proportion of idleness
                               per trial, boxed by postnatal age band.

Use as its own module (import squircle.idleness as idl) or paste into
squircle/visualization.py. Trials are the tuples from st.load_trials():
name = t[0], kind = t[3], env = t[4], age = t[6], rewarded well = t[8],
trial number = t[10], df (cm positions) = t[15].
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import squircle.tools as st

try:
    from squircle.design import fs
except ImportError:
    fs = 30.0

AGE_BANDS = ((17, 20, 'Pre-weaning (P17-20)'),
             (21, 24, 'Peri-weaning (P21-24)'),
             (25, 32, 'Post-weaning (P25-32)'))   # matches visualization.py

DEFAULT_MOVING = 2.00    # cm/s, above this the rat is definitely moving
DEFAULT_NOT_MOVING = 1.75  # cm/s, below this the rat is definitely not moving
DEFAULT_MIN_IDLE_S = 1.0  # s, idle bouts shorter than this are absorbed
DEFAULT_DISP_WINDOW_S = 1.0  # s, sliding window for the dispersion method
DEFAULT_DISP_RADIUS_CM = 1.0  # cm, bounding-circle radius (set from noise!)


# ======================================================================
# Smoothing and speed  (Hen et al. 2004)
# ======================================================================

def savitzky_golay(x, window_s=0.25, polyorder=2, framerate=None):
    """
    Savitzky-Golay smoothing of a sampled signal (Hen et al. 2004 recommend
    smoothing the raw centroid path before deriving any quantity).

    Implemented directly in numpy (local least-squares polynomial fit), so no
    scipy dependency: identical results to scipy.signal.savgol_filter with
    mode='interp'. Window is rounded up to the next odd number of frames.
    """
    framerate = framerate if framerate is not None else fs
    x = np.asarray(x, dtype=float)
    half = int(window_s * framerate / 2)
    if half < 1:
        return x.copy()
    win = 2 * half + 1
    if win <= polyorder:
        return x.copy()

    # Vandermonde matrix of local time offsets, centered
    t = np.arange(-half, half + 1, dtype=float)
    A = np.vander(t, polyorder + 1, increasing=True)
    # SG projection: fitted value at offset 0 == constant term of the local
    # polynomial fit == first row of the pseudo-inverse of the Vandermonde
    inv = np.linalg.pinv(A)
    coef = inv[0]

    n = len(x)
    out = np.empty_like(x)
    # Interior: sliding dot product
    if n >= win:
        out[win // 2:-(win // 2) or None] = np.convolve(x, coef[::-1], mode='valid')
        # Edges: fit the same polynomial to the first/last `win` samples
        out[:win // 2] = np.vander(t[:win // 2], polyorder + 1, increasing=True) @ (inv @ x[:win])
        out[-(win // 2):] = np.vander(t[-(win // 2):], polyorder + 1, increasing=True) @ (inv @ x[-win:])
    else:
        # signal shorter than the window: fall back to a plain polynomial fit
        tt = np.linspace(-1, 1, n)
        B = np.vander(tt, polyorder + 1, increasing=True)
        out = B @ (np.linalg.pinv(B) @ x)
    return out


def smooth_trajectory(x, y, window_s=0.25, polyorder=2, framerate=None):
    """Savitzky-Golay smoothing of both coordinates (cm)."""
    xs = savitzky_golay(x, window_s, polyorder, framerate)
    ys = savitzky_golay(y, window_s, polyorder, framerate)
    return xs, ys


def instantaneous_speed(x, y, framerate=None):
    """
    Instantaneous speed in cm/s from smoothed positions (cm), by central
    differences (Delta t = 1/fps). Edges use one-sided differences.
    Sorry about the length-1 edge case: a 1-frame trial has no speed.
    """
    framerate = framerate if framerate is not None else fs
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return np.zeros(len(x))
    dx = np.diff(x)
    dy = np.diff(y)
    seg = np.hypot(dx, dy) * framerate
    return np.concatenate(([seg[0]], 0.5 * (seg[:-1] + seg[1:]), [seg[-1]]))


# ======================================================================
# Data-driven threshold  (Drai, Benjamini & Golani 2000)
# ======================================================================

def drai_benjamini_threshold(speeds, n_iter=200, floor=1e-3, random_state=0):
    """
    Threshold separating the stopping mode from the progression mode,
    estimated from the data as in Drai, Benjamini & Golani (2000).

    log(speed) is modelled as a mixture of two Gaussians ('stop' mode with
    the lower mean, 'progression' mode with the higher mean) fitted by EM.
    The threshold is the log-speed at which the two posterior probabilities
    are equal (the crossing of the mixture components).

    Returns (threshold_cm_s, mu, sd, weights) or None if the data are
    degenerate (no spread, or the modes do not separate).
    """
    v = np.asarray(speeds, dtype=float)
    v = v[np.isfinite(v) & (v > floor)]
    if len(v) < 30:
        return None
    lg = np.log(v)

    rng = np.random.default_rng(random_state)
    mu = np.percentile(lg, [25, 75]).astype(float)
    sd = np.array([max(lg.std() / 2, 0.05)] * 2)
    w = np.array([0.5, 0.5])

    for _ in range(n_iter):
        # E step: responsibilities
        p0 = w[0] * _normpdf(lg, mu[0], sd[0])
        p1 = w[1] * _normpdf(lg, mu[1], sd[1])
        tot = p0 + p1
        tot[tot < 1e-300] = 1e-300
        r0 = p0 / tot
        r1 = p1 / tot
        n0, n1 = r0.sum(), r1.sum()
        if n0 < 1 or n1 < 1:
            return None
        # M step
        new_mu = np.array([(r0 * lg).sum() / n0, (r1 * lg).sum() / n1])
        new_sd = np.array([np.sqrt(max((r0 * (lg - new_mu[0]) ** 2).sum() / n0, 1e-4)),
                           np.sqrt(max((r1 * (lg - new_mu[1]) ** 2).sum() / n1, 1e-4))])
        new_w = np.array([n0, n1]) / len(lg)
        # keep component 0 = slow mode
        if new_mu[0] > new_mu[1]:
            new_mu = new_mu[::-1]
            new_sd = new_sd[::-1]
            new_w = new_w[::-1]
        if (np.allclose(new_mu, mu, atol=1e-8) and
                np.allclose(new_sd, sd, atol=1e-8) and
                np.allclose(new_w, w, atol=1e-8)):
            mu, sd, w = new_mu, new_sd, new_w
            break
        mu, sd, w = new_mu, new_sd, new_w

    # Crossing point of the two weighted densities in log-space
    grid = np.linspace(lg.min(), lg.max(), 2000)
    d0 = w[0] * _normpdf(grid, mu[0], sd[0])
    d1 = w[1] * _normpdf(grid, mu[1], sd[1])
    diff = d0 - d1
    sign_change = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(sign_change) == 0:
        return None
    # crossing closest to the midpoint between the modes
    mid = 0.5 * (mu[0] + mu[1])
    k = sign_change[np.argmin(np.abs(grid[sign_change] - mid))]
    thr = float(0.5 * (grid[k] + grid[k + 1]))
    return float(np.exp(thr)), mu, sd, w


def _normpdf(x, mu, sd):
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))


def _idle_bouts(frames_idle, framerate):
    """Contiguous runs of True as (start_frame, end_frame_inclusive)."""
    frames_idle = np.asarray(frames_idle, dtype=bool)
    if frames_idle.sum() == 0:
        return []
    starts = np.where(frames_idle & ~np.concatenate(([False], frames_idle[:-1])))[0]
    ends = np.where(frames_idle & ~np.concatenate((frames_idle[1:], [False])))[0]
    return list(zip(starts.tolist(), ends.tolist()))


# ======================================================================
# Method A - speed thresholding with hysteresis (Hen 2004, Noldus dual cutoff)
# ======================================================================

def classify_movement_hysteresis(speed, moving=DEFAULT_MOVING,
                                  not_moving=DEFAULT_NOT_MOVING,
                                  framerate=None):
    """
    Binary not-moving/moving state per frame with a dual (hysteresis)
    threshold: state switches to not-moving under `not_moving` and back to
    moving above `moving` (cm/s). Between the two cutoffs the previous state
    persists. Returns a boolean array, True = idle.
    """
    v = np.asarray(speed, dtype=float)
    idle = np.zeros(len(v), dtype=bool)
    state = False  # start assuming moving (rats are handled at trial onset)
    for i in range(len(v)):
        if v[i] <= not_moving:
            state = True
        elif v[i] >= moving:
            state = False
        idle[i] = state
    return idle


def enforce_min_idle_duration(idle, min_idle_s=DEFAULT_MIN_IDLE_S,
                             framerate=None):
    """
    Drop idle bouts shorter than `min_idle_s` seconds (their frames become
    moving), as in EthoVision's minimum-bout setting. The dual threshold
    already suppresses most flicker; this removes short pauses like a
    one-frame head turn.
    """
    framerate = framerate if framerate is not None else fs
    idle = np.asarray(idle, dtype=bool).copy()
    min_len = int(round(min_idle_s * framerate))
    for s, e in _idle_bouts(idle, framerate):
        if (e - s + 1) < min_len:
            idle[s:e + 1] = False
    return idle


# ======================================================================
# Method B - spatial bounding circle / dispersion (stationary bouts)
# ======================================================================

def classify_dispersion(x, y, window_s=DEFAULT_DISP_WINDOW_S,
                        radius_cm=DEFAULT_DISP_RADIUS_CM, framerate=None):
    """
    Stationary (in-place) frames from positional dispersion, not speed:
    for each sliding window of `window_s` seconds, compute the largest
    distance of any position in the window to the window's mean position
    (the radius of the bounding circle). Windows whose radius is <=
    `radius_cm` are marked; every frame in a marked window is stationary.

    Overlapping and touching marked windows merge into one bout, so the
    window length doubles as the minimum bout duration. Returns a boolean
    array, True = stationary (includes grooming/head turns in place).
    """
    framerate = framerate if framerate is not None else fs
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    win = int(round(window_s * framerate))
    if n < win or win < 2:
        return np.zeros(n, dtype=bool)
    win = max(win, 2)
    if win % 2 == 0:
        win += 1

    sq = x * x + y * y
    csum = np.concatenate(([0.0], np.cumsum(sq)))
    sx = np.concatenate(([0.0], np.cumsum(x)))
    sy = np.concatenate(([0.0], np.cumsum(y)))
    half = win // 2
    starts = np.arange(0, n - win + 1)
    counts = win
    mean_sq = (csum[starts + win] - csum[starts]) / counts
    mean_x = (sx[starts + win] - sx[starts]) / counts
    mean_y = (sy[starts + win] - sy[starts]) / counts
    # E[d^2] - |mean|^2 = variance; bounding radius <= sqrt(max d^2)
    # Exact max distance needs a loop, so use the 2-sigma dispersion bound
    # computed exactly per window below (loop is short: n windows only).
    radii = np.empty(len(starts))
    for j, s0 in enumerate(starts):
        dx = x[s0:s0 + win] - mean_x[j]
        dy = y[s0:s0 + win] - mean_y[j]
        radii[j] = np.sqrt(np.max(dx * dx + dy * dy))

    marked = radii <= radius_cm
    stationary = np.zeros(n, dtype=bool)
    # every frame in a marked window is stationary
    for j, s0 in enumerate(starts):
        if marked[j]:
            stationary[s0:s0 + win] = True
    return stationary


# ======================================================================
# Per-trial measurement
# ======================================================================

def _clean_xy(df):
    """Finite (x, y) arrays in cm from a trial DataFrame, NaNs dropped."""
    if df is None or getattr(df, 'empty', True) or len(df) < 2:
        return None, None
    x = df['x'].to_numpy(dtype=float)
    y = df['y'].to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return None, None
    return x[ok], y[ok]


def idle_proportion(trial, window_s=0.25, polyorder=2,
                    moving=DEFAULT_MOVING, not_moving=DEFAULT_NOT_MOVING,
                    min_idle_s=DEFAULT_MIN_IDLE_S,
                    disp_window_s=DEFAULT_DISP_WINDOW_S,
                    disp_radius_cm=DEFAULT_DISP_RADIUS_CM,
                    threshold='fixed', framerate=None):
    """
    Proportion of each trial the rat is idle.

    Parameters
    ----------
    trial : tuple from st.load_trials() (df, in cm, at t[15])
    window_s, polyorder : Savitzky-Golay smoothing parameters (Hen et al. 2004)
    moving, not_moving : dual speed cutoffs in cm/s (Noldus hysteresis)
    min_idle_s : minimum idle-bout duration in s (shorter bouts drop out)
    threshold : 'fixed' uses the given cutoffs; 'drai' derives the cutoff
        from this trial's own log-speed distribution (Drai et al. 2000) and
        uses +-7% of it as the hysteresis band around the modal boundary.

    Returns dict with keys
        idle_prop   - speed-method idle proportion
        stat_prop    - dispersion-method stationary proportion
        quiet_prop   - idle AND stationary (quiet immobility; excludes
                       in-place activity like grooming, per the A/B split)
        thr_cm_s     - the threshold actually used (cm/s)
        idle_frames / stationary_frames : boolean per-frame arrays
        speed        - smoothed speed (cm/s)
        n_bouts      - number of idle bouts after the minimum-duration rule
    """
    x, y = _clean_xy(trial[15])
    if x is None:
        return None

    xs, ys = smooth_trajectory(x, y, window_s, polyorder, framerate)
    v = instantaneous_speed(xs, ys, framerate)

    thr_lo, thr_hi = not_moving, moving
    if threshold == 'drai':
        fit = drai_benjamini_threshold(v)
        if fit is None:
            # not enough structure in this trial - keep the fixed cutoffs
            thr_used = None
        else:
            thr_used = fit[0]
            thr_lo, thr_hi = 0.93 * thr_used, 1.07 * thr_used
    else:
        thr_used = 0.5 * (thr_lo + thr_hi)

    idle = classify_movement_hysteresis(v, thr_hi, thr_lo, framerate)
    idle = enforce_min_idle_duration(idle, min_idle_s, framerate)
    stationary = classify_dispersion(xs, ys, disp_window_s, disp_radius_cm,
                                    framerate)

    n = len(x)
    return {
        'name': trial[0],
        'kind': trial[3],
        'env': trial[4],
        'date': trial[5],
        'age': trial[6],
        'trial_number': trial[10],
        'idle_prop': float(idle.sum() / n),
        'stat_prop': float(stationary.sum() / n),
        'quiet_prop': float((idle & stationary).sum() / n),
        'thr_cm_s': thr_used,
        'n_frames': n,
        'n_bouts': len(_idle_bouts(idle, framerate)),
        'speed': v,
        'xy': (x, y),
        'idle_frames': idle,
        'stationary_frames': stationary,
    }


def load_litters(working_folder, litters=(9, 10, 11), **filters):
    """Load trials from litter_<n>_processed_cm.json, as in the other modules."""
    trials = []
    for n in litters:
        path = os.path.join(working_folder, f'litter_{n}_processed_cm.json')
        if not os.path.exists(path):
            continue
        selected, _ = st.load_trials(path, **filters)
        trials.extend(selected)
    return trials


def _as_trials(source, litters=(9, 10, 11)):
    """Accept a working-folder path or an already-loaded trial list."""
    if isinstance(source, (str, os.PathLike)):
        return load_litters(source, litters)
    return source


def _band_of(age, age_bands):
    for lo, hi, lbl in age_bands:
        if lo <= age <= hi:
            return lbl
    return None


def idle_proportion_df(trials, **kwargs):
    """
    idle_proportion over a list of trials -> tidy DataFrame (one row per
    trial: name, age, band, session kind, idle_prop, stat_prop, quiet_prop,
    threshold used, number of idle bouts).
    """
    rows = []
    for t in trials:
        r = idle_proportion(t, **kwargs)
        if r is None:
            continue
        rows.append({
            'Name': r['name'],
            'Age': r['age'],
            'Band': _band_of(r['age'], AGE_BANDS),
            'Kind': r['kind'],
            'Trial': r['trial_number'],
            'idle_prop': r['idle_prop'],
            'stat_prop': r['stat_prop'],
            'quiet_prop': r['quiet_prop'],
            'thr_cm_s': r['thr_cm_s'],
            'n_bouts': r['n_bouts'],
        })
    return pd.DataFrame(rows)


# ======================================================================
# Figures
# ======================================================================

def plot_speed_threshold(trials, bins=80, range=None, figsize=(7, 4.5),
                         ax=None, title=None, **kwargs):
    """
    Drai & Benjamini (2000) style figure: pooled log-speed histogram with the
    fitted two-component mixture and the data-derived mode boundary. This is
    the evidence for "where should my threshold be".
    """
    speeds = []
    for t in trials:
        x, y = _clean_xy(t[15])
        if x is None:
            continue
        xs, ys = smooth_trajectory(x, y)
        speeds.append(instantaneous_speed(xs, ys))
    v = np.concatenate(speeds) if speeds else np.array([])
    if len(v) == 0:
        print('No tracking data to plot.')
        return None

    fit = drai_benjamini_threshold(v)
    lg = np.log(v)

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=figsize)
    ax.hist(lg, bins=bins, density=True, color='0.75', edgecolor='white')

    if fit is not None:
        thr, mu, sd, w = fit
        grid = np.linspace(lg.min(), lg.max(), 500)
        ax.plot(grid, w[0] * _normpdf(grid, mu[0], sd[0]), '--',
                color='tab:orange', label='stop mode')
        ax.plot(grid, w[1] * _normpdf(grid, mu[1], sd[1]), '--',
                color='tab:blue', label='progression mode')
        ax.axvline(np.log(thr), color='crimson', lw=2,
                   label=f'threshold = {thr:.2f} cm/s')
        ax.legend(frameon=False)
    ax.set_xlabel('log(speed)   [cm/s]')
    ax.set_ylabel('density')
    ax.set_title(title or 'Log-speed distribution with fitted modes '
                          '(Drai, Benjamini & Golani, 2000)')
    sns.despine()
    return ax


def plot_trial_idleness(trial, figsize=(13, 6), title=None, **kwargs):
    """
    Single-trial diagnostic, top: smoothed speed with the dual thresholds and
    idle spans shaded; bottom: trajectory colored by state (speed-idle red,
    stationary-but-not-idle = in-place activity orange, moving grey).
    """
    r = idle_proportion(trial, **kwargs)
    if r is None:
        print(f"No usable trajectory for {trial[0]} trial {trial[10]}.")
        return None

    x, y = r['xy']
    idle = r['idle_frames']
    stat = r['stationary_frames']
    v = r['speed']
    t = np.arange(len(v)) / (fs if fs else 30.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize,
                                  gridspec_kw={'width_ratios': [1.4, 1]})

    ax1.plot(t, v, color='0.4', lw=0.8)
    ax1.axhline(DEFAULT_NOT_MOVING, color='tab:blue', ls=':',
                label=f"not moving <= {DEFAULT_NOT_MOVING} cm/s")
    ax1.axhline(DEFAULT_MOVING, color='tab:green', ls='--',
                label=f"moving >= {DEFAULT_MOVING} cm/s")
    for s, e in _idle_bouts(idle, fs):
        ax1.axvspan(t[s], t[min(e, len(t) - 1)], color='crimson', alpha=0.25)
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('speed (cm/s)')
    ax1.set_ylim(0, np.nanpercentile(v, 99.5))
    ax1.legend(frameon=False, fontsize=8)

    in_place = stat & ~idle
    ax2.scatter(x[~idle & ~stat], y[~idle & ~stat], s=2, color='0.6',
                label='moving')
    ax2.scatter(x[in_place], y[in_place], s=2, color='tab:orange',
                label='in-place activity')
    ax2.scatter(x[idle], y[idle], s=2, color='crimson', label='idle')
    ax2.set_aspect('equal')
    ax2.legend(frameon=False, fontsize=8, markerscale=3)

    fig.suptitle(title or (f"{r['name']}  P{r['age']}  trial {r['trial_number']}"
                           f"   idle {100 * r['idle_prop']:.0f}%   "
                           f"stationary {100 * r['stat_prop']:.0f}%   "
                           f"quiet {100 * r['quiet_prop']:.0f}%"))
    sns.despine()
    return fig


def plot_idle_proportion_by_band(trials, value='quiet_prop',
                                 age_bands=AGE_BANDS, figsize=(9, 6),
                                 palette=None, ax=None, title=None,
                                 show_points=True, **kwargs):
    """
    THE summary figure: proportion of idleness per trial, boxed by postnatal
    age band (the pipeline's standard bands by default).

    value : 'idle_prop' (speed method), 'stat_prop' (dispersion method,
            includes in-place activity) or 'quiet_prop' (idle AND
            stationary - true quiet immobility). Defaults to 'quiet_prop'.
    """
    D = idle_proportion_df(_as_trials(trials), **kwargs)
    if D.empty:
        print('No usable trials.')
        return None
    D = D.dropna(subset=['Band']).copy()

    labels = [b[2] for b in age_bands]
    D['Band'] = pd.Categorical(D['Band'], categories=labels, ordered=True)
    D = D.sort_values('Band')

    _labels = {
        'idle_prop': 'proportion of trial idle (speed method)',
        'stat_prop': 'proportion of trial stationary (dispersion method)',
        'quiet_prop': 'proportion of trial quiet immobile',
    }
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(data=D, x='Band', y=value, hue='Band', ax=ax,
                palette=palette or 'Set2', legend=False, showfliers=False)
    if show_points:
        sns.stripplot(data=D, x='Band', y=value, ax=ax, color='0.25',
                      size=3, alpha=0.6, jitter=0.15)
    if title is None:
        title = 'Proportion of idleness per trial, by age'
    ax.set_xlabel('')
    ax.set_ylabel(_labels.get(value, value))
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title)
    sns.despine()
    return ax

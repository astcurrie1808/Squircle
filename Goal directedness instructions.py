# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 13:35:51 2026

@author: astcu
"""
# PLACE THE FOLLOWING IN SQUIRCLDANALYSIS.PY AND OPEN THE FOLLOWING FILES:
# GOAL_DIRECTEDNESS_ARENA AND GOAL.DIRECTEDNESS.PY


# %% Goal-directedness — imports + reload  (place near the header block)'
# =====================================================================
# Cells to paste into SquircleAnalysis.py
# Requires squircle/goal_directedness.py and squircle/goal_directedness_arena.py
# on the path (same package folder as tools.py / visualization.py).
# Swap converted_path to litter_11_processed_cm.json for the new paradigm.
# =====================================================================
import squircle.goal_directedness as gd
import squircle.goal_directedness_arena as gda

importlib.reload(gd)          # reloads AND keeps gd bound to the fresh module
importlib.reload(gda)         # (avoids the phantom AttributeError after edits)


# %% -- arena illustration (distance to target, 2 snapshots)
converted_path = os.path.join(working_folder, "litter_11_processed_cm.json")

train, _ = st.load_trials(converted_path, name='Jinkx',
                          kind='Training', date='29_6_2026')

gda.plot_onset_distance_arena(train[0], window=5)   # one illustrative trial
gda.plot_onset_distance_grid(train, window=5)       # whole session at a glance

# %% -- paired distance summary (target vs unfilled wells)
converted_path = os.path.join(working_folder, "litter_11_processed_cm.json")

tr, pr = st.load_trials(converted_path, name='Jinkx', kind='Training',
                        date='29_6_2026', include_probes=True, same_day=True)

gda.plot_onset_distance_paired(tr + pr, window=5)
gda.plot_onset_distance_paired(tr + pr, window=10)  # sensitivity: 5 cm vs 10 cm


# %% -- angular version across trials / ages (population)
merged = os.path.join(working_folder, 'combined_litters_9_10_11_.json')
train, probe = st.load_trials(merged, kind='Training', include_probes=True)
trials = train + probe

gd.plot_goal_directedness(trials, metric='goal_rank_pct',     x='TrialNumber')
gd.plot_goal_directedness(trials, metric='heading_error_deg', x='Age')
gd.plot_goal_directedness(trials, metric='directedness',      x='Age')


# %%-- export tidy dataframe for stats (both proxies in one frame)
merged = os.path.join(working_folder, 'combined_litters_9_10_11_.json')
train, probe = st.load_trials(merged, kind='Training', include_probes=True)
trials = train + probe

gd_df = gd.build_goal_directedness_df(trials, window=5, window_unit='cm')

# columns: Subject, Sex, SessionType, Env, Age, TrialNumber, Baited,
#   heading_error_deg, directedness, goal_rank_pct, heading_z,
#   approach_t0, approach_tw, approach_delta, frames_used
gd_df.to_csv(os.path.join(working_folder, 'goal_directedness.csv'), index=False)

# quick sanity read: the two testable outputs by session type
print(gd_df.groupby('SessionType')[['goal_rank_pct', 'approach_delta']].mean())


# =====================================================================
# PATH-PROGRESS VERSION  (the "20% of the way in" paradigm)
# needs squircle/goal_directedness_progress.py
# =====================================================================
import squircle.goal_directedness_progress as gdp
importlib.reload(gdp)

# %% -- single trial: where is the animal at its 20% mark?
converted_path = os.path.join(working_folder, "litter_11_processed_cm.json")
train, _ = st.load_trials(converted_path, name='Jinkx',
                          kind='Training', date='29_6_2026')

gdp.plot_progress_arena(train[0], frac=0.2)                    # 20% of total path walked
gdp.plot_progress_arena(train[0], frac=0.2, frac_of='start_distance')    # sensitivity variant

# %% -- THE key figure: approach profile (distance to target vs % of path)
# Same calling convention as the sv.* plots: working_folder first, litters=(...)
gdp.plot_approach_profile(working_folder, litters=(9, 10, 11), hue='Env')
gdp.plot_approach_profile(working_folder, litters=(9, 10, 11), hue='SessionType')

# %% -- MAIN FIGURE: both x-axes at once
# Columns = Training | Probe, rows = dD and g_rank_pct, one line per age band.
# The two rows disagreeing IS the result - see the module docstring.
gdp.plot_progress_panels(working_folder, litters=(9, 10, 11))              # x = trial number
gdp.plot_progress_panels(working_folder, litters=(9, 10, 11), x='Age')     # x = age

# %% -- CONCEPT 1: Within-session — dD across trials, split by age band
# NOTE this is a NULL result: slopes are +0.16 / +0.04 / +0.11 cm per trial for
# Pre / Peri / Post-weaning, i.e. drifting slightly AWAY from the goal, and path
# efficiency is flat across trials too (rho=+0.009, p=0.68). Present it as
# "no within-session change", not as calibration. It is a useful control figure:
# it rules out order and satiation effects.
gdp.plot_progress_delta(working_folder, litters=(9, 10, 11),
                        metric='dD', x='TrialNumber', hue='Band',
                        title='Within-session: no change in goal-directedness')

# Probe only, if you want the 1-10 trial axis (Training runs 1-20)
probes = gdp.load_litters(working_folder, litters=(9, 10, 11), kind='Probe')
gdp.plot_progress_delta(probes, metric='dD', x='TrialNumber', hue='Band',
                        title='Within-session, Probe trials only')

# %% -- CONCEPT 2: Developmental trajectory — dD across postnatal age
# This one IS significant: Training rho=-0.090 p=7e-5, Probe rho=-0.100 p=0.002,
# 13/15 animals improve (p=0.003). But see the control immediately below before
# calling it "spatial planning" - total path length also collapses with age.
gdp.plot_progress_delta(working_folder, litters=(9, 10, 11),
                        metric='dD', x='Age', hue='SessionType',
                        title='Developmental trajectory: dD by postnatal age')

# Circle vs Square is the contrast this project is named after - worth a look:
gdp.plot_progress_delta(working_folder, litters=(9, 10, 11),
                        metric='dD', x='Age', hue='Env')

# THE CONTROL for concept 2: same axes, chance-referenced metric. Flat (p=0.93
# per animal) => the age effect in dD is not target-specific. Show both or neither.
gdp.plot_progress_delta(working_folder, litters=(9, 10, 11),
                        metric='g_rank_pct', x='Age', hue='SessionType',
                        title='Control: target-selectivity does not change with age')

# %% -- single summary panels (other combinations)
# g_rank_pct is the chance-referenced headline (chance = 0.5); dD is the raw cm version.
gdp.plot_progress_delta(working_folder, litters=(9, 10, 11), metric='g_rank_pct', x='Age')
gdp.plot_progress_delta(working_folder, litters=(9, 10, 11), metric='g_rank_pct', x='TrialNumber')

# %% -- diagnostic: how much of this is efficiency in disguise?
# Raw g is bounded by frac*efficiency, so it rides the dashed ceiling.
# Compare the Spearman rho printed in each title.
gdp.plot_progress_vs_efficiency(working_folder, litters=(9, 10, 11), metric='g')
gdp.plot_progress_vs_efficiency(working_folder, litters=(9, 10, 11), metric='g_rank_pct')

# Sensitivity check: does the result depend on where the snapshot is taken?
gdp.plot_progress_delta(working_folder, litters=(9, 10, 11),
                        metric='g_rank_pct', frac_of='start_distance', x='Age')

# %% -- sensitivity to the choice of fraction
for f in (0.1, 0.2, 0.3):
    gdp.plot_progress_delta(working_folder, litters=(9, 10, 11),
                            metric='g_rank_pct', frac=f, x='Age')

# %% -- Training and Probe mean different things here: in Training the path ENDS at
# the well (L is outcome-determined), in unbaited Probes it runs to a fixed duration.
# plot_progress_panels already splits them; this is the single-panel version.
trials = gdp.load_litters(working_folder, litters=(9, 10, 11))
for k in ('Training', 'Probe'):
    gdp.plot_progress_delta([t for t in trials if t[3] == k],
                            metric='g_rank_pct', x='Age', hue='Env',
                            title=f'{k}: goal-directedness at the 20% mark')

# %% -- saving a figure (these return a Figure; nothing is written automatically)
figdir = os.path.join(working_folder, 'goal_directedness_figures')
os.makedirs(figdir, exist_ok=True)
fig = gdp.plot_progress_panels(working_folder, litters=(9, 10, 11))
fig.savefig(os.path.join(figdir, 'panels_by_trialnumber.png'),
            dpi=300, bbox_inches='tight')

# %% -- export tidy frame for stats
prog_df = gdp.build_progress_df(working_folder, litters=(9, 10, 11), frac=0.2)
# columns: Subject, Sex, SessionType, Env, Date, Age, TrialNumber, Baited,
#   d0, d_at, delta_cm, g, g_rank_pct, g_z, g_null_mean, total_path,
#   efficiency, s_used
prog_df.to_csv(os.path.join(working_folder, 'goal_directedness_progress.csv'),
               index=False)

print(prog_df.groupby(['Env', 'SessionType'])[['g', 'g_rank_pct']].mean())

# Is the group above chance at all? (chance = 0.5)
from scipy import stats
per_animal = prog_df.groupby('Subject')['g_rank_pct'].mean()
print(stats.wilcoxon(per_animal - 0.5))   # animal, not trial, is the unit of analysis

# Per-animal age slopes - animal is the unit of analysis, and this needs only scipy.
import numpy as np
T = prog_df[prog_df.SessionType == 'Training']
for c in ['dD', 'g_rank_pct']:
    pa = T.groupby(['Subject', 'Age'])[c].mean().reset_index()
    sl = (pa.groupby('Subject')[['Age', c]]
            .apply(lambda g: np.polyfit(g.Age, g[c], 1)[0] if len(g) > 3 else np.nan)
            .dropna())
    print(f'{c:12s} n={len(sl)} animals, median slope {sl.median():+.4f}/day, '
          f'p={stats.wilcoxon(sl).pvalue:.3g}, {(sl < 0).sum()}/{len(sl)} negative')

# OPTIONAL mixed model - statsmodels is NOT installed in the miniconda env, so this
# cell will fail until you `pip install statsmodels`. The per-animal test above is
# the more defensible analysis anyway (it does not assume trials are independent).
# d0 covariate: entry point (27 of them) sets how much there is to gain.
# efficiency covariate: partials out the ceiling effect of frac_of='total_path'.
import statsmodels.formula.api as smf
m = smf.mixedlm("g_rank_pct ~ TrialNumber + Env + Age + d0 + efficiency", prog_df,
                groups=prog_df["Subject"]).fit()
print(m.summary())
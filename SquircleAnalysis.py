# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 16:56:17 2026

@author: JoseTeixeira

"""
#%% Pipeline
import sys
sys.path.append(r"C:\Users\astcu\OneDrive\Documents\2025-2026\Neurobiologie\Internship")

import os
import importlib
import squircle.visualization as sv
import squircle.tools as st
importlib.reload(sv)
importlib.reload(st)
import squircle.goal_directedness_progress as gdp
importlib.reload(gdp)


# Directory where to save outputs
base_folder = 'W:\\ephys\\Jose\\Squircle Project\\Data\\Litter11'
working_folder = r'C:\Users\astcu\OneDrive\Documents\2025-2026\Neurobiologie\Internship'

# Paths
json_path      = st.create_json(base_folder, working_folder, verbose=True)
processed_path = st.preprocess_json_positions(json_path)
converted_path = st.convert_positions_to_cm(processed_path)

# Check sessions, trials, and that the Completed status came through
import json
from collections import Counter
with open(converted_path) as f:
    data = json.load(f)
for s in data:
    n_trials = sum(len(sess["Trials"]) for sess in s["Sessions"])
    print(f"{s['Name']:8s} sessions={len(s['Sessions']):3d} trials={n_trials}")
status = Counter(t.get('Completed') for s in data
                 for sess in s['Sessions'] for t in sess['Trials'])
print("Litter 11 status:", dict(status))

#%% Merge data paths
# Merge NP
merged_data_path_new_paradigm = st.merge_litters([
    os.path.join(working_folder, f'litter_{n}_processed_cm.json')
    for n in (9, 10, 11)
])
merged_data_new = merged_data_path_new_paradigm   # use returned path, not hand-typed

# Code for merging OP + NP
merged_5_to_11 = st.merge_litters([
    os.path.join(working_folder, f'litter_{n}_processed_cm.json')
    for n in range(5, 12)
])
print(merged_5_to_11)

# Check sessions and trials per animal
import json
with open(converted_path) as f:
    data = json.load(f)
for s in data:
    n_trials = sum(len(sess["Trials"]) for sess in s["Sessions"])
    print(f"{s['Name']:8s} sessions={len(s['Sessions']):3d} trials={n_trials}")
    
#%% Figures

#%% -- Learning — development across age
# Path efficiency across trial number (within-session; age-at-test banding; pooled median + IQR)
sv.plot_efficiency_across_trials(working_folder, litters=(9, 10, 11), band_by='session')

# Same, but first session-day only (naive day; band by entry age)
sv.plot_efficiency_across_trials(working_folder, litters=(9, 10, 11), band_by='entry', days=[1])

# Learning curves overlaid by first-trial baiting (solid = unbaited, dashed = baited)
sv.plot_efficiency_baited_overlay(working_folder, litters=(9, 10, 11), band_by='session')
# ...pooled across ages for a clean 2-line comparison:
sv.plot_efficiency_baited_overlay(working_folder, litters=(9, 10, 11), split_bands=False)

# Path efficiency vs postnatal age (x = P-day; pooled median + IQR) -- no banding ambiguity
sv.plot_efficiency_by_age(working_folder, litters=(9, 10, 11))

# Plot first trial versus trial 2-10
sv.plot_naive_vs_early_first_session(working_folder,litters=(9, 10, 11), ylim=(1, 20))

# Coupled lineplot T1 vs P1 with average (per litter)
sv.plot_daily_t1_vs_p1_with_avg(working_folder,litter=9)

# Efficiency per subject across ages (litter-based grouping)
sv.plot_efficiency_per_subject_across_ages(working_folder)

#%% -- Retention

# Lineplot Training 3-4-5 vs probe, with failure-rate overlay
sv.plot_retention_blocks(working_folder, litters=(9, 10, 11), show_failures=True)

# Boxplot P1 - T1 over all sessions, Pre -peri vs post-weaning
sv.plot_allsessions_t1_vs_probe1_by_band(working_folder,litters=(9, 10, 11))

# Boxplot P1-T1 (first day)
sv.plot_first_day_t1_vs_probe1_by_band(working_folder,litters=(9, 10, 11))

# Boxplot T1 P1 per band, unbaited probes only (P1 unbaited) -- cleaner retention
sv.plot_raw_t1_p1_by_band(working_folder, litters=(9, 10, 11),
                          require_p1_unbaited=True, color_by_animal=True)

# Boxplot T1 P1 per band, first paired day only
sv.plot_raw_t1_p1_by_band(working_folder, litters=(9, 10, 11), days=[1])


#%% -- Baited vs Unbaited

# Baited vs unbaited, pooled (Litter 11) 
sv.plot_baited_vs_unbaited_boxplot(working_folder, litter=9)

# Baited vs unbaited across ages (Litter 11)
sv.plot_baited_vs_unbaited_across_ages(working_folder, litter=11)

# Baited vs unbaited gap across age (new-paradigm litters)
sv.plot_baited_gap(working_folder, litters=(9, 10, 11))

# Trials run per age band, baited/unbaited with failed portion shaded (all sessions, litters 9-11)
sv.plot_failure_rate_train_vs_probe_by_band(working_folder, litters=(9, 10, 11), days=[1], show_ci=False)

#%% -- Summary Boxplot
sv.plot_all_animals_boxplot(working_folder)

#%% 3* 7 trajectory grid (per age band)
sv.plot_trajectory_grids(
    working_folder, litters=(9, 10, 11),
    band_representatives={'Pre-weaning':  'Marimba',
                          'Peri-weaning': 'Trumpet',
                          'Post-weaning': 'Cello'})

#%% Speed of learning

# Main figure: Training | Probe columns, dD and g_rank_pct rows, age bands in colour
gdp.plot_progress_panels(working_folder, litters=(9, 10, 11))
gdp.plot_progress_panels(working_folder, litters=(9, 10, 11), x='Age')

# Single panels
gdp.plot_progress_delta(working_folder, litters=(9, 10, 11), metric='dD', x='Age')
gdp.plot_progress_delta(working_folder, litters=(9, 10, 11), metric='g_rank_pct', x='Age')

# Within-session, sessions whose FIRST trial was UNBAITED only (memory-driven runs).
# first_trial_baited follows sv.plot_efficiency_across_trials: 0 = trial 1 unbaited,
# 1 = baited, None = all. The whole session is kept; only its inclusion depends on
# trial 1. Note the age-band colours here are confounded - only 2 animals span all
# three bands and pre-weaning rests on 5 - so read the flatness, not the offsets.
gdp.plot_progress_delta(working_folder, litters=(9, 10, 11),
                        metric='dD', x='TrialNumber', hue='Band',
                        first_trial_baited=0, frac=0.1,
                        title='Within-session (first trial unbaited): no change in goal-directedness')

# ...and the baited-first comparison, if you want the contrast
gdp.plot_progress_delta(working_folder, litters=(9, 10, 11),
                        metric='dD', x='TrialNumber', hue='Band',
                        first_trial_baited=1, frac=0.1,
                        title='Within-session (first trial baited)')

# Approach profile
gdp.plot_approach_profile(working_folder, litters=(9, 10, 11), hue='Env')

# Efficiency diagnostic
gdp.plot_progress_vs_efficiency(working_folder, litters=(9, 10, 11), metric='g')

#%% -- Day-1 within-session learning speed  (three parallel views)
# Speed of learning WITHIN the naive day, per age band, three ways to choose between.
# Efficiency is real/ideal (lower = more direct); slope negative / block drop
# positive = learning. N per band = 5/6/4, so between-band is descriptive.
# Defaults: day 1, trials 1-10, models on log(efficiency). Probe day-1 is shown
# but is post-training (not naive) — the naivety story is strongest for Training.

# All three at once + a combined per-band summary table (recommended entry point):
res = gdp.compare_day1_learning_speed(working_folder, litters=(9, 10, 11))
# res is a dict: res['summary'] (table), res['slopes'], res['block'],
# res['mixedlm'] (per-session fit + coef table), res['data'], res['figs'].

# ...or each view on its own:
# View 1 — per-animal learning-rate slopes (Theil-Sen robust; vs Trial and vs log Trial):
gdp.plot_day1_slopes(working_folder, litters=(9, 10, 11))
# View 2 — early-vs-late block, median(T1-3) - median(T8-10):
gdp.plot_day1_block(working_folder, litters=(9, 10, 11))
# View 3 — mixed model, Band x trial interaction = between-band rate difference:
gdp.plot_day1_mixedlm(working_folder, litters=(9, 10, 11), session='Training')

# Sensitivity: full 20-trial training session instead of the first 10
# gdp.compare_day1_learning_speed(working_folder, litters=(9, 10, 11), trials=range(1, 21))

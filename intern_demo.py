# -*- coding: utf-8 -*-
"""
Demo script for the other interns.
---------------------------------------------------------------------------
Companion to INTERN_TEACHING_GUIDE.md.

This file shows, end-to-end, how to go from a JSON file full of behavioural
data to a subset of that data and then a plot -- using only the two modules
the supervisor sent you:

    squircle/tools.py            -> st   (reading + filtering the JSON)
    squircle/visualization.py    -> sv   (making figures)

It is meant to be run cell-by-cell in Spyder (put the cursor in a cell and
press Ctrl+Enter). Read the comments above each cell: they say what the cell
does and what to change to see a different result.

===========================================================================
IMPORTANT -- folder setup (read this before running anything)
===========================================================================
The two modules above import each other as a *package*:

    import squircle.tools as st
    import squircle.visualization as sv

That means the .py files must live in a folder called `squircle/`, and the
PARENT of that folder must be on Python's path. On the lab laptop this is
already arranged, but if `import squircle.tools` fails, fix the path in
cell 0 below so it points at the parent of your `squircle/` folder.

Example folder layout:

    my_internship/                 <-- point sys.path here
        squircle/
            __init__.py            <-- can be an empty file
            tools.py
            visualization.py
            design.py              <-- also needed (supplied with the package)
        intern_demo.py             <-- THIS file, run it from here

If there is no __init__.py, just create an empty one inside squircle/.

The JSON data file (`*_processed_cm.json`) can live anywhere -- you point
`data_path` at it in cell 1.
"""

# %% Cell 0 -- imports + path setup (run this FIRST)
# ============================================================================
# Put the cursor here and press Ctrl+Enter. This makes `st` and `sv`
# available in the Spyder console for all later cells.
# ============================================================================
import sys
import os
import json
import importlib

# >>> EDIT THIS LINE if `import squircle...` fails <<<
# It must be the parent folder of your `squircle/` folder (see docstring).
sys.path.append(r"C:\Users\astcu\OneDrive\Documents\2025-2026\Neurobiologie\Internship")

import squircle.tools as st
import squircle.visualization as sv

# Re-run these two lines whenever you edit tools.py / visualization.py so
# Spyder picks up the change. (Same habit shown in the teaching guide.)
importlib.reload(st)
importlib.reload(sv)

print("tools and visualization imported OK.")


# %% Cell 1 -- point at the data
# ============================================================================
# `data_path` is the converted JSON file produced by the pipeline
# (create_json -> preprocess_json_positions -> convert_positions_to_cm).
# You only need the final *_processed_cm.json file, not the raw video.
# Change the path to wherever your file lives.
# ============================================================================
data_path = os.path.join(
    os.path.dirname(__file__),              # same folder as this script, or
    "litter_11_processed_cm.json"           # <-- edit the filename if needed
) if "__file__" in dir() else r"litter_11_processed_cm.json"

# Quick sanity check: open the JSON and print one line per animal, showing
# how many sessions and how many trials each has. This is the same "check
# the data loaded" block from SquircleAnalysis.py -- it just confirms the
# file is real before we filter it.
with open(data_path) as f:
    raw_data = json.load(f)

for animal in raw_data:
    n_sessions = len(animal["Sessions"])
    n_trials = sum(len(sess["Trials"]) for sess in animal["Sessions"])
    print(f"{animal['Name']:8s}  sessions={n_sessions:3d}  trials={n_trials}")

# Expected output: one line per animal with sensible numbers (not 0).
# If you see 0 trials everywhere, data_path is wrong or the file is empty.


# %% Cell 2 -- read + filter the data with st.load_trials()
# ============================================================================
# load_trials() is the workhorse. It reads the JSON and returns ONLY the
# trials matching the filters you give it. Two things to notice:
#
#   1. It always returns TWO things:  (selected_trials, probe_trials)
#      - selected_trials = the trials matching your filters
#      - probe_trials     = probe trials (empty unless include_probes=True)
#      We unpack them into `train, probe` in one line.
#
#   2. Every filter is optional. Set the ones you care about; leave the
#      rest as None and they are ignored.
#
# Here we ask for:  subject 'Jinkx',  Training sessions,  on one date.
# Try changing `name=` to a different animal, or `date=` to another day,
# and re-run -- the length printed below will change.
# ============================================================================
train, probe = st.load_trials(
    data_path,
    name="Jinkx",          # <-- change me: another subject's name
    kind="Training",       # <-- or 'Probe'
    date="29_6_2026",      # <-- or another session date 'dd_mm_yyyy'
)

print("training trials loaded:", len(train))
print("probe trials loaded    :", len(probe))   # 0, because include_probes=False


# %% Cell 3 -- what is actually inside one trial?
# ============================================================================
# load_trials() returns a LIST of tuples. Each tuple is one trial. The
# useful fields, by position, are:
#
#   train[0][0]  -> name           (str)
#   train[0][3]  -> kind           ('Training' or 'Probe')
#   train[0][4]  -> env            ('Circle' or 'Square')
#   train[0][5]  -> date           (str)
#   train[0][6]  -> age            (int, days)
#   train[0][8]  -> rewarded_well  (int, 1-based well number)
#   train[0][10] -> trial_number   (int)
#   train[0][15]-> df              (a pandas DataFrame of x,y positions, in cm)
#
# `df` (index 15) is the actual animal track for that trial -- the thing we
# plot. Let's look at the first trial:
# ============================================================================
t0 = train[0]
print("subject     :", t0[0])
print("session type :", t0[3])
print("environment :", t0[4])
print("trial number:", t0[10])
print("rewarded well:", t0[8])

df = t0[15]                 # the position data for this trial
print("number of tracked frames:", len(df))
print(df.head())           # first few x,y rows -- should be numbers in cm


# %% Cell 4 -- filter further, WITHOUT reloading the file
# ============================================================================
# Because `train` already lives in the Spyder console, you can slice it with
# plain Python to get a subset -- no need to call load_trials() again.
# This is the "get a subset of that data" part the supervisor mentioned.
#
# Example: take only the first 5 trials of this session.
# Change [:5] to [:10] or [5:] and re-run to see the subset change.
# ============================================================================
subset = train[:5]          # <-- change me: [:10], [5:], or a specific list
print("subset size:", len(subset))

# You can also filter by a condition, e.g. only trials where the animal was
# rewarded at well 3:
well3 = [t for t in train if t[8] == 3]
print("trials to well 3:", len(well3))


# %% Cell 5 -- plot a single trial's trajectory (sv.plot_trajectory)
# ============================================================================
# Now we use one plotting function from visualization.py.
# plot_trajectory(df, rewarded_well, env) draws the animal's path for ONE
# trial, with the environment boundary, all the wells, and the rewarded well
# highlighted. We feed it the trial we inspected in cell 3.
# ============================================================================
sv.plot_trajectory(df, t0[8], t0[4])   # df, rewarded_well, env

# Try it on a different trial: change the index below and re-run.
sv.plot_trajectory(train[1][15], train[1][8], train[1][4])


# %% Cell 6 -- plot several trials together (sv.plot_multiple_trajectories)
# ============================================================================
# plot_multiple_trajectories(trials_data) draws many trials on one set of
# axes, each in a different colour. Give it the subset from cell 4.
# ============================================================================
sv.plot_multiple_trajectories(subset)

# Now give it a bigger subset and re-run -- more coloured lines appear.
sv.plot_multiple_trajectories(train[:10])


# %% Cell 7 -- training + probe trials together (sv.plot_training_and_probes)
# ============================================================================
# This is the function with the boolean toggle from the teaching guide.
# First we need some probe trials, so we re-load WITH include_probes=True.
# `same_day=True` grabs probes from the same date as the training session;
# set it to False to grab them from the next day instead and compare.
# ============================================================================
train2, probe2 = st.load_trials(
    data_path,
    name="Jinkx",
    kind="Training",
    date="29_6_2026",
    include_probes=True,    # <-- now probe2 will be non-empty
    same_day=True,          # <-- try False: probes come from the next day
)
print("train:", len(train2), "| probe:", len(probe2))

# Two separate colour palettes (training vs probe) -- the default.
sv.plot_training_and_probes(train2, probe2, separate_color_scale=True)

# Flip the boolean: every trial gets its own colour from a rainbow instead.
sv.plot_training_and_probes(train2, probe2, separate_color_scale=False)


# %% Cell 8 -- a population-style plot (sv.plot_trial_completion)
# ============================================================================
# Some functions take the whole JSON (not the filtered trial list) and do
# their own counting. plot_trial_completion(filepath, subjects) computes the
# % of completed vs failed probe trials per subject and stacks them to 100%.
# ============================================================================
with open(data_path) as f:
    subjects = [s["Name"] for s in json.load(f)]

sv.plot_trial_completion(data_path, subjects)        # all animals
sv.plot_trial_completion(data_path, subjects[:3])    # <-- fewer animals

# ============================================================================
# End of demo. To explore more, open visualization.py, pick any function that
# starts with `plot_`, and call it from the console using the `train` /
# `probe` lists you already have in memory. Use the teaching guide for the
# "change one input, re-run, see the output change" exercises.
# ============================================================================

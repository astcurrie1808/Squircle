# Intern Spyder Walkthrough — Teaching Guide

A self-contained script for running the Squircle code in Spyder with the
other interns. The goal (from the supervisor) is:

> "run a function, show the output, then change the inputs to the
> function, run it again and show that the output is different."

Every exercise below is set up exactly that way: a **baseline call**, then a
**changed call** where you flip one input and re-run. That is the whole
teaching loop — do not improvise extra code live.

---

## 0. How to set up before the session

1. Open Spyder, then open these three files from this repo:
   - `SquircleAnalysis.py` ← this is your "console" (you run things from here)
   - `tools.py` ← where data is loaded (the `load_trials` function lives here)
   - `visualization.py` ← where plots are made (the functions you'll toggle)

2. At the very top of `SquircleAnalysis.py` there are two lines that point at
   folders on the supervisor's machine:

   ```python
   base_folder   = 'W:\\ephys\\Jose\\Squircle Project\\Data\\Litter11'
   working_folder = r'C:\Users\astcu\OneDrive\Documents\2025-2026\Neurobiologie\Internship'
   ```

   These paths must exist on the laptop you use. If a line throws
   `FileNotFoundError`, the path is wrong — fix it, do **not** change the
   code below it. The script builds three files in order:

   ```python
   json_path       = st.create_json(base_folder, working_folder, verbose=True)
   processed_path  = st.preprocess_json_positions(json_path)
   converted_path  = st.convert_positions_to_cm(processed_path)
   ```

   `converted_path` is the file everything else reads. Run these once at the
   start so the data exists.

3. Run the import block (top of `SquircleAnalysis.py`) once so `sv`
   (visualization) and `st` (tools) are available in the console.
   If you edit `visualization.py` or `tools.py` mid-session, re-run the
   `importlib.reload(sv)` / `importlib.reload(st)` lines so Spyder picks up
   the change — this is the single most important Spyder habit to show them.

---

## The one Spyder concept to teach first

Spyder runs a persistent Python console. When you run a cell (Ctrl+Enter) or
a selection (F9), the variables stay in memory. That is why this works:

```python
train, probe = st.load_trials(converted_path, name='Jinkx',
                              kind='Training', date='29_6_2026',
                              include_probes=True, same_day=True)
```

After this runs, `train` and `probe` exist in the console. You can then type
`len(train)` in the console and see the answer immediately, or call a plot
function with them. You do **not** have to re-load the data each time.

Also show the **Variable Explorer** pane (top-right by default): after the
call above, `train` and `probe` appear there and you can click to inspect
them. This is how the interns see that "changing the input changed the
output" even before plotting.

---

## Exercise 1 — Two color palettes vs. all-different colors
*(the "rainbow → reds+blues" exercise, now a single toggle)*

### What the code does
`plot_training_and_probes` (in `visualization.py`, around line 367) now has a
boolean parameter:

```python
def plot_training_and_probes(trials_data, probe_trials,
                             colors=None,
                             separate_color_scale=True):
```

The interesting part is right at the top of the function body:

```python
if colors is None:
    if separate_color_scale:
        train_map = plt.cm.YlGn
        probe_map = plt.cm.RdPu
        train_colors = [train_map(x) for x in np.linspace(0.4, 0.9, num_train)] if num_train > 0 else []
        probe_colors = [probe_map(x) for x in np.linspace(0.4, 0.9, num_probe)] if num_probe > 0 else []
        colors = train_colors + probe_colors
    else:
        combined_map = plt.cm.turbo
        colors = [combined_map(x) for x in np.linspace(0, 1, num_trials)]
```

### The two concepts to explain (from your own notes)

- **`len()`** returns how many items are in a list.
  `num_train = len(trials_data)` = "how many training trials did we load?"
  This number decides how many colors we need.
- **`plt.get_cmap()` / `plt.cm.<name>`** returns a colormap, which is itself a
  function: you give it a number between 0 and 1 and it gives back an RGBA
  color. `np.linspace(0.4, 0.9, n)` produces `n` evenly spaced numbers between
  0.4 and 0.9, so each trial gets a slightly different shade of the same
  colormap. That is why one group is "all greens" and the other "all purples"
  instead of a rainbow.

### Run it — baseline (two separate palettes, the default)

```python
import squircle.visualization as sv
import squircle.tools as st
import importlib
importlib.reload(sv); importlib.reload(st)

converted_path = os.path.join(working_folder, "litter_11_processed_cm.json")

train, probe = st.load_trials(converted_path, name='Jinkx',
                              kind='Training', date='29_6_2026',
                              include_probes=True, same_day=True)

sv.plot_training_and_probes(train, probe, separate_color_scale=True)
```

You should see training trials in one color family and probe trials in
another, with a compact legend (just "Training Trials", "Probe Trials", …).

### Run it again — flip the boolean

Change **one** word and re-run only the plot line:

```python
sv.plot_training_and_probes(train, probe, separate_color_scale=False)
```

Now every trial gets its own color from the `turbo` rainbow, and the legend
lists every individual trial. That visible jump is the whole point: a single
boolean changed which branch of the `if` ran, which changed the `colors`
list, which changed the picture.

### Teaching note
After showing both, open `visualization.py` at the `if separate_color_scale:`
line and walk them through the two branches literally line by line. The
"aha" is that `True` runs the first block and `False` runs the `else` block —
nothing else changes. That is what a boolean parameter *is*: a switch that
picks a branch.

---

## Exercise 2 — `include_probes` and `same_day` (loading data differently)
*(the "include probe on the same day" exercise)*

### What the code does
`load_trials` (in `tools.py`, around line 672) has two booleans:

```python
def load_trials(filepath, name=None, kind=None, environment=None, date=None,
                number=None, rotation=None, baited=None,
                include_probes=False, same_day=False):
```

and the logic that uses them:

```python
probe_trials = []
if include_probes:
    if same_day:
        probe_date = date                       # use today's date
    else:
        date_obj = datetime.strptime(date, "%d_%m_%Y") if date else None
        probe_date = (date_obj + timedelta(days=1)).strftime("%d_%m_%Y") if date_obj else None
    probe_trials = load_population_trials(data=litter_data, name=name,
                                          kind='Probe', date=probe_date, ...)
```

### The concept to explain
- `include_probes=False` (the default) → no probe trials are loaded at all;
  the function returns an empty `probe_trials` list.
- `include_probes=True, same_day=False` → probes come from the **next day**
  (`date + 1 day`).
- `include_probes=True, same_day=True` → probes come from the **same day**
  (`probe_date = date`).

So `same_day` only matters *because* `include_probes` is True — it chooses
*which day's* probes to grab. This is the exact same boolean-toggle pattern
as Exercise 1: a flag that decides which branch runs.

### Run it — baseline (no probes)

```python
train, probe = st.load_trials(converted_path, name='Jinkx',
                              kind='Training', date='29_6_2026')
print("training trials:", len(train), "| probe trials:", len(probe))
```

`len(probe)` is `0` because `include_probes` defaults to `False`.

### Run it again — flip the first boolean

```python
train, probe = st.load_trials(converted_path, name='Jinkx',
                              kind='Training', date='29_6_2026',
                              include_probes=True)
print("training trials:", len(train), "| probe trials:", len(probe))
```

Now `len(probe)` should be > 0 — probes were loaded from the next day
(`30_6_2026`).

### Run it a third time — flip the second boolean too

```python
train, probe = st.load_trials(converted_path, name='Jinkx',
                              kind='Training', date='29_6_2026',
                              include_probes=True, same_day=True)
print("training trials:", len(train), "| probe trials:", len(probe))
```

The probe date is now `29_6_2026` (same day). If that session actually had
probes on the same day, `len(probe)` changes again — proving the second
flag had an effect. (If the count is the same, that just means the next
day and the same day happened to have the same number of probe trials for
this animal — pick a subject/date where they differ, or just explain it.)

### Teaching note
Show the printed `len()` numbers changing in the console. This is the
cleanest "input changed → output changed" demonstration because it's a
number, not a picture, so there's no ambiguity.

---

## Exercise 3 — Stacked bars that add up to 100%
*(the "stack bars so they add up to 100%" exercise)*

### What the code does
`plot_trial_completion` (in `visualization.py`, around line 581) used to use
`sns.barplot` with `hue`. It now uses two `plt.bar` calls:

```python
x = np.arange(len(subjects))

plt.bar(x, completed_counts, label="Completed", color="tab:blue")
plt.bar(x, failed_counts, bottom=completed_counts, label="Failed", color="tab:red")
```

### The concept to explain
- `plt.bar(x, heights)` draws bars from 0 up to `heights`.
- `plt.bar(x, heights, bottom=other_heights)` draws bars that **start at**
  `other_heights` and go up. So the "Failed" bars sit on top of the
  "Completed" bars — that's what "stacked" means. The single keyword
  `bottom=completed_counts` is the whole trick.
- `plt.ylim(0, 100)` + the fact that completed% + failed% = 100% is why
  every stacked bar reaches exactly the top.

### Run it — baseline

```python
import json
converted_path = os.path.join(working_folder, "litter_11_processed_cm.json")
with open(converted_path) as f:
    data = json.load(f)
subjects = [s["Name"] for s in data]

sv.plot_trial_completion(converted_path, subjects)
```

Each subject gets one bar; the blue part is completed %, the red part
stacked on top is failed %, and the total always hits 100.

### Run it again — change the subject list

```python
sv.plot_trial_completion(converted_path, subjects[:3])   # only first 3 subjects
```

Fewer bars appear, but each bar still stacks to 100%. This shows that the
function responds to its input (the `subjects` list) without you touching
the plotting code.

### Optional "show the old way" contrast
If you want to show *why* the new code is better, you can mention (don't
have to run) that the old `sns.barplot(..., hue="Trial Type")` version put
Completed and Failed **side by side** for each subject, so they didn't add
to 100 visually. The `bottom=` keyword is what turned "side by side" into
"stacked". The teaching point: one keyword (`bottom=`) changed the geometry
of the chart.

---

## Exercise 4 — The "graphs switch around" puzzle
*(changing `env_split=True` → `False` reorders the Q1 vs Q2 boxplots)*

### What the code does
`boxplot_quantiles` (in `visualization.py`, around line 805) has:

```python
def boxplot_quantiles(population_trials, q1, q2, min_age=None, max_age=None,
                      env_split=True):
```

The key difference is what string goes into the `Quartile` column of the
data, which then becomes the legend/hue label:

- `env_split=True` → the label is
  `f'{environment} ({baited_label}) - {label}'`, e.g.
  `"Circle (Baited) - T1"`. So you get **separate boxes per environment**.
- `env_split=False` → the label is just `label`, e.g. `"T1"`. All
  environments are pooled together.

### Why the graphs "switch around"
The order of the boxes on the plot is decided by `make_boxplot` (around
line 729), specifically this sort key:

```python
def quartile_sort_key(q):
    if " - " in q:
        env_part, label_part = q.split(" - ", 1)
    else:
        env_part, label_part = "", q
    return (env_part, labels.get(label_part, 99))
```

- With `env_split=True`, every label contains `" - "`, so each one is split
  into `(environment_part, quartile_part)` and sorted by **environment
  first**, then quartile. The boxes cluster by environment
  (all Circle entries together, then all Square).
- With `env_split=False`, no label contains `" - "`, so `env_part` is `""`
  for all of them and they're sorted purely by `labels.get(label_part)` —
  i.e. by `q1` then `q2`.

So flipping the boolean doesn't just "remove the environment split", it
also **changes the sort key**, which changes the left-to-right order of the
boxes. That's the answer to the interns' "why did they switch around?"
question: the same flag controls both *what's plotted* and *the order it's
plotted in*, because the label string format changes, and the sort key
reads that string format.

### Run it — baseline (split by environment)

```python
merged = os.path.join(working_folder, "combined_litters_9_10_11_.json")
trials = st.load_population_trials(data=json.load(open(merged)))

sv.boxplot_quantiles(trials, 't1', 't4', env_split=True)
```

Boxes are grouped by environment.

### Run it again — flip the boolean

```python
sv.boxplot_quantiles(trials, 't1', 't4', env_split=False)
```

Now there are only two box categories (T1 and T4), pooled across all
environments, and their left-to-right order is different from before. Point
at the legend order before and after — that's the "switch".

### Teaching note
This is the best exercise for the "booleans have side effects you don't
expect" lesson. The boolean is named `env_split` so you'd think it only
toggles the split; but because it also changes the label *string*, and a
sort key parses that string, the order changes too. Good lesson: a flag's
effect can ripple through code that doesn't obviously mention the flag.

---

## Hands-on exercises for the interns

The four sections above are **demos** — you run them, the interns watch.
This part is different: these are tasks the **interns do themselves**, at
their own keyboard, in Spyder. Each one is deliberately tiny: predict →
run → observe → change one thing → run again. That is the supervisor's
whole method, turned into things they can touch.

Hand them `intern_demo.py` (the companion script) and ask them to work
through these in order. The demo script already has cells set up for each;
these instructions tell them what to predict and what to change.

### Exercise A — Read the data and inspect one trial (warm-up)

**Goal:** confirm they can load data and look inside one trial tuple before
plotting anything.

1. Run **Cell 0** (imports) and **Cell 1** (point at the data) of
   `intern_demo.py`. They should see one line printed per animal with
   non-zero session and trial counts.
2. Run **Cell 2** to load one subject's training trials:
   ```python
   train, probe = st.load_trials(data_path, name='Jinkx',
                                 kind='Training', date='29_6_2026')
   print(len(train), len(probe))
   ```
3. **Predict:** before running the next line, ask: "what does `train[0][15]`
   give us?" (answer: the position DataFrame of the first trial — index 15,
   from the tuple structure).
4. **Run it and check:**
   ```python
   print(train[0][0])      # the name
   print(train[0][6])      # the age
   print(len(train[0][15])) # number of tracked frames
   ```
   They should see a name, an age in days, and a frame count.
5. **Change one thing:** swap `name='Jinkx'` for another animal from the
   Cell 1 printout, re-run Cell 2, and confirm `train[0][0]` now prints the
   new name. **Lesson:** the tuple's position 0 is always the name, and
   changing the input changed which animal's data we got.

### Exercise B — Subset the data with plain Python

**Goal:** show that filtering does not require reloading the file — you can
slice a list you already have.

1. After Exercise A, `train` is still in memory. Run:
   ```python
   subset = train[:5]
   print(len(subset))
   ```
   They should see `5`.
2. **Predict:** what does `train[5:]` give? (`len(train) - 5` trials —
   everything *after* the first 5.)
3. **Run it and check:** `print(len(train[5:]))`.
4. **Change one thing:** make a subset of only trials to a specific well:
   ```python
   to_well3 = [t for t in train if t[8] == 3]
   print(len(to_well3))
   ```
   `t[8]` is the rewarded-well field (index 8 of the tuple). Changing the
   `3` to another well number changes the count. **Lesson:** the tuple's
   numbered fields let you filter on any property without touching the JSON
   or reloading.

### Exercise C — Flip the colour toggle (the `separate_color_scale` demo, but they drive)

**Goal:** they experience a boolean changing a visible output.

1. Run **Cell 7** of `intern_demo.py` up to the load line, so `train2` and
   `probe2` exist.
2. Run the first plot call:
   ```python
   sv.plot_training_and_probes(train2, probe2, separate_color_scale=True)
   ```
   Ask them to describe what they see (two colour families, compact legend).
3. **Predict:** "what will change if I set `separate_color_scale=False`?"
4. **Run it:**
   ```python
   sv.plot_training_and_probes(train2, probe2, separate_color_scale=False)
   ```
   They should see a rainbow with many legend entries.
5. **Open the code:** jump to `visualization.py` at the
   `if separate_color_scale:` line and walk through the two branches
   literally — True runs the two-palette block, the `else:` runs the turbo
   block. **Lesson:** a boolean is a switch that picks a branch; the branch
   picked is what changed the picture.

### Exercise D — Flip two booleans in the loader (the `include_probes` / `same_day` demo, but they drive)

**Goal:** they see a boolean change a *number* in the console, not just a
picture — the cleanest possible input → output demo.

1. Run, and read the count:
   ```python
   train, probe = st.load_trials(data_path, name='Jinkx',
                                 kind='Training', date='29_6_2026')
   print("probes:", len(probe))
   ```
   Expect `0` (default `include_probes=False`).
2. **Predict:** "what will `len(probe)` be if I add `include_probes=True`?"
3. **Run it:**
   ```python
   train, probe = st.load_trials(data_path, name='Jinkx',
                                 kind='Training', date='29_6_2026',
                                 include_probes=True)
   print("probes:", len(probe))
   ```
   Expect > 0 — probes from the next day.
4. **Predict:** "what will `same_day=True` change?"
5. **Run it:**
   ```python
   train, probe = st.load_trials(data_path, name='Jinkx',
                                 kind='Training', date='29_6_2026',
                                 include_probes=True, same_day=True)
   print("probes:", len(probe))
   ```
   The probe date is now the same as the training date; the count may change
   (or not — see the teaching note in the demo above).
6. **Open the code:** show the `if include_probes:` / `if same_day:` block in
   `tools.py` and point at `probe_date = date` vs the `+ 1 day` branch.
   **Lesson:** two nested booleans, each picking a branch; `same_day` only
   matters *because* `include_probes` is True.

### Exercise E — The stacking keyword (the stacked-bars demo, but they drive)

**Goal:** they see one keyword (`bottom=`) change bar geometry.

1. Run **Cell 8** of `intern_demo.py`:
   ```python
   sv.plot_trial_completion(data_path, subjects)
   ```
   Each bar reaches 100 (blue completed + red failed stacked).
2. **Change the input:**
   ```python
   sv.plot_trial_completion(data_path, subjects[:3])
   ```
   Fewer bars, each still hits 100.
3. **Optional — see the old way:** if you want to show *why* the new code is
   better, open `plot_trial_completion` and point at the two `plt.bar`
   lines. Mention (don't necessarily run) that removing `bottom=` would put
   the bars side by side instead of stacked, so they wouldn't add to 100
   visually. **Lesson:** one keyword (`bottom=`) turned "side by side" into
   "stacked".

### Exercise F — The reordering puzzle (the `env_split` demo, but they drive)

**Goal:** they discover that a boolean can have a *side effect* beyond its
obvious name — here, changing the on-plot order.

1. Run:
   ```python
   sv.boxplot_quantiles(trials, 't1', 't4', env_split=True)
   ```
   Note the legend order (grouped by environment).
2. **Predict:** "what will `env_split=False` change?" (Most will say "it just
   removes the environment split" — that's the setup.)
3. **Run it:**
   ```python
   sv.boxplot_quantiles(trials, 't1', 't4', env_split=False)
   ```
   Not only are environments pooled — the left-to-right order of the boxes is
   *different*. Point at the legend before and after: that's the switch.
4. **Open the code:** show the `quartile_sort_key` in `make_boxplot`. The
   flag changed the *label string* (e.g. `"Circle (Baited) - T1"` → `"T1"`),
   and the sort key parses that string, so the order changed too.
   **Lesson:** a flag's effect can ripple through code that doesn't obviously
   mention the flag. This is the most important conceptual exercise — it's
   the one where being wrong in your prediction is the point.

### How to run these with the interns

- Do **Exercise A and B together** at the front first, because they teach
  the tuple structure everyone needs for the rest.
- Then let them do **C, D, E** independently — they're all the same pattern
  (flip one input, re-run), so once they've done one, the others are easy.
- Save **F** for last and do it as a group discussion, because the "predict
  wrong on purpose" moment only works once, and the reordering cause is the
  one thing in this guide that isn't obvious from the boolean's name.

---

## Cheat sheet: the four boolean toggles in this repo

| Function (file)              | Parameter             | `True` does                       | `False` does                       |
|------------------------------|-----------------------|-----------------------------------|------------------------------------|
| `plot_training_and_probes` (visualization.py) | `separate_color_scale` | two palettes (train vs probe)     | one rainbow, every trial different |
| `load_trials` (tools.py)     | `include_probes`      | load probe trials too             | load none (empty list)             |
| `load_trials` (tools.py)     | `same_day`            | probes from same date as training | probes from next day               |
| `boxplot_quantiles` (visualization.py) | `env_split`           | separate boxes per environment    | pool all environments; reorders boxes |

Every one of these is the same pattern: **a boolean parameter picks which
branch of an `if` runs, and that branch changes the output.** That single
sentence is the thing the interns should leave with.

---

## If something breaks live

- `FileNotFoundError` → the `working_folder`/`base_folder` path at the top of
  `SquircleAnalysis.py` is wrong for this laptop. Fix the path, not the code.
- `NameError: name 'sv' is not defined` → you didn't run the import cell.
- A plot didn't change after you edited `visualization.py` → run
  `importlib.reload(sv)` and call the function again. (Same for `st`.)
- `ValueError: No trials selected for plotting.` → your `load_trials`
  filters returned 0 trials; loosen the filters (e.g. remove `date=` or
  try a different `name`).

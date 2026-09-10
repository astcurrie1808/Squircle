# Understanding the Squircle Code — A Guide for You

This is not for the interns. This is for *you*: a personal map of how the
`.py` files in this repository fit together, how the data is structured, and
how you can read the code well enough to explain it and to tweak it with
confidence — even if you can't yet write a plotting function from scratch.

It assumes you "recognise" Python: you know what a function, a list, a
`for` loop, a `dict`, and an `if` are when you see them, but you would not
necessarily produce them yourself. That is exactly the level this is
written for.

---

## 1. The five-second mental model

The whole project is a small **pipeline**:

```
   raw video files (on the lab server)
            |
            v   st.create_json(...)          <- tools.py
   a JSON file of raw pixel positions
            |
            v   st.preprocess_json_positions(...)   <- tools.py
   a JSON file of cleaned pixel positions
            |
            v   st.convert_positions_to_cm(...)     <- tools.py
   a JSON file of cleaned positions in centimetres   <-- THIS is the file everything reads
            |
            v   st.load_trials(filepath, ...)        <- tools.py
   Python lists of trial "tuples"  (one tuple = one trial, in memory)
            |
            v   sv.plot_<something>(trials, ...)      <- visualization.py
   matplotlib figures
```

Three files matter for you:

| File                  | Role                                                       |
|-----------------------|------------------------------------------------------------|
| `tools.py`            | Builds and loads the data. You call `load_trials()`.       |
| `visualization.py`    | Makes the plots. Every `plot_...` function lives here.      |
| `SquircleAnalysis.py` | Not code logic — it is your "console". You run cells here.  |

There is also a `design.py` that is **not in this repository** but is
imported by both modules (`from squircle.design import ...`). It holds the
physical constants of the experiment: where the wells are, the circle
radius, the square side, frames-per-second, cm-per-pixel. You do not edit
it; you just need it to exist on the laptop (see section 9).

---

## 2. How the data is nested (the JSON)

The converted JSON file is a nested structure. Learning this shape is the
single most useful thing you can do, because *every* function in the repo
is just walking through this shape and pulling pieces out.

```
[                              <-- a list of ANIMALS
  {                            <-- one animal = a dict
    "Name": "Jinkx",
    "DOB": "...",
    "Sex": "F",
    "Sessions": [              <-- a list of SESSIONS
      {                        <-- one session = a dict
        "Type": "Training",     <-- 'Training' or 'Probe'
        "Environment": "Circle",
        "Date": "29_6_2026",
        "Age": 45,             <-- subject's age that day
        "Rotation": 0,         <-- 0 normal, 1 rotated
        "Rewarded well": 3,
        "Center x": ..., "Center y": ...,
        "Trials": [            <-- a list of TRIALS
          {                    <-- one trial = a dict
            "Number": 1,
            "Start frame": ...,
            "End frame": ...,
            "Baited": 0,
            "Completed": "Completed",
            "Position": "<JSON string of x,y rows>"
          },
          ...
        ]
      },
      ...
    ]
  },
  ...
]
```

So the depth is: **file → animals → sessions → trials**. Almost every
piece of code in this repo is a loop that says, in effect, "for each
animal, for each session, for each trial, do something". Once you see
that, most of the file stops being intimidating.

You can confirm this shape live in Spyder at any time:

```python
import json
with open(converted_path) as f:
    data = json.load(f)
data[0]                          # the first animal
data[0]["Sessions"][0]           # its first session
data[0]["Sessions"][0]["Trials"][0]   # the first trial of that session
```

That three-level drill-down is how you should explore any data structure
you don't understand: read one element at each level with `[0]`.

---

## 3. The trial tuple (the thing you said you didn't yet know)

This is the heart of the whole codebase, and it is simpler than it looks.

When you call `load_trials(...)`, it walks the JSON and, for every trial
that passes your filters, builds a **tuple** — an ordered, fixed-length
container, written with round brackets. That tuple is the "standard trial
format" that every plotting function expects.

You do not need to memorise it. You need to know **that it exists** and
**where to look it up**. Here is the exact order, built at
`tools.py` around line 650:

```python
trial_info = (
    subject_data["Name"],      # 0   name           (str)
    subject_data["DOB"],       # 1   dob            (str)
    subject_sex,               # 2   sex            (str)
    session["Type"],           # 3   kind           ('Training' or 'Probe')
    session["Environment"],    # 4   env            ('Circle' or 'Square')
    session["Date"],           # 5   date           (str, 'dd_mm_yyyy')
    session["Age"],            # 6   age            (int, days)
    session["Rotation"],       # 7   rotation       (0 or 1)
    session["Rewarded well"],  # 8   rewarded_well  (int, 1-based)
    session_center,            # 9   session_center ((x, y) tuple)
    trial["Number"],           # 10  trial_number   (int)
    trial["Start frame"],      # 11  start_frame
    trial["End frame"],        # 12  end_frame
    trial["Frame offset x"],   # 13  fx
    trial["Frame offset y"],   # 14  fy
    df,                        # 15  df             (pandas DataFrame of x,y)
    trial["Baited"],           # 16  baited         (0 or 1)
    trial.get("Completed", "Completed"),  # 17  completed status (added later)
)
```

The index on the left (0–17) is how the rest of the code reads it: to get
the position data you write `trial[15]`; to get the rewarded well you write
`trial[8]`; to get the age you write `trial[6]`. That is the whole
"tuple structure" — there is nothing fancier going on.

### Two ways the code reads these tuples (learn both)

**Way A — by index.** Used in the bigger analysis functions because it is
fast to type:

```python
subject = trial[0]
age     = trial[6]
df      = trial[15]
```

**Way B — by unpacking.** Used at the top of plotting functions because it
is readable. You assign a name to every position in one line:

```python
(name, dob, sex, kind, env, date, age, rotation, rewarded_well,
 session_center, trial_number, start_frame, end_frame,
 fx, fy, df, baited) = trial
```

After that line, `df`, `rewarded_well`, `env`, etc. are all available by
name. When you see a long `( ... ) = trial` line in `visualization.py`,
that is all it is doing: naming the 17/18 slots in order.

### A real inconsistency you should know about

The docstring in `load_trials` says "17-tuple", and most unpacking lines
list 17 names. But the actual tuple has **18** elements, because
`trial.get("Completed", "Completed")` was appended at the end (index 17).
That means:

- Reading by index (Way A) always works — `trial[15]` is the df regardless.
- Unpacking (Way B) only works if the tuple has exactly as many names as
  the unpacking line lists. `plot_multiple_trajectories` lists 17 names, so
  on an 18-element tuple it would actually raise `ValueError: too many
  values to unpack`. In practice this only bites if a trial's `Completed`
  field is present. If a plotting function ever throws that error, this is
  why: add `completed` as an 18th name (or use `*_` to swallow the rest).

This is the kind of "I'd never have guessed that" detail that only becomes
visible once you know the tuple is the shared currency of the whole repo.

---

## 4. What a plotting function looks like (the universal pattern)

Nearly every function in `visualization.py` follows the same skeleton.
If you learn this one shape, you can read ~80% of the file.

```python
def plot_something(trials_data, ...):
    # 1. Guard: empty input?
    if len(trials_data) == 0:
        raise ValueError(...)

    # 2. Set up colours / figure
    fig = plt.figure(figsize=standard_figsize)

    # 3. Loop over trials, unpacking the tuple each time
    for i, trial in enumerate(trials_data):
        (name, ..., env, ..., rewarded_well, ..., df, ...) = trial

        # skip trials with no tracking
        if df.empty:
            continue

        # 4. Compute or extract something from df
        #    e.g. distance, efficiency, or just plot x vs y
        plt.plot(df["x"], df["y"], color=..., label=...)

    # 5. Decoration: axis labels, title, legend, limits
    plt.xlabel(...); plt.ylabel(...); plt.title(...)
    plt.legend(...)

    # 6. Show + return the figure
    plt.show()
    return fig
```

The only part that changes between functions is **step 4**: what you
compute from `df` and which `plt.*` call you make. Everything else is
boilerplate. So when you open an unfamiliar `plot_...` function, scroll
straight to the `for` loop and read only the `plt.plot` / `plt.scatter`
/ `sns.boxplot` line inside it — that is the actual graph. The rest is
scaffolding.

### The one helper worth knowing: `st.calculate_efficiency(df, rewarded_well)`

A lot of functions don't plot positions at all; they plot an **efficiency
ratio** per trial. That number comes from this helper in `tools.py`. It is
just: how far the animal actually walked (sum of step lengths) divided by
the straight-line distance from start to reward. A perfect path = 1.0; a
wandering path = 2.0, 5.0, etc. When a function talks about "efficiency",
it is calling this. You don't need to reimplement it; you call it the same
way the plots do: `st.calculate_efficiency(df, rewarded_well)`.

---

## 5. `load_trials()` read carefully (because it is your main tool)

```python
def load_trials(filepath, name=None, kind=None, environment=None, date=None,
                number=None, rotation=None, baited=None,
                include_probes=False, same_day=False):
```

Everything except `filepath` has a default of `None`/`False`, which means
"don't filter on this". So:

- `load_trials(filepath)` → every trial of every animal (big list).
- `load_trials(filepath, name="Jinkx")` → only Jinkx's trials.
- `load_trials(filepath, name="Jinkx", kind="Training", date="29_6_2026")`
  → only Jinkx's training trials on that one day.

It returns **two** things: `(selected_trials, probe_trials)`. Always unpack
both, even if you ignore the second:

```python
train, probe = st.load_trials(...)   # probe is [] unless include_probes=True
```

The `include_probes` / `same_day` pair is the only mildly clever part, and
it is just date arithmetic:

```python
if include_probes:
    if same_day:
        probe_date = date                       # same date as training
    else:
        probe_date = date + 1 day               # the next day
    probe_trials = load_population_trials(..., kind='Probe', date=probe_date, ...)
```

So the two booleans simply decide *whether* to fetch probes and *which
day* to fetch them from. Nothing else.

### The filtering is "and", not "or"

Every filter you set must match for a trial to be kept. If you ask for
`name="Jinkx"` AND `kind="Training"` AND `date="29_6_2026"`, you get only
trials that satisfy **all three**. If you get zero trials back, you have
over-constrained — drop one filter (often `date=`) and try again.

---

## 6. How to read code you can't yet write (your actual situation)

You said you can recognise Python but not produce it. Here is a method
that works specifically for this repo, in Spyder:

1. **Run, don't read.** Before reading a function, call it with real data
   and look at the picture. Now you know what the code is *for*, which
   makes the code far easier to read.
   ```python
   sv.plot_multiple_trajectories(train[:5])
   ```
2. **Then read only the `plt.*` line.** Open the function, find the
   `for` loop, and look at the single `plt.plot(...)` or `plt.scatter(...)`
   line. That line *is* the graph. The arguments tell you what is on each
   axis (`df["x"]`, `df["y"]`), what colour, and what the legend says.
3. **Change one argument, re-run.** Change the colour, or the `label=`,
   or `alpha=`, re-run, and watch the picture change. This is how you
   learn what each argument does without writing anything from scratch.
   This is also exactly what your supervisor wants you to do with the
   interns.
4. **Use the Variable Explorer.** After `load_trials`, click `train` in
   the Variable Explorer. You can see it is a list; open it; element 0 is
   a tuple; open that; you see the 18 fields. This is faster than guessing
   from code.
5. **Use `?` in the console.** Type `st.load_trials?` and Spyder shows the
   docstring. Type `sv.plot_trajectory?` for that one. The docstrings here
   are decent — they list each parameter and what it returns.

The key reframe: you do not need to be able to *write* `plt.plot(...)`.
You need to be able to *find* the `plt.plot(...)` line that already
exists, change one word in it, and re-run. That is a much smaller skill,
and it is enough for instructing the interns.

---

## 7. The matplotlib vocabulary you'll keep seeing

You don't need to know all of matplotlib. You need about eight calls:

| Call                        | What it draws                         |
|-----------------------------|----------------------------------------|
| `plt.figure(figsize=(w,h))` | starts a new figure                    |
| `plt.plot(x, y, ...)`       | a connected line (a trajectory)        |
| `plt.scatter(x, y, ...)`    | dots (wells, start positions)          |
| `plt.bar(x, h, bottom=...)` | a bar (optionally stacked)             |
| `plt.Circle((cx,cy), r)`    | the arena boundary (added with `ax.add_patch`) |
| `plt.Rectangle(...)`        | the square arena boundary              |
| `plt.legend(handles=...)`   | the legend                             |
| `plt.show()`                | actually display the figure           |

Every "complicated" figure in this repo is just several of these called
in a row inside a `for` loop. The two non-obvious ones:

- `alpha=0.7` on `plt.plot` makes lines semi-transparent so overlapping
  trajectories are visible. That is why plots of many trials look "see-through".
- `plt.scatter(*zip(*well_locations_cm))` looks scary but just means
  "plot all the wells at once". `zip(*...)` transposes a list of (x,y)
  pairs into two separate lists (all x's, all y's), and the `*` feeds them
  as two arguments. You do not need to reproduce this; just recognise it.

---

## 8. A worked example: read `plot_trajectory` end to end

Let's apply the method to a real function. Find it at `visualization.py`
around line 99. Stripped to its bones it is:

```python
def plot_trajectory(df, rewarded_well, env):
    reward_x, reward_y = well_locations_cm[rewarded_well - 1]   # where's the reward?

    fig = plt.figure(figsize=(8, 6))                  # new figure

    plt.plot(df["x"], df["y"], color="black", alpha=0.6)   # <-- THE graph: path
    plt.scatter(df.iloc[0]["x"], df.iloc[0]["y"], ...)     # start dot
    plt.scatter(reward_x, reward_y, color="red", ...)      # rewarded well
    plt.scatter(*zip(*well_locations_cm), color="gray")   # other wells

    # ... draws the Circle or Square boundary depending on env ...

    plt.xlim(...); plt.ylim(...); plt.gca().set_aspect("equal")  # square axes
    plt.legend(...)
    plt.show()
    return fig
```

Notice: this function takes a **single trial's `df`** directly (not a
list of tuples), because it plots one trial. Compare with
`plot_multiple_trajectories(trials_data)`, which takes a **list of
tuples** and loops. The signature tells you which to expect:

- argument named `df` → one trial's positions, no loop.
- argument named `trials_data` / `trials` → list of tuples, expect a loop.

That single rule will tell you, before you read the body, whether a
function plots one trial or many.

---

## 9. Package layout, and why `import squircle...` can fail

Both modules start with:

```python
import squircle.tools as st
from squircle.design import well_locations_cm, circle_radius_cm, ...
```

`squircle` is the **package** (a folder), and `design` is a sibling module
inside it. For these imports to work, Python must find a folder structured
like:

```
some_folder/                 <-- this folder must be on sys.path
    squircle/
        __init__.py          <-- can be empty; must exist
        tools.py
        visualization.py
        design.py             <-- NOT in this GitHub repo; must be supplied
```

This repo contains the loose `.py` files but **not** the `squircle/`
folder and **not** `design.py`. That is why `SquircleAnalysis.py` does
`sys.path.append(r"C:\Users\astcu\...")` at the top — that line adds the
parent of the `squircle/` folder so the imports resolve on the lab laptop.

So when something fails with `ModuleNotFoundError: No module named
'squircle'` or `No module named 'squircle.design'`, it is never a code
bug. It is one of:
1. `sys.path` doesn't point at the parent of the `squircle/` folder, or
2. the files aren't actually inside a folder named `squircle/`, or
3. `design.py` is missing from that folder.

Fix the folder, not the code. This is worth knowing cold because it is the
single most likely thing to go wrong when you hand the files to the
interns on a different laptop.

---

## 10. A routine for "explain this code to the interns"

When you are standing in front of them with a function open, use this
order — it is the order that minimises confusion:

1. **Say what it produces.** "This function draws the rat's path for one
   trial." (one sentence, no code).
2. **Point at the inputs.** "It needs the position data — that's `df` —
   the number of the rewarded well, and whether the arena is a circle or
   square."
3. **Point at the one `plt.plot` line.** "This is the actual path: x
   against y." Resist explaining every line; explain the line that makes
   the picture.
4. **Run it.** Show the output.
5. **Change one input, re-run.** Change `env` from "Circle" to "Square",
   or feed a different trial's `df`, and show the picture change. This is
   the supervisor's whole method, and it works because the inputs were
   chosen to be visible.

If a question goes deeper than you can answer on the spot, the honest and
correct move is: "Good question — let me check the code and get back to
you." You are teaching them how to *use* and *explore* the code, not how
to author it. Keep the session at that level.

---

## 11. Quick reference: where things are

| You want to...                         | Look at                                |
|----------------------------------------|----------------------------------------|
| understand the trial tuple             | `tools.py`, the `trial_info = (...)` block (~line 650) |
| filter trials                          | `load_trials()` in `tools.py` (~line 672) |
| see how a single-trial plot works      | `plot_trajectory()` in `visualization.py` (~line 99) |
| see how a many-trial plot works        | `plot_multiple_trajectories()` (~line 248) |
| see the boolean colour toggle         | `plot_training_and_probes()` (~line 367) |
| see stacked bars                       | `plot_trial_completion()` (~line 581) |
| understand efficiency                  | `calculate_efficiency()` in `tools.py` (~line 769) |
| see a real usage example               | `SquircleAnalysis.py` and `Goal directedness instructions.py` |
| know the physical constants            | `design.py` (not in repo; on the laptop) |

Keep this guide open in a second tab while you work. The two things to
memorise are the **depth of the JSON** (animals → sessions → trials) and
**that a trial is a tuple where index 15 is the position DataFrame**.
Everything else you can look up.

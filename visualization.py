# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 14:23:35 2025

@author: mencia
"""

#%% Imports

from collections import defaultdict

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
from matplotlib.legend_handler import HandlerTuple
from collections import defaultdict
import seaborn as sns
import os
import squircle.tools as st
from io import StringIO
working_folder = r'C:\Users\astcu\OneDrive\Documents\2025-2026\Neurobiologie\Internship'
from squircle.design import environments, fs, circle_radius_cm, square_side_cm, well_locations_cm

#%% Definitions

standard_figsize = (8, 8)
default_colormap = plt.get_cmap('gist_rainbow')
square_pad = 7 # TODO check if padding needs to be adjusted


def _ideal_distance(df, rewarded_well):
    """
    Straight-line distance (cm) from a trial's start position to the reward well.

    This is the denominator inside calculate_efficiency. Trials that start very
    close to the reward give a near-zero denominator and therefore an inflated
    efficiency that reflects geometry rather than behaviour, so plotting
    functions use this to screen such trials out.
    """
    start_x, start_y = df.iloc[0]["x"], df.iloc[0]["y"]
    reward_x, reward_y = well_locations_cm[rewarded_well - 1]
    return np.sqrt((reward_x - start_x)**2 + (reward_y - start_y)**2)

def _is_completed(trial):
    """True if the trial's logged Completed status marks it as a success."""
    status = str(trial[17]).strip().lower()
    return status in ('completed', 'complete', '1', 'true')

#%% Single trial analysis

def plot_distance_to_reward(df, rewarded_well):
    """
    Plots the distance from the animal to the rewarded well over time.
    This function calculates the Euclidean distance to the rewarded well
    for each frame and plots it as a function of time.

    Parameters
    ----------
    df : DataFrame
        Tracking data in cm with 'x' and 'y' columns.
    rewarded_well : Int
        Index (1-based) of the rewarded well.

    Returns
    -------
    fig : Figure
        Matplotlib Figure object containing the produced figure.

    """

    # Get rewarded well coordinates (convert 1-based index to 0-based)
    reward_x, reward_y = well_locations_cm[int(rewarded_well - 1)]

    # Time in seconds based on sampling rate
    time = np.arange(len(df)) / fs  # fs = frames per second

    # Distance to the reward location
    distance_to_reward = np.sqrt((df["x"] - reward_x) ** 2 + (df["y"] - reward_y) ** 2)

    sns.set_style("white")

    fig = plt.figure(figsize=(8, 5))
    sns.lineplot(x=time,
                 y=distance_to_reward,
                 color="b")

    plt.xlabel("Time (seconds)")
    plt.ylabel("Distance to Reward (cm)")
    plt.title("Distance to Reward Over Time")
    plt.ylim(0, 60)
    plt.grid(False)
    plt.show()
    
    return fig

def plot_trajectory(df, rewarded_well, env):
    """
    Plots the animal's trajectory during a trial, including start location,
    rewarded well, unrewarded wells, and environment boundary.

    Parameters
    ----------
    df : DataFrame
        Tracking data in cm with 'x' and 'y' columns.
    rewarded_well : Int
        Index (1-based) of the rewarded well.
    env : String
        Environment type; either 'Circle' or 'Square'.

    Returns
    -------
    fig : Figure.
        Matplotlib Figure object containing the produced figure.

    """
    
    if env not in environments:
        raise NameError('Environment not correct!')

    # Coordinates of reward and starting position
    reward_x, reward_y = well_locations_cm[rewarded_well - 1]
    start_x, start_y = df.iloc[0]

    fig = plt.figure(figsize=standard_figsize)

    # Plot all wells, rewarded well, and start position
    plt.scatter(*zip(*well_locations_cm), color="gray", label="Wells")
    plt.scatter(reward_x, reward_y, color="red", label="Rewarded Well")
    plt.scatter(start_x,
                start_y,
                color="lime",
                edgecolor="black",
                linewidth=1.5,
                marker="*",
                s=200,
                label="Start Position")

    # Plot trajectory line
    plt.plot(df["x"], df["y"], color="blue", alpha=0.7, label="Trajectory")

    # Draw environment boundary
    if env == "Circle":
        circle = plt.Circle((0, 0),
                            circle_radius_cm,
                            color="black",
                            fill=False,
                            linewidth=1.5)
        plt.gca().add_patch(circle)

    elif env == "Square":
        square_side = square_side_cm + square_pad
        half_side = square_side / 2
        square = plt.Rectangle((-half_side, -half_side),
                               square_side, square_side,
                               color="black",
                               fill=False,
                               linewidth=1.5)
        plt.gca().add_patch(square)

    plt.xlabel("X Position (cm)")
    plt.ylabel("Y Position (cm)")
    plt.title("Trajectory")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.xlim(-40, 40)
    plt.ylim(-40, 40)
    plt.gca().set_aspect("equal")
    plt.grid(False)
    plt.tight_layout()
    plt.show()
    
    return fig

#%% Session analysis

def plot_trial_durations(trials_data, colors=None):
    """
    Plots the duration of each trial based on the number of position samples.

    Parameters
    ----------
    selected_trials : List
        List of trial tuples structured from load_trials().
    colors : List, optional
        List of RGB colors used to plot each trial. The default is None.

    Returns
    -------
    fig : Figure
        Matplotlib Figure object containing the produced figure.
    colors : List
        List of RGB colors used to plot each trial.

    """

    trial_numbers = []
    trial_durations = []

    for i, trial in enumerate(trials_data):
        if len(trial) != 17:
            raise ValueError(f"Expected trial with 15 elements, got {len(trial)}")

        (
            name, dob, sex, session_type, env, date, age, rotation, rewarded_well,
            session_center, trial_number, start_frame, end_frame,
            frame_offset_x, frame_offset_y, df, baited
        ) = trial

        if df is None or df.empty:
            continue

        # Duration in seconds = number of frames / frames per second
        duration = len(df) / fs
        trial_numbers.append(i + 1)
        trial_durations.append(duration)

    if colors is None:
        num_trials = len(trial_numbers)
        colors = [default_colormap(i / num_trials) for i in range(num_trials)]

    sns.set_style("white")
    fig = plt.figure(figsize=(8, 5))

    # Scatter plot of trial durations
    sns.scatterplot(
        x=trial_numbers,
        y=trial_durations,
        hue=trial_numbers,
        palette=colors,
        legend=False,
        s=100,
        alpha=0.7
    )

    plt.xlabel("Trial Index")
    plt.ylabel("Trial Duration (seconds)")
    plt.title("Duration of the Trials")
    plt.ylim(0, max(trial_durations) + 1.5)
    plt.xticks(trial_numbers)
    plt.grid(False)
    plt.tight_layout()
    plt.show()
    
    return fig, colors

def plot_multiple_trajectories(trials_data, colors=None):
    """
    Plots trajectories of multiple trials for a single subject with color-coded paths.

    Parameters
    ----------
    trials_data : List
        List of trial tuples (structured from load_trials()).
    env : String
        Environment type; either 'Circle' or 'Square'.
    colors : List, optional
        List of RGB colors used to plot each trial. The default is None.
    
    Returns
    -------
    fig : Figure
        Matplotlib Figure object containing the produced figure.
    colors : List
        List of RGB colors used to plot each trial.

    """

    num_trials = len(trials_data)
    if num_trials == 0:
        raise ValueError("No trials selected for plotting.")

    # Check that all trials have the same rewarded well
    rewarded_wells = {trial[7] for trial in trials_data}
    if len(rewarded_wells) > 1:
        raise ValueError('Selected trials have different rewarded wells.')
        
    # Determine environment type from the first trial available
    _, _, _, env, *_ = trials_data[0]

    rewarded_well = next(iter(rewarded_wells))
    reward_x, reward_y = well_locations_cm[rewarded_well - 1]
    
    if colors is None:
        colors = [default_colormap(i / num_trials) for i in range(num_trials)]

    fig = plt.figure(figsize=standard_figsize)
    
    # Plot all wells, highlight rewarded well
    plt.scatter(*zip(*well_locations_cm), color="gray", label="Wells")
    plt.scatter(reward_x, reward_y, color="red", label="Rewarded Well")

    # Plot each trial's trajectory and start position
    for i, trial in enumerate(trials_data):
        (
            name, dob, sex, session_type, env, date, age, rotation, rewarded_well,
            session_center, trial_number, start_frame, end_frame,
            frame_offset_x, frame_offset_y, df, baited
        ) = trial

        if df.empty:
            continue

        color = colors[i]
        start_x, start_y = df.iloc[0]
        plt.scatter(start_x,
                    start_y,
                    color=color,
                    edgecolor="black",
                    linewidth=1.5,
                    marker="*",
                    s=200,
                    label="Start Position" if i == 0 else "")
        plt.plot(df["x"],
                 df["y"],
                 color=color,
                 alpha=0.7,
                 label=f"Trial {trial_number}")

    # Draw environment boundary
    ax = plt.gca()
    
    if env == "Circle":
        circle = plt.Circle((0, 0), circle_radius_cm, color="black", fill=False, linewidth=1.5)
        ax.add_patch(circle)

    
    elif env == "Square":
        square_side = square_side_cm + square_pad
        half_side = square_side / 2
        square = plt.Rectangle((-half_side, -half_side),
                               square_side,
                               square_side,
                               color="black",
                               fill=False,
                               linewidth=1.5)
        ax.add_patch(square)
    
    else:
        raise ValueError('Environment type not correct!')

    plt.xlabel("X Position (cm)")
    plt.ylabel("Y Position (cm)")
    plt.title("Training Trials")

    # Clean up legend to only show one consistent "Start Position" marker
    handles, labels = ax.get_legend_handles_labels()
    for i, label in enumerate(labels):
        if label == "Start Position":
            handles[i] = plt.scatter([], [], color="gray", marker="*", s=200)

    plt.legend(handles,
               labels,
               loc="center left",
               bbox_to_anchor=(1, 0.5),
               fontsize=12)

    plt.xlim(-40, 40)
    plt.ylim(-40, 40)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()

    return fig, colors

def plot_training_and_probes(trials_data,
                             probe_trials, 
                             colors=None,
                             separate_color_scale=True):
    """
    Plots all training and probe trial trajectories on the same figure.

    Parameters
    ----------
    trials_data : List
        List of training trial tuples (structured from load_trials()).
    probe_trials : List
        List of probe trial tuples (structured from load_trials()).
    colors : List, optional
        List of RGB colors used to plot each trial. The default is None.
    Seperate color scale : bool, optional
        True means new version - two color scales 
    Returns
    -------
    fig : Figure
        Matplotlib Figure object containing the produced figure.
    colors : List
        List of RGB colors used to plot each trial.

    """
    num_train = len(trials_data)
    num_probe = len(probe_trials)
    num_trials =  num_train + num_probe
    
    if num_trials == 0:
        raise ValueError('No trials selected for plotting.')

    if colors is None:
        if separate_color_scale:
            train_map = plt.cm.YlGn  
            probe_map = plt.cm.RdPu   
            
            # Create two lists and join them
            train_colors = [train_map(x) for x in np.linspace(0.4, 0.9, num_train)] if num_train > 0 else []
            probe_colors = [probe_map(x) for x in np.linspace(0.4, 0.9, num_probe)] if num_probe > 0 else []
            colors = train_colors + probe_colors
            
        else:
            combined_map = plt.cm.turbo
            colors = [combined_map(x) for x in np.linspace(0, 1, num_trials)]

    fig = plt.figure(figsize=standard_figsize)

    # Plot static well locations
    well_x = [x for x, y in well_locations_cm]
    well_y = [y for x, y in well_locations_cm]
    plt.scatter(well_x, well_y, color="gray", label="Wells", s=50)

    # Determine environment type from the first trial available
    _, _, _, env, *_ = trials_data[0]

    # Draw training trials
    for i, trial in enumerate(trials_data):
        (_, _,sex, _, _, _, _, rotation, rewarded_well, _, trial_number, _, _, _, _, df, baited) = trial
        reward_x, reward_y = well_locations_cm[rewarded_well - 1]
        plt.scatter([reward_x],
                    [reward_y],   
                    color="red",
                    s=150,
                    label="Rewarded Well" if i == 0 else "")  
        plt.plot(df["x"],
                 df["y"],
                 color=colors[i],
                 alpha=0.7,
                 label=f"Training {trial_number}")

        if not df.empty:
            start_x, start_y = df.iloc[0]
            plt.scatter([start_x],
                        [start_y],
                        color=colors[i],
                        edgecolors="black",
                        linewidths=1.5,
                        marker="*",
                        s=200,
                        label="Start Position" if i == 0 else "")

    # Draw probe trials
    for j, trial in enumerate(probe_trials):
        (_, _, sex, _, _, _, _, rotation, rewarded_well, _, trial_number, _, _, _, _, df, baited) = trial
        probe_index = len(trials_data) + j
        linestyle = "--" if rotation == 1 else "-"
        linewidth = 1.5 if rotation == 0 else 1.5
        label = "Probe Trial Rotation" if rotation == 1 else "Probe Trial"
        plt.plot(df["x"],
                 df["y"],
                 linestyle=linestyle,
                 linewidth=linewidth,
                 color=colors[probe_index],
                 label=label)

        if not df.empty:
            start_x, start_y = df.iloc[0]
            plt.scatter([start_x],
                        [start_y],
                        color=colors[probe_index],
                        edgecolors="black",
                        linewidths=1.5,
                        marker="*", s=200)

        if rotation == 1:
            reward_x, reward_y = well_locations_cm[rewarded_well - 1]
            plt.scatter([reward_x], [reward_y], color="blue", s=150, label="Rewarded Well Rotation")

    # Draw boundary of the environment
    ax = plt.gca()
    if env == "Circle":
        circle = plt.Circle((0, 0),
                            circle_radius_cm,
                            color="black",
                            fill=False,
                            linewidth=1.5)
        ax.add_patch(circle)

    elif env == "Square":
        pad = 7
        square_side = square_side_cm + pad
        half_side = square_side / 2
        square = plt.Rectangle((-half_side, -half_side),
                               square_side,
                               square_side,
                               color="black",
                               fill=False,
                               linewidth=1.5)
        ax.add_patch(square)

    else:
        raise ValueError('Environment type not correct!')

    # Build a clean and consistent legend
    # Define Static Markers/legend entries that never change 
    legend_elements = [
    
        # Use Line2D with linestyle="None" for scatter-like markers in legends
        mlines.Line2D([], [],
                      color="none",
                      marker="o", 
                      markerfacecolor="gray", 
                      markersize=10, 
                      label="Wells"),
        mlines.Line2D([], [],
                      color="none",
                      marker="*",
                      markeredgecolor="black", 
                      markerfacecolor="gray",
                      markersize=12,
                      label="Start Position"),
        mlines.Line2D([], [],
                      color="none",
                      marker="o",
                      markerfacecolor="red", 
                      markersize=10,
                      label="Target Well"),
        ]

# Add Grouped Trial Types (instead of every individual trial)
    # Training Group
    if len(trials_data) > 0:
        legend_elements.append(
            mlines.Line2D([], [],
                          color=colors[0],
                          linestyle="-",
                          lw=2,
                          label="Training Trials"))

    # Probe Group (Standard)
    if len(probe_trials) > 0:
        legend_elements.append(
            mlines.Line2D([], [],
                          color=colors[len(trials_data)],
                          linestyle="-",
                          lw=2,
                          label="Probe Trials"))
        
        # Check if any probe trials were rotated to add the dashed line to legend
        if any(trial[6] == 1 for trial in probe_trials):
            legend_elements.append(
                mlines.Line2D([], [],
                              color=colors[-1],
                              linestyle="--",
                              lw=2,
                              label="Probe (Rotated)"))
            legend_elements.append(
                mlines.Line2D([], [],
                              color="none",
                              marker="o",
                              markerfacecolor="blue",
                              markersize=10,
                              label="Rotated Target"))

    # Final plot details
    plt.legend(handles=legend_elements,
               loc="center left",
               bbox_to_anchor=(1, 0.5),
               fontsize=12)
    plt.xlabel("X Position (cm)")
    plt.ylabel("Y Position (cm)")
    plt.title("Training and Probe Trials")
    plt.xlim(-45, 45)
    plt.ylim(-45, 45)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()


    return fig, colors

#%% Population analysis

def plot_trial_completion(filepath, subjects):
    """
    Plot the percentage of completed vs failed probe trials per subject.
    This function computes the number of completed and failed probe trials for
    each subject, expressed as a percentage. It then plots the breakdown
    using a grouped barplot. A trial is considered completed if both start
    and end frames are present; otherwise, it is marked as failed.

    Parameters
    ----------
    filepath : String
        Path to the JSON file for the litter.
    subjects : List
        List of subject names to plot.

    Returns
    -------
    fig : Figure
        Matplotlib Figure object containing the produced figure.

    """

    with open(filepath, 'r') as json_file:
        data = json.load(json_file)
        
    completed_counts = []
    failed_counts = []

    for subject in subjects:
        # Find subject entry from dataset
        subject_data = next((s for s in data if s["Name"] == subject), None)
        if not subject_data:
            # No data for this subject
            completed_counts.append(0)
            failed_counts.append(0)
            continue

        # Collect all probe trials across sessions
        probe_trials = []
        for session in subject_data["Sessions"]:
            if session["Type"] == "Probe":
                probe_trials.extend(session["Trials"])

        total_trials = len(probe_trials)
        if total_trials == 0:
            completed_counts.append(0)
            failed_counts.append(0)
            continue

        # Count as completed if start and end frames are not NaN
        completed = sum(
            1 for trial in probe_trials
            if not (pd.isna(trial["Start frame"]) or pd.isna(trial["End frame"]))
        )
        failed = total_trials - completed

        # Convert to percentages
        completed_counts.append((completed / total_trials) * 100)
        failed_counts.append((failed / total_trials) * 100)

    # Create grouped barplot
    # Create figure
    fig = plt.figure(figsize=(8, 5))
    x = np.arange(len(subjects))

    # Plot the first layer
    plt.bar(x, completed_counts,
            label="Completed",
            color="tab:blue")

    # Plot the second layer on top of the first
    plt.bar(x, failed_counts,
            bottom=completed_counts, # This is the "Stacking" logic
            label="Failed",
            color="tab:red")

    # FIX: Map the numbers in 'x' to the names in 'subjects'
    plt.xticks(x, subjects, rotation=0) 
    plt.xlabel("Subjects")
    plt.ylabel("Percentage of Trials")
    plt.title("Percentage of Completed and Failed Probe Trials per Subject")
    plt.legend(title='Trial Type', loc='center left', bbox_to_anchor = (1, 0.8))
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout() # Added to prevent label cutoff
    plt.show()

    
    return fig

def plot_efficiency_ratio_per_subject(trials):
    """
    Plots the trajectory efficiency over trials for each subject,
    with each line representing a different age.

    Parameters:
    - trials (list): List of trial tuples.
    - circle_radius_cm (float): Circle radius for computing distances.
    - square_side_cm (float): Square side for computing distances.
    """

    subject_efficiency = defaultdict(lambda: defaultdict(list))
    subject_environment = {}  

    # Compute efficiencies
    for trial in trials:
        subject, dob, sex, session_type, env, date, age, rotation, rewarded_well, session_center, trial_num, start_frame, end_frame, fx, fy, df, baited = trial
        if df is None or df.empty:
            continue
        if subject not in subject_environment:
            subject_environment[subject] = env  

        efficiency = st.calculate_efficiency(df, rewarded_well)
        subject_efficiency[subject][age].append((trial_num, efficiency))

    figs = []
    # Plot per subject
    for subject, age_trials in subject_efficiency.items():
        fig = plt.figure(figsize=(10, 6))

        all_ages = sorted(age_trials.keys())
        norm = mcolors.Normalize(vmin=min(all_ages), vmax=max(all_ages))
        cmap = mcolors.LinearSegmentedColormap.from_list("orange_purple", ["#ff7f0e", "#9467bd"])

        for age, trial_data in sorted(age_trials.items()):
            trial_data.sort()
            trial_nums = [t[0] for t in trial_data]
            efficiencies = [t[1] for t in trial_data]
            color = cmap(norm(age))
            plt.plot(trial_nums, efficiencies, marker="o", label=f"Age {age}", color=color)

        plt.axhline(1.0,
                    linestyle="--",
                    color="red",
                    label="Perfect efficiency")
        plt.xlabel("Trial Number")
        plt.ylabel("Efficiency (Ideal / Real Path)")
        plt.title(f"Trajectory Efficiency by Trial — Subject {subject} ({subject_environment[subject]})")
        all_trial_nums = [t[0] for age_vals in age_trials.values() for t in age_vals]
        plt.xticks(sorted(set(all_trial_nums)))
        plt.ylim(0, 20)
        plt.legend(title="Age", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.show()
        figs.append(fig)
        
    return figs

def make_boxplot(df, labels):
    """
    Helper function for boxplot_quantiles().
    Produces the figure using the data in df.

    Parameters
    ----------
    df : DataFrame
        Pandas DataFrame.
    labels : Dict
        Dictionary containing the plot labels and their order.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    label_keys = list(labels.keys())

    sns.set_style('white')
    fig = plt.figure(figsize=(12, 6))
    
    unique_quartiles = list(df['Quartile'].unique())

    def quartile_sort_key(q):
        # try:
        #     env_part, label_part = q.split(' - ', 1)
        # except ValueError:
        #     env_part, label_part = q, ''
        # return (env_part, labels.get(label_part, 99))
        
        if " - " in q:
            env_part, label_part = q.split(" - ", 1)
        else:
            env_part, label_part = "", q
        return (env_part, labels.get(label_part, 99))


    hue_order = sorted(unique_quartiles, key=quartile_sort_key)

    sns.boxplot(
        data=df,
        x='Age',
        y='Efficiency',
        hue='Quartile',
        hue_order=hue_order,
        palette='Set2'
    )

# Create display names for title - updated 
    display_names = {
        'T1': 'Training Q1',
        'T2': 'Training Q2',
        'T3': 'Training Q3',
        'T4': 'Training Q4',
        'P1': 'Probe Q1',
        'P2': 'Probe Q2'}
# Which title to pick for place 1 and 2 - updated      
    title_part1 = display_names.get(label_keys[0],
                                    label_keys[0]) #second one is backup
    title_part2 = display_names.get(label_keys[1],
                                    label_keys[1]) #second one is backup

    plt.xlabel('Age (days)')
    plt.ylabel('Path efficiency')
    plt.title(f"{title_part1} vs {title_part2}")
    plt.ylim(0, 1.1)
    plt.legend(title='Quantiles')
    plt.tight_layout()
    plt.show()
    
    return fig

# FIXME there are differences in some datapoints when env_split = True or False that shouldn't be there!
def boxplot_quantiles(population_trials,
                      q1,
                      q2,
                      min_age=None,
                      max_age=None,
                      env_split=True):
    """
    Produces a figure showing the path efficiency distribution for the
    specified quantiles of training and probe sessions for all subjects
    across age.
    The quantiles are quartiles for training sessions and halves for probe
    sessions, so that each quantile is 5 trials long.
    The quantiles are specified as 'xn', where 'x' is the session type
    (training or probe) and 'n' the quantile number.
    
    Example
    ----------
    fig = boxplot_quantiles(trial_tuples, 't4', 'p2')
        Plots the last 5 trials of training sessions vs the last 5 trials of
        probe sessions for all trials in trial_tuples.
    
    Parameters
    ----------
    population_trials : List.
        List of 15-element tuples.
    q1 : String
        What quantile to plot first.
    q2 : String
        What quantile to plot second.
    min_age : Int, optional
        Lower cutoff for subject age to plot. The default is None.
    max_age : Int, optional
        Higher cutoff for subject age to plot. The default is None.
    env_split : Bool, optional
        Whether to split boxplots for environments. The default is True.
    Returns
    -------
    fig : Figure
        Matplotlib Figure object.

    """
    # t1 is q1 for training
    # p1 is q1 for probe
    
    quant1 = list(q1)
    quant2 = list(q2)
    
    if len(quant1) > 2:
        raise ValueError('Parameter q1 is not correct!')
        
    elif len(quant2) > 2:
        raise ValueError('Parameter q2 is not correct!')

    quant1_kind = quant1[0]
    quant2_kind = quant2[0]
    
    session_trials = defaultdict(list)
    data_for_plot = []
    
    if env_split:
        # Group all trials by (subject, environment, date)
        for trial in population_trials:
            session_key = (trial[0], trial[4], trial [16], trial[5])
            session_trials[session_key].append(trial)
    
        for session_key, trials in session_trials.items():
            subject, environment, baited, date_str = session_key
            
            baited_label = 'Baited' if baited == 1 else 'Unbaited'
            
            quant1_trials = []
            quant2_trials = []

            # TRAINING
            if quant1_kind == 't' or quant2_kind == 't':
            
                trials_to_get = [t for t in trials if t[3] == 'Training']
                trials_to_get = sorted(trials_to_get, key=lambda t: t[9])
                
                if quant1_kind == 't':
                    qtile = int(quant1[1])
                    
                    if qtile == 1:
                        quant1_trials = trials_to_get[:5]  # first 5 trials only
                    elif qtile == 2:
                        quant1_trials = trials_to_get[5:10]
                    elif qtile == 3:
                        quant1_trials = trials_to_get[10:15]
                    elif qtile == 4:
                        quant1_trials = trials_to_get[-5:]  # last 5 trials only
                    else:
                        raise ValueError('Quantile number for q1 not correct!')
                
                if quant2_kind == 't':
                    qtile = int(quant2[1])
                    
                    if qtile == 1:
                        quant2_trials = trials_to_get[:5]  # first 5 trials only
                    elif qtile == 2:
                        quant2_trials = trials_to_get[5:10]
                    elif qtile == 3:
                        quant2_trials = trials_to_get[10:15]
                    elif qtile == 4:
                        quant2_trials = trials_to_get[-5:]  # last 5 trials only
                    else:
                        raise ValueError('Quantile number for q2 not correct!')
    
            # TODO implement handling of rotation
            # PROBE (rotation == 0)
            if quant1_kind == 'p' or quant2_kind == 'p':
    
                trials_to_get = [t for t in trials if t[3] == 'Probe' and t[7] == 0]
                trials_to_get = sorted(trials_to_get, key=lambda t: t[9])
                
                if quant1_kind == 'p':
                    qtile = int(quant1[1])
                    
                    if qtile == 1:
                        quant1_trials = trials_to_get[:5]  # first 5 trials only
                    elif qtile == 2:
                        quant1_trials = trials_to_get[5:10] # last 5 trials only
                    else:
                        raise ValueError('Quantile number for q1 not correct!')
                
                if quant2_kind == 'p':
                    qtile = int(quant2[1])
                    
                    if qtile == 1:
                        quant2_trials = trials_to_get[:5]  # first 5 trials only
                    elif qtile == 2:
                        quant2_trials = trials_to_get[5:10] # last 5 trials only
                    else:
                        raise ValueError('Quantile number for q2 not correct!')
    
            # Make legend
            quartile_sets = [(quant1_trials, q1.upper(),
                              'Training' if quant1_kind == 't' else 'Probe'),
                             (quant2_trials, q2.upper(),
                              'Training' if quant2_kind == 't' else 'Probe'),]
    
            for trial_set, label, source in quartile_sets:
                if len(trial_set) <= 0:
                    continue
    
                efficiencies = []
                for trial in trial_set:
                    age = trial[6]
    
                    # Apply age filters
                    if min_age is not None and age < min_age:
                        continue
                    if max_age is not None and age > max_age:
                        continue
    
                    start_frame = trial[11]
                    end_frame = trial[12]
                    df = trial[15]
                    rewarded_well = trial[8]
    
                    if pd.isna(start_frame) or pd.isna(end_frame) or df is None or df.empty:
                        continue
    
                    efficiency = st.calculate_efficiency(df, rewarded_well)
                    efficiencies.append(efficiency)
    
                if efficiencies:
                    # Age logic: shift probe trials to previous training age
                    raw_age = trial_set[0][6]
                    adjusted_age = raw_age - 1 if source == 'Probe' else raw_age
    
                    data_for_plot.append({
                        'Age': adjusted_age,
                        'Efficiency': np.mean(efficiencies),
                        'Environment': environment,
                        'Quartile': f'{environment} ({baited_label}) - {label}'
                    })
    else:
        # Group all trials by (subject, date)
        for trial in population_trials:
            session_key = (trial[0], trial[16], trial[5])
            session_trials[session_key].append(trial)
    
        for session_key, trials in session_trials.items():
            subject, baited, date_str = session_key
            baited_label = 'Baited' if baited == 1 else 'Unbaited'
            
            quant1_trials = []
            quant2_trials = []
            
            # TRAINING
            if quant1_kind == 't' or quant2_kind == 't':
            
                trials_to_get = [t for t in trials if t[3] == 'Training']
                trials_to_get = sorted(trials_to_get, key=lambda t: t[9])
                
                if quant1_kind == 't':
                    qtile = int(quant1[1])
                    
                    if qtile == 1:
                        quant1_trials = trials_to_get[:5]  # first 5 trials only
                    elif qtile == 2:
                        quant1_trials = trials_to_get[5:10]
                    elif qtile == 3:
                        quant1_trials = trials_to_get[10:15]
                    elif qtile == 4:
                        quant1_trials = trials_to_get[-5:]  # last 5 trials only
                    else:
                        raise ValueError('Quantile number for q1 not correct!')
                
                if quant2_kind == 't':
                    qtile = int(quant2[1])
                    
                    if qtile == 1:
                        quant2_trials = trials_to_get[:5]  # first 5 trials only
                    elif qtile == 2:
                        quant2_trials = trials_to_get[5:10]
                    elif qtile == 3:
                        quant2_trials = trials_to_get[10:15]
                    elif qtile == 4:
                        quant2_trials = trials_to_get[-5:]  # last 5 trials only
                    else:
                        raise ValueError('Quantile number for q2 not correct!')
    
            # TODO implement handling of rotation
            # PROBE (rotation == 0)
            if quant1_kind == 'p' or quant2_kind == 'p':
    
                trials_to_get = [t for t in trials if t[3] == 'Probe' and t[7] == 0]
                trials_to_get = sorted(trials_to_get, key=lambda t: t[9])
                
                if quant1_kind == 'p':
                    qtile = int(quant1[1])
                    
                    if qtile == 1:
                        quant1_trials = trials_to_get[:5]  # first 5 trials only
                    elif qtile == 2:
                        quant1_trials = trials_to_get[5:10] # last 5 trials only
                    else:
                        raise ValueError('Quantile number for q1 not correct!')
                
                if quant2_kind == 'p':
                    qtile = int(quant2[1])
                    
                    if qtile == 1:
                        quant2_trials = trials_to_get[:5]  # first 5 trials only
                    elif qtile == 2:
                        quant2_trials = trials_to_get[5:10] # last 5 trials only
                    else:
                        raise ValueError('Quantile number for q2 not correct!')
    
            # Make legend
            quartile_sets = [(quant1_trials, q1.upper(),
                              'Training' if quant1_kind == 't' else 'Probe'),
                             (quant2_trials, q2.upper(),
                              'Training' if quant2_kind == 't' else 'Probe'),]
    
            for trial_set, label, source in quartile_sets:
                if len(trial_set) <= 0:
                    continue
    
                efficiencies = []
                for trial in trial_set:
                    age = trial[6]
    
                    # Apply age filters
                    if min_age is not None and age < min_age:
                        continue
                    if max_age is not None and age > max_age:
                        continue
    
                    start_frame = trial[11]
                    end_frame = trial[12]
                    df = trial[15]
                    rewarded_well = trial[8]
                    if pd.isna(start_frame) or pd.isna(end_frame) or df is None or df.empty:
                        continue
    
                    efficiency = st.calculate_efficiency(df, rewarded_well)
                    efficiencies.append(efficiency)
    
                if efficiencies:
                    # Age logic: shift probe trials to previous training age
                    raw_age = trial_set[0][6]
                    adjusted_age = raw_age - 1 if source == 'Probe' else raw_age
    
                    data_for_plot.append({
                        'Age': adjusted_age,
                        'Efficiency': np.mean(efficiencies),
                        'Environment': 'All', #added this in
                        'Quartile': label
                    })
    
    df = pd.DataFrame(data_for_plot)
    
    # Ordering the plot labels
    label_order_map = {
        q1.upper(): 1,
        q2.upper(): 2
        }
    
    fig = make_boxplot(df, label_order_map)
    plt.ylim(0, 1.1)
    
    return fig

# --- Function 1: Summary boxplot for all litters NP

#%% Failure rate from the processed JSON, for overlaying on retention figures
def failure_rate_by_age(
    working_folder,
    litters=(9, 10, 11),
    first_n=10,
    exclude_rotation=True
):
    """
    Percentage of failed trials among the first `first_n` training and probe
    trials of each session, aggregated by postnatal age, read from the
    processed JSON using the logged 'Completed' flag.

    Returns
    -------
    dict: {'Training': {age: pct}, 'Probe': {age: pct}, '_counts': {...}}
    """
    counts = {'Training': defaultdict(lambda: [0, 0]),
              'Probe':    defaultdict(lambda: [0, 0])}

    for litter in litters:
        path = os.path.join(working_folder, f"litter_{litter}_processed_cm.json")
        if not os.path.exists(path):
            continue
        with open(path, 'r') as f:
            data = json.load(f)

        for subject in data:
            for s in subject['Sessions']:
                stype = s['Type']
                if stype not in ('Training', 'Probe'):
                    continue
                if stype == 'Probe' and exclude_rotation and s['Rotation'] == 1:
                    continue
                age = s['Age']
                for trial in s['Trials']:
                    if trial['Number'] > first_n:
                        continue
                    status = str(trial.get('Completed', 'Completed')).strip().lower()
                    completed = status in ('completed', 'complete', '1', 'true')
                    counts[stype][age][1] += 1
                    if not completed:
                        counts[stype][age][0] += 1

    pct = {t: {a: 100 * c[0] / c[1] for a, c in counts[t].items() if c[1]}
           for t in counts}
    pct['_counts'] = counts
    return pct


def plot_failed_trials_by_band(
    working_folder,
    litters=(9, 10, 11),
    age_bands=((17, 20, 'Pre-weaning (P17-20)'),
               (21, 24, 'Peri-weaning (P21-24)'),
               (25, 32, 'Post-weaning (P25-32)')),
    exclude_rotation=False,
    annotate_rate=True,
    figsize=(9, 6),
    title=None
):
    """
    Grouped bar chart of the NUMBER of failed trials per developmental age band,
    split by baited vs unbaited condition. Pools every trial of every
    Training/Probe session across the new-paradigm litters; each trial is banded
    by its own session's postnatal age (not the animal's entry cohort). A trial
    counts as failed when its logged 'Completed' flag is not a success value
    (same rule as `failure_rate_by_age` / `_is_completed`).

    Parameters
    ----------
    working_folder : str
        Folder holding the `litter_<n>_processed_cm.json` files.
    litters : tuple of int
        Litters to pool (default the new-paradigm 9, 10, 11).
    age_bands : tuple of (lo, hi, label)
        Inclusive postnatal-age bins.
    exclude_rotation : bool
        If True, drop rotation probe sessions (Rotation == 1). Default False so
        that literally all sessions are included.
    annotate_rate : bool
        If True, label each bar with 'n_failed / n_total (pct%)'.
    figsize : tuple
    title : str or None

    Returns
    -------
    fig : matplotlib Figure
    df_out : DataFrame, one row per band x condition with failed/total/rate.
    """
    band_labels = [lbl for _, _, lbl in age_bands]

    def _band_of(age):
        for lo, hi, lbl in age_bands:
            if lo <= age <= hi:
                return lbl
        return None

    # counts[band][baited] = [n_failed, n_total]
    counts = {lbl: {0: [0, 0], 1: [0, 0]} for lbl in band_labels}

    for litter in litters:
        path = os.path.join(working_folder, f"litter_{litter}_processed_cm.json")
        if not os.path.exists(path):
            print(f"Missing: {path}")
            continue
        with open(path, 'r') as f:
            data = json.load(f)

        for subject in data:
            for s in subject['Sessions']:
                if s['Type'] not in ('Training', 'Probe'):
                    continue
                if (s['Type'] == 'Probe' and exclude_rotation
                        and s.get('Rotation', 0) == 1):
                    continue
                band = _band_of(s['Age'])
                if band is None:
                    continue
                for trial in s['Trials']:
                    baited = int(trial.get('Baited', 0))
                    if baited not in (0, 1):
                        continue
                    status = str(trial.get('Completed', 'Completed')).strip().lower()
                    completed = status in ('completed', 'complete', '1', 'true')
                    counts[band][baited][1] += 1
                    if not completed:
                        counts[band][baited][0] += 1

    rows = []
    for lbl in band_labels:
        for baited in (0, 1):
            failed, total = counts[lbl][baited]
            rows.append({
                'Age Band': lbl,
                'Condition': 'Baited' if baited else 'Unbaited',
                'Failed': failed,
                'Total': total,
                'Failure rate (%)': 100 * failed / total if total else np.nan,
            })
    df_out = pd.DataFrame(rows)

    present = [lbl for lbl in band_labels
               if df_out.loc[df_out['Age Band'] == lbl, 'Total'].sum() > 0]
    if not present:
        print("No trials found for the requested litters/bands.")
        return None, df_out

    # --- grouped bar chart: failed-trial COUNTS ---
    cond_order = ['Unbaited', 'Baited']
    cond_colors = {'Unbaited': '#4C72B0', 'Baited': '#DD8452'}
    x = np.arange(len(present))
    width = 0.38

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=figsize)
    for i, cond in enumerate(cond_order):
        vals, rates, totals = [], [], []
        for lbl in present:
            r = df_out[(df_out['Age Band'] == lbl) & (df_out['Condition'] == cond)].iloc[0]
            vals.append(r['Failed'])
            rates.append(r['Failure rate (%)'])
            totals.append(r['Total'])
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=cond,
                      color=cond_colors[cond], edgecolor='black', linewidth=0.6)
        if annotate_rate:
            for b, v, rate, tot in zip(bars, vals, rates, totals):
                txt = f"{v}/{tot}\n({rate:.0f}%)" if tot else f"{v}"
                ax.annotate(txt, (b.get_x() + b.get_width() / 2, v),
                            ha='center', va='bottom', fontsize=8,
                            xytext=(0, 2), textcoords='offset points')

    ax.set_xticks(x)
    ax.set_xticklabels(present)
    ax.set_ylabel('Number of failed trials')
    ax.set_xlabel('Developmental stage (session age)')
    ax.set_title(title or
                 "Failed trials by age band and condition "
                 f"(litters {', '.join(map(str, litters))}, all sessions)")
    ax.legend(title='Condition')
    # a little headroom for the annotations
    ymax = df_out['Failed'].max()
    ax.set_ylim(0, ymax * 1.18 if ymax else 1)
    plt.tight_layout()
    plt.show()

    return fig, df_out


def plot_trials_stacked_by_band(
    working_folder,
    litters=(9, 10, 11),
    age_bands=((17, 20, 'Pre-weaning (P17-20)'),
               (21, 24, 'Peri-weaning (P21-24)'),
               (25, 32, 'Post-weaning (P25-32)')),
    exclude_rotation=False,
    figsize=(9, 6),
    title=None
):
    """
    Stacked bar chart of ALL run trials per developmental age band. Each bar's
    full height is the total number of trials run in that band; within the bar,
    the baited and unbaited conditions are separate colour hues, and each hue is
    split by shade into the FAILED portion (solid) and the COMPLETED portion
    (faded). This shows the failed slice against the full workload per band.

    Bottom-to-top stacking per bar (matches the two-hue / two-shade legend):
        unbaited failed  -> unbaited completed -> baited failed -> baited completed

    A trial counts as failed when its logged 'Completed' flag is not a success
    value (same rule as `failure_rate_by_age` / `_is_completed`). Each trial is
    banded by its own session's postnatal age.

    Parameters
    ----------
    working_folder : str
        Folder holding the `litter_<n>_processed_cm.json` files.
    litters : tuple of int
        Litters to pool (default the new-paradigm 9, 10, 11).
    age_bands : tuple of (lo, hi, label)
        Inclusive postnatal-age bins.
    exclude_rotation : bool
        If True, drop rotation probe sessions (Rotation == 1). Default False so
        that literally all sessions are included.
    figsize : tuple
    title : str or None

    Returns
    -------
    fig : matplotlib Figure
    df_out : DataFrame, one row per band x condition with failed/completed/total.
    """
    band_labels = [lbl for _, _, lbl in age_bands]

    def _band_of(age):
        for lo, hi, lbl in age_bands:
            if lo <= age <= hi:
                return lbl
        return None

    # counts[band][baited] = [n_failed, n_total]
    counts = {lbl: {0: [0, 0], 1: [0, 0]} for lbl in band_labels}

    for litter in litters:
        path = os.path.join(working_folder, f"litter_{litter}_processed_cm.json")
        if not os.path.exists(path):
            print(f"Missing: {path}")
            continue
        with open(path, 'r') as f:
            data = json.load(f)

        for subject in data:
            for s in subject['Sessions']:
                if s['Type'] not in ('Training', 'Probe'):
                    continue
                if (s['Type'] == 'Probe' and exclude_rotation
                        and s.get('Rotation', 0) == 1):
                    continue
                band = _band_of(s['Age'])
                if band is None:
                    continue
                for trial in s['Trials']:
                    baited = int(trial.get('Baited', 0))
                    if baited not in (0, 1):
                        continue
                    status = str(trial.get('Completed', 'Completed')).strip().lower()
                    completed = status in ('completed', 'complete', '1', 'true')
                    counts[band][baited][1] += 1
                    if not completed:
                        counts[band][baited][0] += 1

    rows = []
    for lbl in band_labels:
        for baited in (0, 1):
            failed, total = counts[lbl][baited]
            rows.append({
                'Age Band': lbl,
                'Condition': 'Baited' if baited else 'Unbaited',
                'Failed': failed,
                'Completed': total - failed,
                'Total': total,
                'Failure rate (%)': 100 * failed / total if total else np.nan,
            })
    df_out = pd.DataFrame(rows)

    present = [lbl for lbl in band_labels
               if df_out.loc[df_out['Age Band'] == lbl, 'Total'].sum() > 0]
    if not present:
        print("No trials found for the requested litters/bands.")
        return None, df_out

    # hue = baitedness, shade = failed (solid) vs completed (faded)
    col = {
        ('Unbaited', 'failed'):    '#4C72B0',
        ('Unbaited', 'completed'): '#C6D2E8',
        ('Baited',   'failed'):    '#DD8452',
        ('Baited',   'completed'): '#F2D2BD',
    }
    # bottom-to-top: (condition, part)
    seg_order = [('Unbaited', 'failed'), ('Unbaited', 'completed'),
                 ('Baited', 'failed'), ('Baited', 'completed')]

    def _val(lbl, cond, part):
        r = df_out[(df_out['Age Band'] == lbl) & (df_out['Condition'] == cond)].iloc[0]
        return r['Failed'] if part == 'failed' else r['Completed']

    x = np.arange(len(present))
    width = 0.62

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=figsize)
    bottoms = np.zeros(len(present))
    for cond, part in seg_order:
        vals = np.array([_val(lbl, cond, part) for lbl in present], dtype=float)
        ax.bar(x, vals, width, bottom=bottoms,
               color=col[(cond, part)], edgecolor='black', linewidth=0.5,
               label=f"{cond} – {part}")
        # label the failed slices with count + failure rate
        if part == 'failed':
            for xi, lbl, v, b in zip(x, present, vals, bottoms):
                r = df_out[(df_out['Age Band'] == lbl)
                           & (df_out['Condition'] == cond)].iloc[0]
                ax.annotate(f"{int(v)} failed ({r['Failure rate (%)']:.0f}%)",
                            (xi, b + v / 2), ha='center', va='center',
                            fontsize=8, fontweight='bold', color='black',
                            xytext=(0, 0), textcoords='offset points')
        bottoms += vals

    # total run trials on top of each bar
    for xi, lbl, top in zip(x, present, bottoms):
        ax.annotate(f"n = {int(top)}", (xi, top), ha='center', va='bottom',
                    fontsize=9, xytext=(0, 3), textcoords='offset points')

    ax.set_xticks(x)
    ax.set_xticklabels(present)
    ax.set_ylabel('Number of trials run')
    ax.set_xlabel('Developmental stage (session age)')
    ax.set_title(title or
                 "Trials run per age band, failed portion shaded "
                 f"(litters {', '.join(map(str, litters))}, all sessions)")
    ax.legend(title='Condition / outcome', ncol=2, fontsize=8)
    ax.set_ylim(0, bottoms.max() * 1.12 if bottoms.max() else 1)
    plt.tight_layout()
    plt.show()

    return fig, df_out


def plot_failure_rate_by_band(
    working_folder,
    litters=(9, 10, 11),
    age_bands=((17, 20, 'Pre-weaning (P17-20)'),
               (21, 24, 'Peri-weaning (P21-24)'),
               (25, 32, 'Post-weaning (P25-32)')),
    exclude_rotation=False,
    min_trials_for_dot=5,
    show_ci=True,
    figsize=(9, 6),
    title=None
):
    """
    Failure RATE (%) per developmental age band, split by baited vs unbaited.

    Each grouped bar is the pooled failure rate (all trials of that band x
    condition), with a Wilson 95% confidence interval. Overlaid dots are each
    ANIMAL's own failure rate in that cell (only animals with at least
    `min_trials_for_dot` trials there), so group effects can be told apart from
    single-animal outliers. A trial counts as failed when its logged 'Completed'
    flag is not a success value (same rule as `_is_completed`). Each trial is
    banded by its own session's postnatal age.

    Returns
    -------
    fig : matplotlib Figure
    df_pool : pooled failed/total/rate + Wilson CI per band x condition
    df_animal : per-animal failed/total/rate per band x condition
    """
    band_labels = [lbl for _, _, lbl in age_bands]

    def _band_of(age):
        for lo, hi, lbl in age_bands:
            if lo <= age <= hi:
                return lbl
        return None

    def _wilson(k, n, z=1.96):
        if n == 0:
            return (np.nan, np.nan)
        p = k / n
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
        return (100 * (centre - half), 100 * (centre + half))

    # per-animal counts: (subject, band, cond) -> [failed, total]
    from collections import defaultdict
    cell = defaultdict(lambda: [0, 0])
    subjects_seen = set()

    for litter in litters:
        path = os.path.join(working_folder, f"litter_{litter}_processed_cm.json")
        if not os.path.exists(path):
            print(f"Missing: {path}")
            continue
        with open(path, 'r') as f:
            data = json.load(f)

        for subject in data:
            name = subject['Name']
            for s in subject['Sessions']:
                if s['Type'] not in ('Training', 'Probe'):
                    continue
                if (s['Type'] == 'Probe' and exclude_rotation
                        and s.get('Rotation', 0) == 1):
                    continue
                band = _band_of(s['Age'])
                if band is None:
                    continue
                for trial in s['Trials']:
                    baited = int(trial.get('Baited', 0))
                    if baited not in (0, 1):
                        continue
                    cond = 'Baited' if baited else 'Unbaited'
                    status = str(trial.get('Completed', 'Completed')).strip().lower()
                    completed = status in ('completed', 'complete', '1', 'true')
                    cell[(name, band, cond)][1] += 1
                    if not completed:
                        cell[(name, band, cond)][0] += 1
                    subjects_seen.add(name)

    # per-animal frame
    arows = []
    for (name, band, cond), (failed, total) in cell.items():
        arows.append({'Subject': name, 'Age Band': band, 'Condition': cond,
                      'Failed': failed, 'Total': total,
                      'Rate': 100 * failed / total if total else np.nan})
    df_animal = pd.DataFrame(arows)

    # pooled frame
    prows = []
    for lbl in band_labels:
        for cond in ('Unbaited', 'Baited'):
            sub = df_animal[(df_animal['Age Band'] == lbl)
                            & (df_animal['Condition'] == cond)]
            failed = int(sub['Failed'].sum())
            total = int(sub['Total'].sum())
            lo, hi = _wilson(failed, total)
            prows.append({'Age Band': lbl, 'Condition': cond,
                          'Failed': failed, 'Total': total,
                          'Rate': 100 * failed / total if total else np.nan,
                          'CI low': lo, 'CI high': hi,
                          'n_animals': sub['Subject'].nunique()})
    df_pool = pd.DataFrame(prows)

    present = [lbl for lbl in band_labels
               if df_pool.loc[df_pool['Age Band'] == lbl, 'Total'].sum() > 0]
    if not present:
        print("No trials found for the requested litters/bands.")
        return None, df_pool, df_animal

    cond_order = ['Unbaited', 'Baited']
    cond_colors = {'Unbaited': '#4C72B0', 'Baited': '#DD8452'}
    x = np.arange(len(present))
    width = 0.36
    rng = np.random.default_rng(0)

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=figsize)
    for i, cond in enumerate(cond_order):
        offset = (i - 0.5) * width
        rates, lerrs, herrs = [], [], []
        for lbl in present:
            r = df_pool[(df_pool['Age Band'] == lbl)
                        & (df_pool['Condition'] == cond)].iloc[0]
            rates.append(r['Rate'])
            lerrs.append(max(0.0, r['Rate'] - r['CI low']))
            herrs.append(max(0.0, r['CI high'] - r['Rate']))
        ax.bar(x + offset, rates, width, label=cond,
               color=cond_colors[cond], edgecolor='black', linewidth=0.6,
               alpha=0.85, zorder=2)
        if show_ci:
            ax.errorbar(x + offset, rates, yerr=[lerrs, herrs],
                        fmt='none', ecolor='black', elinewidth=1.2, capsize=4,
                        zorder=4)
        # per-animal dots
        for j, lbl in enumerate(present):
            sub = df_animal[(df_animal['Age Band'] == lbl)
                            & (df_animal['Condition'] == cond)
                            & (df_animal['Total'] >= min_trials_for_dot)]
            if sub.empty:
                continue
            xs = x[j] + offset + rng.uniform(-width * 0.28, width * 0.28, len(sub))
            ax.scatter(xs, sub['Rate'], s=32,
                       color=cond_colors[cond], edgecolor='black', linewidth=0.5,
                       alpha=0.9, zorder=5)
        # pooled rate label above the CI
        for j, lbl in enumerate(present):
            r = df_pool[(df_pool['Age Band'] == lbl)
                        & (df_pool['Condition'] == cond)].iloc[0]
            ytop = max(r['CI high'], r['Rate']) if show_ci else r['Rate']
            ax.annotate(f"{r['Rate']:.1f}%\n({r['Failed']}/{r['Total']})",
                        (x[j] + offset, ytop),
                        ha='center', va='bottom', fontsize=7.5,
                        xytext=(0, 3), textcoords='offset points')

    ax.set_xticks(x)
    ax.set_xticklabels(present)
    ax.set_ylabel('Failure rate (%)')
    ax.set_xlabel('Developmental stage (session age)')
    ax.set_ylim(0, None)
    ax.set_title(title or
                 "Failure rate by age band and condition "
                 f"(litters {', '.join(map(str, litters))}, all sessions)")
    # legend: conditions + dot meaning
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [Patch(facecolor=cond_colors[c], edgecolor='black', label=c)
               for c in cond_order]
    handles.append(Line2D([0], [0], marker='o', color='none',
                          markerfacecolor='gray', markeredgecolor='black',
                          markersize=7, label=f"per animal (≥{min_trials_for_dot} trials)"))
    if show_ci:
        handles.append(Line2D([0], [0], color='black', lw=1.2,
                              label='Wilson 95% CI'))
    ax.legend(handles=handles, title='', fontsize=8)
    plt.tight_layout()
    plt.show()

    return fig, df_pool, df_animal


def plot_p1_failure_rate_by_band(
    working_folder,
    litters=(9, 10, 11),
    age_bands=((17, 20, 'Pre-weaning (P17-20)'),
               (21, 24, 'Peri-weaning (P21-24)'),
               (25, 32, 'Post-weaning (P25-32)')),
    days=None,
    exclude_rotation=True,
    min_trials_for_dot=3,
    show_ci=True,
    figsize=(9, 6),
    title=None
):
    """
    Failure rate of the FIRST PROBE TRIAL (P1) per developmental age band, split
    by baited vs unbaited. Same layout as `plot_failure_rate_by_band` but every
    contributing trial is the single P1 (Probe session, trial Number == 1) of a
    probe day, rather than all trials.

    Each probe day contributes one P1, banded by that day's postnatal age and
    coloured by that P1's own baiting. The bar is the pooled P1 failure rate
    (failed P1s / total P1s in that band x condition); dots are each animal's own
    P1 failure rate there.

    Parameters
    ----------
    days : iterable of int or None
        Ordinal probe-day numbers to include (probe sessions sorted by date per
        animal; 1 = that animal's first probe day). None = all days
        -> "P1 across all days". days=[1] -> first day only. Matches the `days`
        convention used elsewhere in this module.
    exclude_rotation : bool
        Drop rotation probe sessions (Rotation == 1). Default True, because a
        rotated probe moves the reward and P1 there is a different condition.
    min_trials_for_dot : int
        Minimum P1s an animal must have in a cell to be drawn as a dot. With a
        single day each animal has one P1 (rate is 0% or 100%); pass
        min_trials_for_dot=1 to still show those dots.
    show_ci : bool
        Draw Wilson 95% CI whiskers (binomial; ignores animal clustering).

    Returns
    -------
    fig, df_pool, df_animal
    """
    from collections import defaultdict
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    band_labels = [lbl for _, _, lbl in age_bands]
    day_set = set(days) if days is not None else None

    def _band_of(age):
        for lo, hi, lbl in age_bands:
            if lo <= age <= hi:
                return lbl
        return None

    def _date_key(d):
        dd, mm, yy = d.split('_')
        return (int(yy), int(mm), int(dd))

    def _wilson(k, n, z=1.96):
        if n == 0:
            return (np.nan, np.nan)
        p = k / n
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
        return (100 * (centre - half), 100 * (centre + half))

    cell = defaultdict(lambda: [0, 0])   # (subject, band, cond) -> [failed, total]

    for litter in litters:
        path = os.path.join(working_folder, f"litter_{litter}_processed_cm.json")
        if not os.path.exists(path):
            print(f"Missing: {path}")
            continue
        with open(path, 'r') as f:
            data = json.load(f)

        for subject in data:
            name = subject['Name']
            probes = [s for s in subject['Sessions'] if s['Type'] == 'Probe']
            if exclude_rotation:
                probes = [s for s in probes if s.get('Rotation', 0) == 0]
            probes = sorted(probes, key=lambda s: _date_key(s['Date']))
            for day_idx, s in enumerate(probes, start=1):
                if day_set is not None and day_idx not in day_set:
                    continue
                band = _band_of(s['Age'])
                if band is None:
                    continue
                p1 = next((t for t in s['Trials'] if t['Number'] == 1), None)
                if p1 is None:
                    continue
                baited = int(p1.get('Baited', 0))
                if baited not in (0, 1):
                    continue
                cond = 'Baited' if baited else 'Unbaited'
                status = str(p1.get('Completed', 'Completed')).strip().lower()
                completed = status in ('completed', 'complete', '1', 'true')
                cell[(name, band, cond)][1] += 1
                if not completed:
                    cell[(name, band, cond)][0] += 1

    df_animal = pd.DataFrame(
        [{'Subject': n, 'Age Band': b, 'Condition': c,
          'Failed': f, 'Total': t, 'Rate': 100 * f / t if t else np.nan}
         for (n, b, c), (f, t) in cell.items()])

    if df_animal.empty:
        print("No P1 trials found for the requested litters/days/bands.")
        return None, pd.DataFrame(), df_animal

    prows = []
    for lbl in band_labels:
        for cond in ('Unbaited', 'Baited'):
            sub = df_animal[(df_animal['Age Band'] == lbl)
                            & (df_animal['Condition'] == cond)]
            failed = int(sub['Failed'].sum()); total = int(sub['Total'].sum())
            lo, hi = _wilson(failed, total)
            prows.append({'Age Band': lbl, 'Condition': cond,
                          'Failed': failed, 'Total': total,
                          'Rate': 100 * failed / total if total else np.nan,
                          'CI low': lo, 'CI high': hi,
                          'n_animals': sub['Subject'].nunique()})
    df_pool = pd.DataFrame(prows)

    present = [lbl for lbl in band_labels
               if df_pool.loc[df_pool['Age Band'] == lbl, 'Total'].sum() > 0]
    if not present:
        print("No P1 trials found for the requested litters/days/bands.")
        return None, df_pool, df_animal

    cond_order = ['Unbaited', 'Baited']
    cond_colors = {'Unbaited': '#4C72B0', 'Baited': '#DD8452'}
    x = np.arange(len(present))
    width = 0.36
    rng = np.random.default_rng(0)

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=figsize)
    for i, cond in enumerate(cond_order):
        offset = (i - 0.5) * width
        rates, lerrs, herrs = [], [], []
        for lbl in present:
            r = df_pool[(df_pool['Age Band'] == lbl)
                        & (df_pool['Condition'] == cond)].iloc[0]
            rates.append(r['Rate'])
            lerrs.append(max(0.0, r['Rate'] - r['CI low']))
            herrs.append(max(0.0, r['CI high'] - r['Rate']))
        ax.bar(x + offset, rates, width, label=cond,
               color=cond_colors[cond], edgecolor='black', linewidth=0.6,
               alpha=0.85, zorder=2)
        if show_ci:
            ax.errorbar(x + offset, rates, yerr=[lerrs, herrs],
                        fmt='none', ecolor='black', elinewidth=1.2, capsize=4,
                        zorder=4)
        for j, lbl in enumerate(present):
            sub = df_animal[(df_animal['Age Band'] == lbl)
                            & (df_animal['Condition'] == cond)
                            & (df_animal['Total'] >= min_trials_for_dot)]
            if sub.empty:
                continue
            xs = x[j] + offset + rng.uniform(-width * 0.28, width * 0.28, len(sub))
            ax.scatter(xs, sub['Rate'], s=32, color=cond_colors[cond],
                       edgecolor='black', linewidth=0.5, alpha=0.9, zorder=5)
        for j, lbl in enumerate(present):
            r = df_pool[(df_pool['Age Band'] == lbl)
                        & (df_pool['Condition'] == cond)].iloc[0]
            ytop = max(r['CI high'], r['Rate']) if show_ci else r['Rate']
            ax.annotate(f"{r['Rate']:.1f}%\n({r['Failed']}/{r['Total']})",
                        (x[j] + offset, ytop), ha='center', va='bottom',
                        fontsize=7.5, xytext=(0, 3), textcoords='offset points')

    ax.set_xticks(x)
    ax.set_xticklabels(present)
    ax.set_ylabel('P1 failure rate (%)')
    ax.set_xlabel('Developmental stage (session age)')
    ax.set_ylim(0, None)
    day_txt = 'all days' if days is None else f"day(s) {list(days)}"
    ax.set_title(title or
                 "First-probe-trial (P1) failure rate by age band "
                 f"(litters {', '.join(map(str, litters))}, {day_txt})")
    handles = [Patch(facecolor=cond_colors[c], edgecolor='black', label=c)
               for c in cond_order]
    handles.append(Line2D([0], [0], marker='o', color='none',
                          markerfacecolor='gray', markeredgecolor='black',
                          markersize=7,
                          label=f"per animal (≥{min_trials_for_dot} P1)"))
    if show_ci:
        handles.append(Line2D([0], [0], color='black', lw=1.2,
                              label='Wilson 95% CI'))
    ax.legend(handles=handles, title='', fontsize=8)
    plt.tight_layout()
    plt.show()

    return fig, df_pool, df_animal


def plot_failure_rate_train_vs_probe_by_band(
    working_folder,
    litters=(9, 10, 11),
    age_bands=((17, 20, 'Pre-weaning (P17-20)'),
               (21, 24, 'Peri-weaning (P21-24)'),
               (25, 32, 'Post-weaning (P25-32)')),
    days=[1],
    first_trial_only=False,
    exclude_rotation=False,
    min_trials_for_dot=3,
    show_ci=True,
    figsize=(9, 6),
    title=None
):
    """
    Failure rate per developmental age band, split by SESSION TYPE
    (Training vs Probe). Same layout as `plot_failure_rate_by_band`, but the two
    grouped bars are the training and probe sessions instead of baited/unbaited,
    and only the requested session-days are counted.

    Parameters
    ----------
    days : iterable of int or None
        Ordinal day numbers to include (each session type sorted by date per
        animal; 1 = first day). Default [1] = first day only. None = all days.
    first_trial_only : bool
        If True, only the first trial (Number == 1) of each session counts, i.e.
        T1 vs P1. If False (default), every trial in the session counts.
    exclude_rotation : bool
        Drop rotation probe sessions (Rotation == 1) from the Probe group.
    min_trials_for_dot : int
        Minimum trials an animal needs in a cell to be drawn as a dot.
    show_ci : bool
        Draw Wilson 95% CI whiskers (binomial; ignores animal clustering).

    Returns
    -------
    fig, df_pool, df_animal
    """
    from collections import defaultdict
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    band_labels = [lbl for _, _, lbl in age_bands]
    day_set = set(days) if days is not None else None
    type_order = ['Training', 'Probe']
    type_colors = {'Training': '#7F7F7F', 'Probe': '#9467BD'}

    def _band_of(age):
        for lo, hi, lbl in age_bands:
            if lo <= age <= hi:
                return lbl
        return None

    def _date_key(d):
        dd, mm, yy = d.split('_')
        return (int(yy), int(mm), int(dd))

    def _wilson(k, n, z=1.96):
        if n == 0:
            return (np.nan, np.nan)
        p = k / n
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
        return (100 * (centre - half), 100 * (centre + half))

    cell = defaultdict(lambda: [0, 0])   # (subject, band, type) -> [failed, total]

    for litter in litters:
        path = os.path.join(working_folder, f"litter_{litter}_processed_cm.json")
        if not os.path.exists(path):
            print(f"Missing: {path}")
            continue
        with open(path, 'r') as f:
            data = json.load(f)

        for subject in data:
            name = subject['Name']
            for stype in type_order:
                sess = [s for s in subject['Sessions'] if s['Type'] == stype]
                if stype == 'Probe' and exclude_rotation:
                    sess = [s for s in sess if s.get('Rotation', 0) == 0]
                sess = sorted(sess, key=lambda s: _date_key(s['Date']))
                for day_idx, s in enumerate(sess, start=1):
                    if day_set is not None and day_idx not in day_set:
                        continue
                    band = _band_of(s['Age'])
                    if band is None:
                        continue
                    for trial in s['Trials']:
                        if first_trial_only and trial['Number'] != 1:
                            continue
                        status = str(trial.get('Completed', 'Completed')).strip().lower()
                        completed = status in ('completed', 'complete', '1', 'true')
                        cell[(name, band, stype)][1] += 1
                        if not completed:
                            cell[(name, band, stype)][0] += 1

    df_animal = pd.DataFrame(
        [{'Subject': n, 'Age Band': b, 'Type': t,
          'Failed': f, 'Total': tot, 'Rate': 100 * f / tot if tot else np.nan}
         for (n, b, t), (f, tot) in cell.items()])

    if df_animal.empty:
        print("No trials found for the requested litters/days/bands.")
        return None, pd.DataFrame(), df_animal

    prows = []
    for lbl in band_labels:
        for stype in type_order:
            sub = df_animal[(df_animal['Age Band'] == lbl)
                            & (df_animal['Type'] == stype)]
            failed = int(sub['Failed'].sum()); total = int(sub['Total'].sum())
            lo, hi = _wilson(failed, total)
            prows.append({'Age Band': lbl, 'Type': stype,
                          'Failed': failed, 'Total': total,
                          'Rate': 100 * failed / total if total else np.nan,
                          'CI low': lo, 'CI high': hi,
                          'n_animals': sub['Subject'].nunique()})
    df_pool = pd.DataFrame(prows)

    present = [lbl for lbl in band_labels
               if df_pool.loc[df_pool['Age Band'] == lbl, 'Total'].sum() > 0]
    if not present:
        print("No trials found for the requested litters/days/bands.")
        return None, df_pool, df_animal

    x = np.arange(len(present))
    width = 0.36
    rng = np.random.default_rng(0)

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=figsize)
    for i, stype in enumerate(type_order):
        offset = (i - 0.5) * width
        rates, lerrs, herrs = [], [], []
        for lbl in present:
            r = df_pool[(df_pool['Age Band'] == lbl)
                        & (df_pool['Type'] == stype)].iloc[0]
            rates.append(r['Rate'])
            lerrs.append(max(0.0, r['Rate'] - r['CI low']))
            herrs.append(max(0.0, r['CI high'] - r['Rate']))
        ax.bar(x + offset, rates, width, label=stype,
               color=type_colors[stype], edgecolor='black', linewidth=0.6,
               alpha=0.85, zorder=2)
        if show_ci:
            ax.errorbar(x + offset, rates, yerr=[lerrs, herrs],
                        fmt='none', ecolor='black', elinewidth=1.2, capsize=4,
                        zorder=4)
        for j, lbl in enumerate(present):
            sub = df_animal[(df_animal['Age Band'] == lbl)
                            & (df_animal['Type'] == stype)
                            & (df_animal['Total'] >= min_trials_for_dot)]
            if sub.empty:
                continue
            xs = x[j] + offset + rng.uniform(-width * 0.28, width * 0.28, len(sub))
            ax.scatter(xs, sub['Rate'], s=32, color=type_colors[stype],
                       edgecolor='black', linewidth=0.5, alpha=0.9, zorder=5)
        for j, lbl in enumerate(present):
            r = df_pool[(df_pool['Age Band'] == lbl)
                        & (df_pool['Type'] == stype)].iloc[0]
            ytop = max(r['CI high'], r['Rate']) if show_ci else r['Rate']
            ax.annotate(f"{r['Rate']:.1f}%\n({r['Failed']}/{r['Total']})",
                        (x[j] + offset, ytop), ha='center', va='bottom',
                        fontsize=7.5, xytext=(0, 3), textcoords='offset points')

    ax.set_xticks(x)
    ax.set_xticklabels(present)
    ax.set_ylabel('Failure rate (%)')
    ax.set_xlabel('Developmental stage (session age)')
    ax.set_ylim(0, None)
    day_txt = 'all days' if days is None else f"day(s) {list(days)}"
    scope = 'first trial (T1 vs P1)' if first_trial_only else 'all trials'
    ax.set_title(title or
                 "Training vs Probe failure rate by age band "
                 f"(litters {', '.join(map(str, litters))}, {day_txt}, {scope})")
    handles = [Patch(facecolor=type_colors[t], edgecolor='black', label=t)
               for t in type_order]
    handles.append(Line2D([0], [0], marker='o', color='none',
                          markerfacecolor='gray', markeredgecolor='black',
                          markersize=7, label=f"per animal (≥{min_trials_for_dot} trials)"))
    if show_ci:
        handles.append(Line2D([0], [0], color='black', lw=1.2,
                              label='Wilson 95% CI'))
    ax.legend(handles=handles, title='', fontsize=8)
    plt.tight_layout()
    plt.show()

    return fig, df_pool, df_animal


# ============================================================
#  SINGLE-TRIAL / TRAJECTORY / SESSION PLOTS
# ============================================================

def draw_static_environment(env_type="Circle", ax=None):
    """
    Draws the wells and boundaries of the environment.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6)) # Uses standard_figsize if defined
    
    # 1. Plot static well locations
    # (Assuming well_locations_cm is defined globally as in your snippet)
    well_x = [x for x, y in well_locations_cm]
    well_y = [y for x, y in well_locations_cm]
    ax.scatter(well_x, well_y, color="gray", label="Wells", s=50, zorder=2)

    # 2. Draw boundary
    if env_type == "Circle":
        circle = plt.Circle((0, 0), circle_radius_cm, color="black", 
                            fill=False, linewidth=1.5)
        ax.add_patch(circle)
    elif env_type == "Square":
        pad = 7
        side = square_side_cm + pad
        square = plt.Rectangle((-side/2, -side/2), side, side,
                               color="black", fill=False, linewidth=1.5)
        ax.add_patch(square)
    
    # 3. Formatting
    ax.set_xlabel("X Position (cm)")
    ax.set_ylabel("Y Position (cm)")
    ax.set_xlim(-45, 45)
    ax.set_ylim(-45, 45)
    ax.set_aspect("equal")
    return ax

import matplotlib.pyplot as plt


# ============================================================
#%%  Learning — development across age
# ============================================================

#%%  -- Path efficiency across trial number, by age band (mean)
def plot_efficiency_across_trials(
    working_folder,
    litters=(9, 10, 11),
    age_bands=((17, 20, 'Pre-weaning (P17-20)'),
              (21, 24, 'Peri-weaning (P21-24)'),
              (25, 32, 'Post-weaning (P25-32)')),
    session_types=('Training', 'Probe'),
    band_by='entry',
    within_agg='median',
    per_animal_first=False,   # False = one median over pooled trials (higher N); True = per-animal then across
    days=None,                # which session-day numbers to include, e.g. [1] = first day only. None = all days.
    first_trial_baited=None,  # None = all sessions; 0 = keep sessions whose trial 1 was UNbaited; 1 = baited
    show_animals=False,
    min_ideal_cm=8.0,
    exclude_rotation=True,
    band_colors=None,
    figsize=(15, 6),
    ylim=None,
    title=None
):
    """
    Path efficiency at each within-session trial number, split by developmental
    age band, with one panel per session type (Training, Probe).

    Efficiency is real/ideal, so LOWER is better (more direct). By default
    (per_animal_first=False) the line is a single MEDIAN over all trials pooled
    across animals and days at each trial position, with the IQR (25-75th pct)
    as the shaded band. Set per_animal_first=True to instead collapse each
    animal's instances of a trial (over its days) with `within_agg`, then take
    the median across animals. Age bands are (min_age, max_age, label) tuples,
    inclusive.

    band_by : 'entry' (default) assigns each animal ONE fixed band from the age
        at its first training session, so each animal is counted once (N =
        5/6/4 for litters 9-11). 'session' bands each session by its own age, so
        an animal is counted in every band it ages through (N = 5/11/12).
    days : list/iterable of int, optional
        Which session-day numbers to include (day 1 = each animal's earliest
        session of that type). e.g. days=[1] restricts to the first day only.
        None (default) uses all days.
    first_trial_baited : int or None, optional
        Split learning curves by whether the session's FIRST trial was baited.
        0 keeps only sessions whose trial 1 was unbaited, 1 only those baited,
        None (default) keeps all. The whole session's trials are still plotted;
        only the session's inclusion depends on its trial-1 baiting.

    Returns
    -------
    fig : Figure
    df_out : long DataFrame (one row per trial).
    """
    from io import StringIO

    if band_colors is None:
        band_colors = ['tab:green', 'tab:blue', 'tab:orange',
                       'tab:red', 'tab:purple']

    def _band_of(age):
        for lo, hi, label in age_bands:
            if lo <= age <= hi:
                return label
        return None

    def _date_key(d):
        dd, mm, yy = d.split('_')
        return (int(yy), int(mm), int(dd))

    rows = []
    for litter in litters:
        path = os.path.join(working_folder, f"litter_{litter}_processed_cm.json")
        if not os.path.exists(path):
            continue
        with open(path, 'r') as f:
            data = json.load(f)

        for subject in data:
            # entry band = band of the age at the animal's earliest training session
            train = [s for s in subject['Sessions'] if s['Type'] == 'Training']
            entry_band = (_band_of(min(train, key=lambda s: _date_key(s['Date']))['Age'])
                          if train else None)

            day_set = set(days) if days is not None else None
            for stype in session_types:
                sess = [s for s in subject['Sessions'] if s['Type'] == stype]
                if stype == 'Probe' and exclude_rotation:
                    sess = [s for s in sess if s.get('Rotation', 0) == 0]
                sess = sorted(sess, key=lambda s: _date_key(s['Date']))  # ordinal day order

                for day_idx, s in enumerate(sess, start=1):
                    if day_set is not None and day_idx not in day_set:
                        continue
                    band = entry_band if band_by == 'entry' else _band_of(s['Age'])
                    if band is None:
                        continue
                    # keep the session only if its first trial matches the requested baiting
                    if first_trial_baited is not None:
                        fb = next((int(t['Baited']) for t in s['Trials']
                                   if t['Number'] == 1 and t.get('Baited') is not None), None)
                        if fb != first_trial_baited:
                            continue
                    well = s['Rewarded well']
                    for trial in s['Trials']:
                        sf, ef = trial['Start frame'], trial['End frame']
                        if pd.isna(sf) or pd.isna(ef) or ef <= sf:
                            continue
                        df = pd.read_json(StringIO(trial['Position']))
                        if df.empty:
                            continue
                        ideal = _ideal_distance(df, well)
                        if not np.isfinite(ideal) or ideal < min_ideal_cm:
                            continue
                        rows.append({
                            'Trial': trial['Number'],
                            'Efficiency': st.calculate_efficiency(df, well),
                            'Band': band, 'Session': stype,
                            'Subject': subject['Name']
                        })

    if not rows:
        print("No usable trials.")
        return None

    df_out = pd.DataFrame(rows)
    n_per_band = df_out.groupby('Band')['Subject'].nunique()
    band_labels = [b[2] for b in age_bands]
    color_map = {lbl: band_colors[i % len(band_colors)]
                 for i, lbl in enumerate(band_labels)}

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, len(session_types), figsize=figsize,
                             sharey=True)
    if len(session_types) == 1:
        axes = [axes]

    for ax, stype in zip(axes, session_types):
        sdata = df_out[df_out['Session'] == stype]
        for band in band_labels:
            bdata = sdata[sdata['Band'] == band]
            if bdata.empty:
                continue
            # per-animal per-trial value (for optional faint lines / per_animal_first)
            per_animal = bdata.groupby(['Trial', 'Subject'])['Efficiency'].agg(within_agg)

            # Faint per-animal lines (each animal's own per-trial value)
            if show_animals:
                pa = per_animal.reset_index()
                for subj, sd in pa.groupby('Subject'):
                    sd = sd.sort_values('Trial')
                    ax.plot(sd['Trial'], sd['Efficiency'], color=color_map[band],
                            linewidth=0.7, alpha=0.3, zorder=1)

            # median + IQR: pooled over trials (default) OR per-animal-first
            grp = per_animal.groupby('Trial') if per_animal_first \
                else bdata.groupby('Trial')['Efficiency']
            plot_metric = grp.median()
            lo, hi = grp.quantile(0.25), grp.quantile(0.75)

            x = plot_metric.index
            n_a = n_per_band.get(band, 0)
            lbl = (f"{band} (N={n_a})" if per_animal_first
                   else f"{band} (N={n_a}; {len(bdata)} trials)")
            ax.plot(x, plot_metric.values, marker='o', markersize=4,
                    color=color_map[band], linewidth=1.8, zorder=3, label=lbl)

            if not show_animals:
                ax.fill_between(x, lo.values, hi.values,
                                color=color_map[band], alpha=0.2)

        ax.set_xlabel('Trial number')
        ax.set_title(f"{stype} sessions", fontsize=11)
        ax.set_xticks(range(1, int(sdata['Trial'].max()) + 1))
        ax.grid(True, alpha=0.3)
        if ylim is not None:
            ax.set_ylim(ylim)

    axes[0].set_ylabel('Path efficiency (real / ideal; lower = more direct)')
    axes[0].legend(title=f'Age band (by {band_by} age)', loc='best', fontsize=8)

    _spread = "faint lines = individual animals" if show_animals else "shading = IQR"
    _method = (f"median across animals ({_spread}); each animal's trials pooled over days ({within_agg})"
               if per_animal_first
               else f"median of all trials pooled across animals & days ({_spread})")
    _tags = []
    if days is not None:
        _tags.append(f"day {list(days)[0]} only" if len(days) == 1 else f"days {sorted(days)}")
    if first_trial_baited is not None:
        _tags.append("first trial baited" if first_trial_baited == 1 else "first trial unbaited")
    if _tags:
        _method = ", ".join(_tags) + " — " + _method
    fig.suptitle(title or
                 f"Path efficiency across trial number, litters "
                 f"{', '.join(map(str, litters))}\n{_method}",
                 fontsize=12)
    plt.tight_layout()
    plt.show()

    return fig, df_out

#%% -- Plot efficiency vs postnatal age (x = P-day)
def plot_efficiency_by_age(
    working_folder,
    litters=(9, 10, 11),
    session_types=('Training', 'Probe'),
    per_animal_first=False,   # False = one median over pooled trials (higher N)
    show_animals=False,       # overlay each animal's own per-age value as a faint line
    min_ideal_cm=8.0,
    exclude_rotation=True,
    min_obs=5,                # skip an age with fewer than this many observations
    figsize=(13, 6),
    ylim=None,
    title=None
):
    """
    Path efficiency against POSTNATAL AGE (x = P-day), one panel per session type.
    Each session's trials are assigned to the animal's actual age that day, so
    there is no entry/session banding ambiguity -- age is the x-axis. By default
    the line is a single median over all trials pooled at each age (shading = IQR);
    set per_animal_first=True to summarise per animal per age first. Efficiency is
    real/ideal, so LOWER is better.

    Returns
    -------
    fig : Figure
    df_out : long DataFrame (one row per trial).
    """
    from io import StringIO

    rows = []
    for litter in litters:
        path = os.path.join(working_folder, f"litter_{litter}_processed_cm.json")
        if not os.path.exists(path):
            continue
        with open(path, 'r') as f:
            data = json.load(f)
        for subject in data:
            for s in subject['Sessions']:
                stype = s['Type']
                if stype not in session_types:
                    continue
                if stype == 'Probe' and exclude_rotation and s.get('Rotation', 0) == 1:
                    continue
                well = s['Rewarded well']
                for trial in s['Trials']:
                    sf, ef = trial['Start frame'], trial['End frame']
                    if pd.isna(sf) or pd.isna(ef) or ef <= sf:
                        continue
                    df = pd.read_json(StringIO(trial['Position']))
                    if df.empty:
                        continue
                    ideal = _ideal_distance(df, well)
                    if not np.isfinite(ideal) or ideal < min_ideal_cm:
                        continue
                    rows.append({'Session': stype, 'Age': s['Age'],
                                 'Efficiency': st.calculate_efficiency(df, well),
                                 'Subject': subject['Name']})

    if not rows:
        print("No usable trials.")
        return None
    df_out = pd.DataFrame(rows)

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, len(session_types), figsize=figsize, sharey=True)
    if len(session_types) == 1:
        axes = [axes]

    for ax, stype in zip(axes, session_types):
        raw = df_out[df_out['Session'] == stype]
        if raw.empty:
            ax.set_title(f"{stype} sessions (no data)")
            continue

        # per-animal per-age value (for faint lines / per_animal_first)
        per_aa = raw.groupby(['Age', 'Subject'])['Efficiency'].median()
        if show_animals:
            pa = per_aa.reset_index()
            for subj, sd in pa.groupby('Subject'):
                sd = sd.sort_values('Age')
                ax.plot(sd['Age'], sd['Efficiency'], color='0.6',
                        linewidth=0.7, alpha=0.35, zorder=1)

        grp = per_aa.groupby('Age') if per_animal_first else raw.groupby('Age')['Efficiency']
        n_obs = raw.groupby('Age').size()
        keep = n_obs[n_obs >= min_obs].index
        centre = grp.median().loc[keep]
        lo = grp.quantile(0.25).loc[keep]
        hi = grp.quantile(0.75).loc[keep]
        x = centre.index
        ax.plot(x, centre.values, marker='o', markersize=5, linewidth=1.8,
                color='tab:purple', zorder=3)
        if not show_animals:
            ax.fill_between(x, lo.values, hi.values, color='tab:purple', alpha=0.18)

        ax.set_xlabel('Postnatal age (days)')
        ax.set_title(f"{stype} sessions", fontsize=11)
        ax.set_xticks(list(x))
        ax.grid(True, alpha=0.3)
        if ylim is not None:
            ax.set_ylim(ylim)

    axes[0].set_ylabel('Path efficiency (real / ideal; lower = more direct)')
    _m = ("median across animals per age" if per_animal_first
          else "median of all trials pooled at each age")
    fig.suptitle(title or f"Path efficiency vs postnatal age, litters "
                 f"{', '.join(map(str, litters))}\n{_m} (shading = IQR)",
                 fontsize=12)
    plt.tight_layout()
    plt.show()
    return fig, df_out


#%% -- Learning curves overlaid by first-trial baiting
def plot_efficiency_baited_overlay(
    working_folder,
    litters=(9, 10, 11),
    age_bands=((17, 20, 'Pre-weaning (P17-20)'),
               (21, 24, 'Peri-weaning (P21-24)'),
               (25, 32, 'Post-weaning (P25-32)')),
    session_types=('Training', 'Probe'),
    band_by='session',
    split_bands=True,         # True: one line per (band, baiting); False: pool ages, 2 lines
    show_iqr=False,           # faint IQR shading (busy with many lines; off by default)
    min_ideal_cm=8.0,
    exclude_rotation=True,
    band_colors=None,
    figsize=(15, 6),
    ylim=None,
    title=None
):
    """
    Within-session learning curves OVERLAID by whether the session's first trial
    was baited: one panel per session type, colour = age band, linestyle =
    first-trial baiting (solid = unbaited, dashed = baited). Line = median of all
    trials pooled at each trial position; efficiency is real/ideal (LOWER is
    better). Set split_bands=False to pool ages into a single unbaited-vs-baited
    pair per panel.

    Returns
    -------
    fig : Figure
    df_out : long DataFrame (one row per trial, with a FirstBaited column).
    """
    from io import StringIO
    import matplotlib.lines as mlines

    if band_colors is None:
        band_colors = ['tab:green', 'tab:blue', 'tab:orange', 'tab:red', 'tab:purple']
    band_labels = [b[2] for b in age_bands]
    color_map = {lbl: band_colors[i % len(band_colors)] for i, lbl in enumerate(band_labels)}

    def _band_of(age):
        for lo, hi, label in age_bands:
            if lo <= age <= hi:
                return label
        return None

    def _date_key(d):
        dd, mm, yy = d.split('_')
        return (int(yy), int(mm), int(dd))

    rows = []
    for litter in litters:
        path = os.path.join(working_folder, f"litter_{litter}_processed_cm.json")
        if not os.path.exists(path):
            continue
        with open(path, 'r') as f:
            data = json.load(f)
        for subject in data:
            train = [s for s in subject['Sessions'] if s['Type'] == 'Training']
            entry_band = (_band_of(min(train, key=lambda s: _date_key(s['Date']))['Age'])
                          if train else None)
            for s in subject['Sessions']:
                stype = s['Type']
                if stype not in session_types:
                    continue
                if stype == 'Probe' and exclude_rotation and s.get('Rotation', 0) == 1:
                    continue
                band = entry_band if band_by == 'entry' else _band_of(s['Age'])
                if band is None:
                    continue
                fb = next((int(t['Baited']) for t in s['Trials']
                           if t['Number'] == 1 and t.get('Baited') is not None), None)
                if fb is None:
                    continue
                well = s['Rewarded well']
                for trial in s['Trials']:
                    sf, ef = trial['Start frame'], trial['End frame']
                    if pd.isna(sf) or pd.isna(ef) or ef <= sf:
                        continue
                    df = pd.read_json(StringIO(trial['Position']))
                    if df.empty:
                        continue
                    ideal = _ideal_distance(df, well)
                    if not np.isfinite(ideal) or ideal < min_ideal_cm:
                        continue
                    rows.append({'Trial': trial['Number'],
                                 'Efficiency': st.calculate_efficiency(df, well),
                                 'Band': band if split_bands else 'All',
                                 'Session': stype, 'FirstBaited': fb})

    if not rows:
        print("No usable trials.")
        return None
    df_out = pd.DataFrame(rows)

    groups = band_labels if split_bands else ['All']
    style = {0: '-', 1: '--'}            # unbaited solid, baited dashed
    fill = {0: None, 1: None}
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, len(session_types), figsize=figsize, sharey=True)
    if len(session_types) == 1:
        axes = [axes]

    for ax, stype in zip(axes, session_types):
        sdata = df_out[df_out['Session'] == stype]
        for g in groups:
            gdata = sdata[sdata['Band'] == g]
            col = color_map.get(g, 'tab:blue') if split_bands else None
            for fb in (0, 1):
                sub = gdata[gdata['FirstBaited'] == fb]
                if sub.empty:
                    continue
                grp = sub.groupby('Trial')['Efficiency']
                med = grp.median()
                # colour: band when split, else distinguish baiting by colour too
                c = col if split_bands else ('tab:blue' if fb == 0 else 'tab:red')
                ax.plot(med.index, med.values, marker='o', markersize=3,
                        linestyle=style[fb], linewidth=1.6, color=c)
                if show_iqr:
                    lo, hi = grp.quantile(0.25), grp.quantile(0.75)
                    ax.fill_between(med.index, lo.values, hi.values, color=c, alpha=0.10)
        ax.set_xlabel('Trial number')
        ax.set_title(f"{stype} sessions", fontsize=11)
        ax.set_xticks(range(1, int(sdata['Trial'].max()) + 1))
        ax.grid(True, alpha=0.3)
        if ylim is not None:
            ax.set_ylim(ylim)

    axes[0].set_ylabel('Path efficiency (real / ideal; lower = more direct)')

    # legend: colour = band (or baiting), linestyle = first-trial baiting
    handles = []
    if split_bands:
        handles += [mlines.Line2D([], [], color=color_map[g], lw=2, label=g)
                    for g in groups]
    else:
        handles += [mlines.Line2D([], [], color='tab:blue', lw=2, label='pooled ages')]
    handles += [mlines.Line2D([], [], color='0.3', lw=2, linestyle='-',
                              label='first trial unbaited'),
                mlines.Line2D([], [], color='0.3', lw=2, linestyle='--',
                              label='first trial baited')]
    axes[0].legend(handles=handles, loc='best', fontsize=8,
                   title=f'Band (by {band_by} age) / first-trial baiting')

    fig.suptitle(title or
                 f"Learning curves by first-trial baiting, litters "
                 f"{', '.join(map(str, litters))}\n"
                 f"median of all trials pooled at each trial position "
                 f"(solid = first trial unbaited, dashed = baited)",
                 fontsize=12)
    plt.tight_layout()
    plt.show()
    return fig, df_out


#%% -- Line trial 1 vs 2-10
def plot_naive_vs_early_first_session(
    working_folder,
    litters=(9, 10, 11),
    age_bands=((17, 20, 'Pre-weaning (P17-20)'),
               (21, 24, 'Peri-weaning (P21-24)'),
               (25, 32, 'Post-weaning (P25-32)')),
    t1_trial=[1],
    t2_10_trials=list(range(2, 11)),
    min_ideal_cm=8.0,
    exclude_rotation=True,
    use_median=True,
    min_median_n=3,
    band_colors=None,
    figsize=(12, 6),
    ylim=None,
    title=None
):
    """
    Daily Warm-up: Trial 1 vs Trials 2-10 path efficiency, averaged across 
    ALL training sessions within each age band. 
    LOWER is better; a drop from T1 to T2-10 reflects within-session relearning/warm-up.
    """
    import os
    import json
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from collections import defaultdict
    from io import StringIO
    import squircle.tools as st
    from squircle.design import well_locations_cm 

    agg = np.median if use_median else np.mean

    if band_colors is None:
        band_colors = ['tab:green', 'tab:blue', 'tab:orange']

    def _ideal_distance(df, rewarded_well):
        x = df['x'].to_numpy(dtype=float)
        y = df['y'].to_numpy(dtype=float)
        tx, ty = well_locations_cm[rewarded_well - 1]
        return np.hypot(tx - x[0], ty - y[0])

    def _band_of(age):
        for lo, hi, label in age_bands:
            if lo <= age <= hi:
                return label
        return None

    def _eff_of_trials(session, numbers):
        wanted = set(numbers)
        out = []
        for trial in session['Trials']:
            if trial['Number'] not in wanted:
                continue
            sf, ef = trial['Start frame'], trial['End frame']
            if pd.isna(sf) or pd.isna(ef) or ef <= sf:
                continue
            df = pd.read_json(StringIO(trial['Position']))
            if df.empty:
                continue
            ideal = _ideal_distance(df, session['Rewarded well'])
            if not np.isfinite(ideal) or ideal < min_ideal_cm:
                continue
            out.append(st.calculate_efficiency(df, session['Rewarded well']))
        return out

    rows = []
    for litter in litters:
        path = os.path.join(working_folder, f"litter_{litter}_processed_cm.json")
        if not os.path.exists(path):
            continue
        with open(path, 'r') as f:
            data = json.load(f)

        for subject in data:
            name = subject['Name']
            for s in subject['Sessions']:
                if s['Type'] != 'Training':
                    continue
                if exclude_rotation and s.get('Rotation', 0) == 1:
                    continue
                band = _band_of(s['Age'])
                if band is None:
                    continue
                
                t1 = _eff_of_trials(s, t1_trial)
                t2_10 = _eff_of_trials(s, t2_10_trials)
                
                if not t1 or not t2_10:
                    continue

                rows.append({'Subject': name, 'Band': band,
                             'Condition': 'Trial 1 (Naive)', 'Efficiency': t1[0]})
                rows.append({'Subject': name, 'Band': band,
                             'Condition': 'Trials 2-10 (Early)', 'Efficiency': agg(t2_10)})

    if not rows:
        print("No usable sessions found.")
        return None

    df_out = pd.DataFrame(rows)
    band_labels = [b[2] for b in age_bands]
    
    # Calculate N as number of unique subjects per band
    n_per_band = df_out.groupby('Band')['Subject'].nunique()

    conditions = ['Trial 1 (Naive)', 'Trials 2-10 (Early)']
    cond_to_x = {c: i for i, c in enumerate(conditions)}
    color_map = {lbl: band_colors[i % len(band_colors)]
                 for i, lbl in enumerate(band_labels)}

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=figsize)

    for band in band_labels:
        bdata = df_out[df_out['Band'] == band]
        if bdata.empty:
            continue
        centre, err, xs = [], [], []
        for cond in conditions:
            vals = bdata.loc[bdata['Condition'] == cond, 'Efficiency']
            if len(vals) < min_median_n:
                continue
            xs.append(cond_to_x[cond])
            centre.append(vals.median() if use_median else vals.mean())
            err.append(vals.sem())
            
        if not xs:
            continue
        n = n_per_band.get(band, 0)
        ax.errorbar(xs, centre, yerr=err, marker='o', markersize=9,
                    linewidth=2.2, capsize=6, elinewidth=2,
                    color=color_map[band], label=f"{band} (N={n} rats)", zorder=3)

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions)
    ax.set_xlim(-0.4, len(conditions) - 0.6)
    ax.set_ylabel('Path efficiency (real / ideal)')
    ax.set_xlabel('Trial block within session')
    ax.grid(True, alpha=0.3)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(title='Age band', loc='best', fontsize=9)
    fig.suptitle(title or "Trial 1 versus Trial 2-10")
    plt.tight_layout()
    plt.show()

    return fig, df_out

#%% -- Coupled lineplot Per day T1 vs P1 + average
def plot_daily_t1_vs_p1_with_avg(
    working_folder,
    litter=11,
    subjects=None,
    subject_colors=None,
    min_age=None,
    min_ideal_cm=8.0,
    exclude_rotation=True,
    split_average_by_env=True,
    min_avg_n=2,
    figsize=(14, 8),
    ylim=None,
    title=None
):
    """
    Daily Training-trial-1 vs Probe-trial-1 efficiency per subject, one litter.
    Each subject gets a short vertical line per day connecting its naive training
    trial (T1) to its first probe trial (P1) of the same day. A group-mean line
    (optionally split by environment) is drawn per day.

    Unlike the Q1/Q4 version this needs BOTH session types, so it walks the JSON
    and pairs training+probe by date rather than using load_trials(Training).
    Efficiency is real/ideal (LOWER is better); a T1->P1 drop mixes within-session
    learning with across-interval retention.

    Returns
    -------
    fig : Figure, or None if there is no usable data.
    """
    from io import StringIO

    path = os.path.join(working_folder, f"litter_{litter}_processed_cm.json")
    if not os.path.exists(path):
        print(f"No file for litter {litter}.")
        return None
    with open(path, 'r') as f:
        data = json.load(f)

    def _one_eff(session, trial_number):
        """Efficiency of a single trial number in a session, or None."""
        for trial in session['Trials']:
            if trial['Number'] != trial_number:
                continue
            sf, ef = trial['Start frame'], trial['End frame']
            if pd.isna(sf) or pd.isna(ef) or ef <= sf:
                return None
            df = pd.read_json(StringIO(trial['Position']))
            if df.empty:
                return None
            ideal = _ideal_distance(df, session['Rewarded well'])
            if not np.isfinite(ideal) or ideal < min_ideal_cm:
                return None
            return st.calculate_efficiency(df, session['Rewarded well'])
        return None

    # subjects present, optional filter
    all_names = [s['Name'] for s in data]
    if subjects is None:
        subjects = sorted(all_names)

    if subject_colors is None:
        palette = plt.get_cmap('tab10')
        subject_colors = {s: palette(i % 10) for i, s in enumerate(subjects)}

    # Build per-subject list of {Age, Environment, T1, P1}
    plot_data = defaultdict(list)
    all_ages = set()
    dropped = []

    for subject in data:
        name = subject['Name']
        if name not in subjects:
            continue
        by_date = defaultdict(dict)
        for s in subject['Sessions']:
            if s['Type'] == 'Probe' and exclude_rotation and s.get('Rotation', 0) == 1:
                continue
            by_date[s['Date']][s['Type']] = s

        for date, pair in by_date.items():
            if 'Training' not in pair or 'Probe' not in pair:
                continue
            tr, pr = pair['Training'], pair['Probe']
            age = tr['Age']
            if min_age is not None and age < min_age:
                continue

            t1 = _one_eff(tr, 1)          # naive training trial
            p1 = _one_eff(pr, 1)          # first probe trial
            if t1 is None or p1 is None:
                dropped.append((name, age))
                continue

            all_ages.add(age)
            plot_data[name].append({
                'Age': age,
                'Environment': tr['Environment'],
                'T1': t1,
                'P1': p1
            })

    if dropped:
        print(f"Dropped {len(dropped)} subject-days lacking a usable T1 or P1.")
    if not all_ages:
        print(f"No usable paired days for litter {litter}.")
        return None

    sorted_ages = sorted(all_ages)
    age_to_x = {age: i for i, age in enumerate(sorted_ages)}   # equal spacing
    q_positions = (-0.2, 0.2)                                  # T1 left, P1 right
    env_markers = {'Square': 's', 'Circle': 'o'}

    fig, ax = plt.subplots(figsize=figsize)
    sns.set_style("whitegrid")
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=1, alpha=0.6)  # ideal

    # Per-subject vertical connectors
    for name in subjects:
        if name not in plot_data:
            continue
        color = subject_colors.get(name, 'gray')
        for day in plot_data[name]:
            xi = age_to_x[day['Age']]
            ax.plot([xi + q_positions[0], xi + q_positions[1]],
                    [day['T1'], day['P1']],
                    marker=env_markers.get(day['Environment'], 'o'),
                    color=color, linestyle='-',
                    linewidth=2, markersize=6, alpha=0.7, label=name)

    # Group-mean connector per day, optionally split by environment
    env_of = {s: plot_data[s][0]['Environment'] for s in plot_data}
    envs = sorted(set(env_of.values())) if split_average_by_env else [None]
    env_styles = {'Circle': ('black', '-'), 'Square': ('dimgray', '--')}

    for env in envs:
        members = [s for s in plot_data if env is None or env_of[s] == env]
        color, ls = env_styles.get(env, ('black', '-'))
        for age in sorted_ages:
            t1s = [d['T1'] for s in members for d in plot_data[s] if d['Age'] == age]
            p1s = [d['P1'] for s in members for d in plot_data[s] if d['Age'] == age]
            if len(t1s) >= min_avg_n and len(p1s) >= min_avg_n:
                xi = age_to_x[age]
                ax.plot([xi + q_positions[0], xi + q_positions[1]],
                        [np.mean(t1s), np.mean(p1s)],
                        marker=env_markers.get(env, 's'),
                        color=color, linestyle=ls,
                        linewidth=3, markersize=8, alpha=0.9,
                        label=f"Group mean ({env})" if env else "Group mean")

    ax.set_xlabel('Age (days)')
    ax.set_ylabel("Path efficiency (real / ideal): T1 (left) -> P1 (right)")
    ax.set_title(title or
                 f"Litter {litter}: daily Training T1 vs Probe P1 per subject")
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    ax.set_xticks(np.arange(len(sorted_ages)))
    ax.set_xticklabels([f"P{age}" for age in sorted_ages], rotation=45)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(),
              title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return fig


#%% Efficiency per Subject Across Ages (Litter-Based Grouping) ---
def plot_efficiency_per_subject_across_ages(
        working_folder,
        litters=(9, 10, 11),
        palette_by_litter=None,
        ylim=(1, 20),
        figsize=(20, 8),
        title="Avg Efficiency per Subject Across Ages"):
    """
    Line plot of mean Real/Ideal path efficiency (>=1, lower is better) per
    subject across age, subjects grouped and coloured by litter.

    Litter identity is taken from each per-litter filename
    (litter_N_processed_cm.json), NOT from a hand-maintained name map, so new
    codenames in litters 10/11 (and any future litter) are picked up with no
    edits here.
    """
    # --- 1. Load each litter separately so filename -> litter is known ---
    subject_litter = {}                       # subject -> "Litter N"
    all_trials = []
    for n in litters:
        path = os.path.join(working_folder, f'litter_{n}_processed_cm.json')
        if not os.path.exists(path):
            print(f"Skipping litter {n}: file not found ({path})")
            continue
        train, probe = st.load_trials(path, kind='Training', include_probes=True)
        label = f'Litter {n}'
        for trial in train + probe:
            subject_litter.setdefault(trial[0], label)
        all_trials.extend(train + probe)

    if not all_trials:
        print("No trials loaded — check working_folder and litter numbers.")
        return None

    # --- 2. Per-subject colours from a per-litter palette ---
    if palette_by_litter is None:
        palette_by_litter = {
            'Litter 9':  'Greys_r',
            'Litter 10': 'Blues_r',
            'Litter 11': 'Greens_r',
        }
    litter_subjects = defaultdict(list)
    for subj, label in subject_litter.items():
        litter_subjects[label].append(subj)

    subject_colors = {}
    for label, subs in litter_subjects.items():
        subs.sort()
        pal = sns.color_palette(palette_by_litter.get(label, 'viridis'),
                                n_colors=max(len(subs), 2))
        for i, subj in enumerate(subs):
            subject_colors[subj] = pal[i]

    # --- 3. Mean efficiency per (subject, age) ---
    data_map = {}
    all_ages_in_data = set()
    for trial in all_trials:
        subject, age, rewarded_well, df = trial[0], trial[6], trial[8], trial[15]
        if df is None or df.empty:
            continue
        efficiency = st.calculate_efficiency(df, rewarded_well)
        all_ages_in_data.add(age)
        data_map.setdefault(subject, {}).setdefault(age, []).append(efficiency)

    sorted_unique_ages = sorted(all_ages_in_data)
    age_to_x_index = {age: i for i, age in enumerate(sorted_unique_ages)}

    # --- 4. Plot: equal spacing (index positions, real ages as tick labels) ---
    plt.figure(figsize=figsize)
    sns.set_style("whitegrid")
    for subject in sorted(data_map.keys()):
        subj_ages = sorted(data_map[subject].keys())
        x_coords = [age_to_x_index[a] for a in subj_ages]
        y_values = [np.mean(data_map[subject][a]) for a in subj_ages]
        plt.plot(x_coords, y_values, marker="o", label=subject,
                 color=subject_colors[subject], linewidth=2.5,
                 markeredgecolor='black', markeredgewidth=0.5)

    plt.xticks(range(len(sorted_unique_ages)), sorted_unique_ages, rotation=0)
    plt.xlim(-0.5, len(sorted_unique_ages) - 0.5)
    plt.title(title)
    plt.xlabel("Age (days)")
    plt.ylabel("Efficiency (Real / Ideal Path)")
    plt.ylim(*ylim)

    # --- 5. Grouped legend: bold litter headers, subjects underneath ---
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        grouped = defaultdict(list)
        for handle, subject in zip(handles, labels):
            grouped[subject_litter.get(subject, 'Unknown')].append((subject, handle))

        final_handles, final_labels = [], []
        sorted_litters = sorted(
            grouped.keys(),
            key=lambda x: int(x.split()[1]) if x != 'Unknown' else 999)
        for i, litter in enumerate(sorted_litters):
            first_subject = grouped[litter][0][0]
            final_handles.append(mlines.Line2D([0], [0],
                                               color=subject_colors[first_subject]))
            final_labels.append(litter)
            for subject, handle in grouped[litter]:
                final_handles.append(handle)
                final_labels.append(subject)
            if i < len(sorted_litters) - 1:
                final_handles.append(mlines.Line2D([0, 4], [0, 0],
                                                   color='black', linewidth=1))
                final_labels.append('')

        leg = plt.legend(final_handles, final_labels, title='Litter / Subject',
                         loc='center left', bbox_to_anchor=(1.0, 0.5),
                         ncol=1, fontsize=8, handlelength=3.0)
        for text in leg.get_texts():
            if text.get_text().startswith('Litter'):
                text.set_weight('bold')
                text.set_fontsize(9)

    plt.tight_layout()
    plt.show()


# ============================================================
#%% Retention
# ============================================================

#%%  -- Training 3-4-5 probe 2 + failure percentage
def plot_retention_blocks(
    working_folder,
    litters=(9, 10, 11),
    training_trials=(3, 4, 5),
    probe_sets=((2,),),
    probe_labels=('Probe 2',),
    min_valid_training=2,
    use_median=True,
    min_ideal_cm=8.0,
    exclude_rotation=True,
    normalise=False,
    min_training_eff=1.2,
    show_group_median=True,
    min_median_n=3,
    show_failures=False,
    failure_first_n=10,
    jitter=0.12,
    figsize=(15, 8),
    ylim=None,
    title=None
):
    """
    Retention using explicitly specified trial numbers, with an optional
    failure-rate overlay on a secondary axis.
    The training baseline is the median (or mean) of training_trials. Each
    entry in probe_sets is compared against that baseline in its own panel.
    Efficiency is real/ideal, so NEGATIVE values mean retention.
    With show_failures=True, each panel also shows the percentage of failed
    trials among the first failure_first_n training and probe trials per age,
    read from the logged 'Completed' flag, on a right-hand 0-100 axis.
    Returns
    -------
    fig : Figure
    df_out : DataFrame, one row per animal-day per comparison.
    """
    from io import StringIO
    agg = np.median if use_median else np.mean
    stat = 'median' if use_median else 'mean'
    def _effs(session, numbers):
        wanted = set(numbers)
        out = []
        for trial in session['Trials']:
            if trial['Number'] not in wanted:
                continue
            sf, ef = trial['Start frame'], trial['End frame']
            if pd.isna(sf) or pd.isna(ef) or ef <= sf:
                continue
            df = pd.read_json(StringIO(trial['Position']))
            if df.empty:
                continue
            ideal = _ideal_distance(df, session['Rewarded well'])
            if not np.isfinite(ideal) or ideal < min_ideal_cm:
                continue
            out.append(st.calculate_efficiency(df, session['Rewarded well']))
        return out
    rows, dropped = [], []
    subjects_of = defaultdict(list)
    for litter in litters:
        path = os.path.join(working_folder, f"litter_{litter}_processed_cm.json")
        if not os.path.exists(path):
            print(f"Missing file for litter {litter}, skipping.")
            continue
        with open(path, 'r') as f:
            data = json.load(f)
        for subject in data:
            name = subject['Name']
            if name not in subjects_of[litter]:
                subjects_of[litter].append(name)
            by_date = defaultdict(dict)
            for s in subject['Sessions']:
                if s['Type'] == 'Probe' and exclude_rotation and s['Rotation'] == 1:
                    continue
                by_date[s['Date']][s['Type']] = s
            for date, pair in by_date.items():
                if 'Training' not in pair or 'Probe' not in pair:
                    continue
                tr, pr = pair['Training'], pair['Probe']
                base = _effs(tr, training_trials)
                if len(base) < min_valid_training:
                    dropped.append((name, tr['Age'], 'baseline', len(base)))
                    continue
                base_val = agg(base)
                if normalise and base_val < min_training_eff:
                    dropped.append((name, tr['Age'], 'denominator', 0))
                    continue
                for numbers, label in zip(probe_sets, probe_labels):
                    p_eff = _effs(pr, numbers)
                    if not p_eff:
                        dropped.append((name, tr['Age'], label, 0))
                        continue
                    p_val = agg(p_eff)
                    rows.append({
                        'Subject': name, 'Litter': litter, 'Age': tr['Age'],
                        'Environment': tr['Environment'], 'Comparison': label,
                        'Training': base_val, 'Probe': p_val,
                        'Difference': p_val - base_val,
                        'Relative': (p_val - base_val) / base_val
                    })
    if dropped:
        print(f"Dropped {len(dropped)} animal-day/comparison entries for "
              f"insufficient valid trials.")
    if not rows:
        print("No usable animal-days.")
        return None
    df_out = pd.DataFrame(rows)
    ycol = 'Relative' if normalise else 'Difference'
    # Failure rates computed ONCE, if requested
    fr = failure_rate_by_age(working_folder, litters=litters,
                             first_n=failure_first_n,
                             exclude_rotation=exclude_rotation) if show_failures else None
    litter_cmaps = {9: 'Blues', 10: 'Greens', 11: 'Reds'}
    subject_colors = {}
    for litter in litters:
        names = sorted(subjects_of[litter])
        if not names:
            continue
        cmap = plt.get_cmap(litter_cmaps.get(litter, 'Greys'))
        for name, shade in zip(names, cmap(np.linspace(0.45, 0.9, len(names)))):
            subject_colors[name] = shade
    markers = {'Square': 's', 'Circle': 'o'}
    ages = sorted(df_out['Age'].unique())
    age_to_x = {a: i for i, a in enumerate(ages)}
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(len(probe_labels), 1, figsize=figsize,
                             sharex=True, sharey=True)
    if len(probe_labels) == 1:
        axes = [axes]
    rng = np.random.default_rng(0)
    tr_label = '-'.join(map(str, training_trials))
    for ax, label in zip(axes, probe_labels):
        sub = df_out[df_out['Comparison'] == label]
        ax.axhline(0, color='black', linestyle='--', linewidth=1.5, zorder=1)
        for _, r in sub.iterrows():
            xi = age_to_x[r['Age']] + rng.uniform(-jitter, jitter)
            ax.scatter(xi, r[ycol],
                       color=subject_colors.get(r['Subject'], 'gray'),
                       marker=markers.get(r['Environment'], 'o'),
                       s=65, edgecolor='black', linewidth=0.4, alpha=0.85,
                       zorder=3, label=r['Subject'])
        if show_group_median:
            med = sub.groupby('Age')[ycol].median()
            cnt = sub.groupby('Age')[ycol].size()
            keep = [a for a in med.index if cnt[a] >= min_median_n]
            if keep:
                ax.plot([age_to_x[a] for a in keep], [med[a] for a in keep],
                        color='black', linewidth=2.5, marker='D', markersize=7,
                        alpha=0.9, zorder=4, label='Median')
        ax.set_ylabel(
            f"({label} - T{tr_label}) / T{tr_label}" if normalise
            else f"{label} - training {tr_label}")
        ax.set_title("Negative = retention", fontsize=10)
        ax.grid(True, alpha=0.3)
        if ylim is not None:
            ax.set_ylim(ylim)
        # --- failure-rate overlay ---
        if show_failures:
            axf = ax.twinx()
            for sess, colour, ls in [('Training', 'tab:gray', '-'),
                                     ('Probe', 'tab:purple', '--')]:
                fa = sorted(a for a in fr[sess] if a in age_to_x)
                axf.plot([age_to_x[a] for a in fa], [fr[sess][a] for a in fa],
                         color=colour, linestyle=ls, linewidth=1.6, marker='^',
                         markersize=5, alpha=0.55, zorder=2,
                         label=f"% failed ({sess})")
            axf.set_ylabel(f"Failed trials (%), first {failure_first_n}",
                           rotation=270, labelpad=18, va='bottom')
            axf.yaxis.set_label_coords(1.04, 0.4)
            axf.set_ylim(0, 50)
            axf.grid(False)
            axf.legend(loc='upper right', fontsize=7)
    axes[-1].set_xlabel('Age (days)')
    axes[-1].set_xticks(np.arange(len(ages)))
    axes[-1].set_xticklabels([f"P{a}" for a in ages], rotation=45)
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[0].legend(by_label.values(), by_label.keys(), title='Animal',
                   bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    fig.suptitle(title or
                 f"Retention: Training trials {tr_label} vs {label} ")
    plt.tight_layout()
    plt.show()
    return fig, df_out


#%% -- Boxplot Probe1 - Training 1  by band (all days)
def plot_allsessions_t1_vs_probe1_by_band(
    working_folder,
    litters=(9, 10, 11),
    exclude_rotation=True,
    min_ideal_cm=8.0,
    normalise=False,
    figsize=(9, 6),
    ylim=None,
    title=None
):
    """
    Training trial 1 vs Probe trial 1 across ALL completed sessions (not just
    the first day). Each individual data point is one SESSION's Probe1 - T1
    score; an animal with N completed sessions contributes N points. Animals
    are grouped by ENTRY cohort (first-day age band); all of an animal's
    sessions sit in that band. The box summarises session-level scores, so
    animals with more sessions weight their band's box more heavily.

    Efficiency is real/ideal; negative = probe better than naive.

    Returns
    -------
    fig : Figure
    df_out : one row per animal per completed session
    """
    import os, json
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from collections import defaultdict
    from io import StringIO
    import squircle.tools as st
    from squircle.design import well_locations_cm

    def _ideal_distance(df, rewarded_well):
        x = df['x'].to_numpy(dtype=float)
        y = df['y'].to_numpy(dtype=float)
        tx, ty = well_locations_cm[rewarded_well - 1]
        return np.hypot(tx - x[0], ty - y[0])

    def _one_eff(session, trial_number):
        for trial in session['Trials']:
            if trial['Number'] != trial_number:
                continue
            sf, ef = trial['Start frame'], trial['End frame']
            if pd.isna(sf) or pd.isna(ef) or ef <= sf:
                return None
            df = pd.read_json(StringIO(trial['Position']))
            if df.empty:
                return None
            ideal = _ideal_distance(df, session['Rewarded well'])
            if not np.isfinite(ideal) or ideal < min_ideal_cm:
                return None
            return st.calculate_efficiency(df, session['Rewarded well'])
        return None

    def _get_band(age):
        if 17 <= age <= 20: return "Pre-weaning (P17-20)"
        elif 21 <= age <= 24: return "Peri-weaning (P21-24)"
        elif 25 <= age <= 32: return "Post-weaning (P25-32)"
        return None

    def _date_key(d):
        dd, mm, yy = d.split('_')
        return (int(yy), int(mm), int(dd))

    rows = []
    for litter in litters:
        path = os.path.join(working_folder, f"litter_{litter}_processed_cm.json")
        if not os.path.exists(path):
            continue
        with open(path, 'r') as f:
            data = json.load(f)

        for subject in data:
            by_date = defaultdict(dict)
            for s in subject['Sessions']:
                if s['Type'] == 'Probe' and exclude_rotation and s.get('Rotation', 0) == 1:
                    continue
                by_date[s['Date']][s['Type']] = s

            valid_dates = [d for d, pair in by_date.items()
                           if 'Training' in pair and 'Probe' in pair]
            if not valid_dates:
                continue

            # ENTRY cohort: fixed from the animal's earliest day
            first_date = min(valid_dates, key=_date_key)
            entry_band = _get_band(by_date[first_date]['Training']['Age'])
            if entry_band is None:
                continue

            # one row per COMPLETED session (both trial-1s usable)
            for date in sorted(valid_dates, key=_date_key):
                pair = by_date[date]
                tr, pr = pair['Training'], pair['Probe']
                base = _one_eff(tr, 1)
                probe = _one_eff(pr, 1)
                if base is None or probe is None:
                    continue
                rows.append({
                    'Subject': subject['Name'], 'Litter': litter,
                    'Age': tr['Age'], 'Date': date, 'Age Band': entry_band,
                    'Environment': tr['Environment'],
                    'Training T1': base, 'Probe P1': probe,
                    'Difference': probe - base,
                    'Relative': (probe - base) / base
                })

    df_out = pd.DataFrame(rows)
    if df_out.empty:
        print("No usable sessions.")
        return None

    ycol = 'Relative' if normalise else 'Difference'
    band_order = ["Pre-weaning (P17-20)", "Peri-weaning (P21-24)",
                  "Post-weaning (P25-32)"]
    present = [b for b in band_order if b in df_out['Age Band'].values]

    # N = animals per band, plus session count for transparency
    n_animals = df_out.groupby('Age Band')['Subject'].nunique()
    n_sessions = df_out.groupby('Age Band').size()

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=figsize)
    ax.axhline(0, color='black', linestyle='--', linewidth=1.5, zorder=1)

    # --- per-animal colours, shaded by litter of origin ---
    litter_cmaps = {9: 'Reds', 10: 'Blues', 11: 'Greens'}
    subject_colors = {}
    for litter in sorted(df_out['Litter'].unique()):
        names = sorted(df_out.loc[df_out['Litter'] == litter, 'Subject'].unique())
        cmap = plt.get_cmap(litter_cmaps.get(litter, 'Greys'))
        # spread shades across the mid-to-dark range so they stay distinguishable
        for name, shade in zip(names, cmap(np.linspace(0.45, 0.9, len(names)))):
            subject_colors[name] = shade

    sns.boxplot(data=df_out, x='Age Band', y=ycol, order=present,
                color='lightgray', showfliers=False, ax=ax, zorder=2)
    sns.stripplot(data=df_out, x='Age Band', y=ycol, order=present,
                  hue='Subject', palette=subject_colors, size=6, alpha=0.85,
                  jitter=True, edgecolor='black', linewidth=0.4, ax=ax, zorder=3)

    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([f"{b}\n({n_animals.get(b, 0)} animals)" for b in present])
    ax.set_ylabel("(P1 - T1) / T1" if normalise else "Probe 1 - Training 1")
    ax.set_xlabel('Developmental Stage (entry cohort)')
    ax.set_title(title or
                 "All completed sessions: Training T1 vs Probe 1 per session "
                 "(negative = probe better)")
    ax.legend(title='Animal', bbox_to_anchor=(1.02, 1), loc='upper left',
              fontsize=7)
    if ylim:
        ax.set_ylim(ylim)
    plt.tight_layout()
    plt.show()

    return fig, df_out

#%% Plot first day T1 P1
def plot_first_day_t1_vs_probe1_by_band(
    working_folder,
    litters=(9, 10, 11),
    exclude_rotation=True,
    min_ideal_cm=8.0,
    normalise=False,
    figsize=(9, 6),
    ylim=None,
    title=None
):
    """
    Training trial 1 vs Probe trial 1, using ONLY each animal's first day
    (earliest chronological date with both a training and a probe session).
    One paired value per animal; band = that animal's first-day age band.
    Because each animal appears once, the three bands are entry-age cohorts.
    Efficiency is real/ideal; negative difference = probe better than naive.
    """
    import os, json
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from collections import defaultdict
    from io import StringIO
    import squircle.tools as st
    from squircle.design import well_locations_cm

    def _ideal_distance(df, rewarded_well):
        x = df['x'].to_numpy(dtype=float)
        y = df['y'].to_numpy(dtype=float)
        tx, ty = well_locations_cm[rewarded_well - 1]
        return np.hypot(tx - x[0], ty - y[0])

    def _one_eff(session, trial_number):
        for trial in session['Trials']:
            if trial['Number'] != trial_number:
                continue
            sf, ef = trial['Start frame'], trial['End frame']
            if pd.isna(sf) or pd.isna(ef) or ef <= sf:
                return None
            df = pd.read_json(StringIO(trial['Position']))
            if df.empty:
                return None
            ideal = _ideal_distance(df, session['Rewarded well'])
            if not np.isfinite(ideal) or ideal < min_ideal_cm:
                return None
            return st.calculate_efficiency(df, session['Rewarded well'])
        return None

    def _get_band(age):
        if 17 <= age <= 20: return "Pre-weaning (P17-20)"
        elif 21 <= age <= 24: return "Peri-weaning (P21-24)"
        elif 25 <= age <= 32: return "Post-weaning (P25-32)"
        return None

    def _date_key(d):
        dd, mm, yy = d.split('_')
        return (int(yy), int(mm), int(dd))

    rows = []
    for litter in litters:
        path = os.path.join(working_folder, f"litter_{litter}_processed_cm.json")
        if not os.path.exists(path):
            continue
        with open(path, 'r') as f:
            data = json.load(f)

        for subject in data:
            by_date = defaultdict(dict)
            for s in subject['Sessions']:
                if s['Type'] == 'Probe' and exclude_rotation and s.get('Rotation', 0) == 1:
                    continue
                by_date[s['Date']][s['Type']] = s

            valid_dates = [d for d, pair in by_date.items()
                           if 'Training' in pair and 'Probe' in pair]
            if not valid_dates:
                continue

            # TRUE chronological first day (parse date, don't string-sort)
            first_date = min(valid_dates, key=_date_key)
            pair = by_date[first_date]
            tr, pr = pair['Training'], pair['Probe']

            band = _get_band(tr['Age'])
            if band is None:
                continue

            base = _one_eff(tr, 1)          # Training trial 1
            probe = _one_eff(pr, 1)         # Probe trial 1
            if base is None or probe is None:
                continue

            rows.append({
                'Subject': subject['Name'], 'Litter': litter,
                'Age': tr['Age'], 'Date': first_date, 'Age Band': band,
                'Training T1': base, 'Probe P1': probe,
                'Difference': probe - base,
                'Relative': (probe - base) / base
            })

    df_out = pd.DataFrame(rows)
    if df_out.empty:
        print("No usable first days.")
        return None

    ycol = 'Relative' if normalise else 'Difference'
    band_order = ["Pre-weaning (P17-20)", "Peri-weaning (P21-24)",
                  "Post-weaning (P25-32)"]
    present = [b for b in band_order if b in df_out['Age Band'].values]
    n_per_band = df_out.groupby('Age Band')['Subject'].nunique()

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=figsize)
    ax.axhline(0, color='black', linestyle='--', linewidth=1.5, zorder=1)

    # --- per-animal colours, shaded by litter of origin (matches
    #     plot_allsessions_t1_vs_probe1_by_band) ---
    litter_cmaps = {9: 'Reds', 10: 'Blues', 11: 'Greens'}
    subject_colors = {}
    for litter in sorted(df_out['Litter'].unique()):
        names = sorted(df_out.loc[df_out['Litter'] == litter, 'Subject'].unique())
        cmap = plt.get_cmap(litter_cmaps.get(litter, 'Greys'))
        # spread shades across the mid-to-dark range so they stay distinguishable
        for name, shade in zip(names, cmap(np.linspace(0.45, 0.9, len(names)))):
            subject_colors[name] = shade

    sns.boxplot(data=df_out, x='Age Band', y=ycol, order=present,
                color='lightgray', showfliers=False, ax=ax, zorder=2)
    sns.stripplot(data=df_out, x='Age Band', y=ycol, order=present,
                  hue='Subject', palette=subject_colors, size=7, alpha=0.85,
                  jitter=True, edgecolor='black', linewidth=0.5, ax=ax, zorder=3)

    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([f"{b}\n(N={n_per_band.get(b, 0)})" for b in present])
    ax.set_ylabel("(P1 - T1) / T1" if normalise else "Probe 1 - Training 1")
    ax.set_title(title or
                 "First-day: Training T1 vs Probe 1 (negative = probe better)")
    ax.legend(title='Animal', bbox_to_anchor=(1.02, 1), loc='upper left',
              fontsize=7)
    if ylim:
        ax.set_ylim(ylim)
    plt.tight_layout()
    plt.show()

    return fig, df_out

#%% -- Boxplot raw efficiencies T1 P1 per band
def plot_raw_t1_p1_by_band(
    working_folder,
    litters=(9, 10, 11),
    exclude_rotation=True,
    min_ideal_cm=8.0,
    max_days=7,
    days=None,                 # which paired-day numbers to include, e.g. [1] = first day only,
                               #   range(3,6) = days 3-5. None = all days up to max_days.
    use_median=False,          # False = mean across each animal's days
    require_p1_unbaited=False,  # keep a day only if its first probe trial is unbaited
    color_by_animal=False,      # grey boxes + per-animal dots (litter-shaded), like the per-session figure
    age_bands=((17, 20, 'Started pre-weaning (P17-20)'),
               (21, 24, 'Started peri-weaning (P21-24)'),
               (25, 32, 'Started post-weaning (P25-32)')),
    figsize=(10, 6),
    ylim=None,
    title=None
):
    """
    Raw efficiencies: naive Training trial 1 (T1) and first Probe trial (P1)
    shown as SEPARATE boxes per entry age band, rather than their difference.
    Makes the baseline visible: you can see that young bands have a high (bad)
    T1 and where P1 sits relative to it.

    Each animal contributes one mean (or median) T1 and one P1 per band, averaged
    across its paired days; band is fixed by entry (day-1) age. Efficiency is
    real/ideal, LOWER is better. Companion to the P1-T1 difference figure (same
    banding, same per-animal collapse).

    days : list/iterable of int, optional
        Which paired-day numbers to include (day 1 = each animal's first paired
        training+probe day). e.g. days=[1] restricts to the first day only,
        days=range(3, 6) to days 3-5. When None (default), all days up to
        `max_days` are used; when set, `max_days` is ignored.

    require_p1_unbaited : bool, optional
        When True, a paired day is kept only if the first probe trial (P1) was
        UNBAITED. T1's baiting is irrelevant, so this retains both
        unbaited-T1/unbaited-P1 and baited-T1/unbaited-P1 days, and drops days
        where the probe was baited. This isolates genuine retention (the animal
        going to the well without odour on the probe) from odour-guided
        performance.

    Returns
    -------
    fig : Figure
    df_long : long DataFrame, one row per animal per band per block (T1/P1)
    """
    import os, json
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from collections import defaultdict
    from io import StringIO
    import squircle.tools as st
    from squircle.design import well_locations_cm

    def _ideal_distance(df, rewarded_well):
        x = df['x'].to_numpy(dtype=float)
        y = df['y'].to_numpy(dtype=float)
        tx, ty = well_locations_cm[rewarded_well - 1]
        return np.hypot(tx - x[0], ty - y[0])

    def _one_eff(session, trial_number):
        for trial in session['Trials']:
            if trial['Number'] != trial_number:
                continue
            sf, ef = trial['Start frame'], trial['End frame']
            if pd.isna(sf) or pd.isna(ef) or ef <= sf:
                return None
            df = pd.read_json(StringIO(trial['Position']))
            if df.empty:
                return None
            ideal = _ideal_distance(df, session['Rewarded well'])
            if not np.isfinite(ideal) or ideal < min_ideal_cm:
                return None
            return st.calculate_efficiency(df, session['Rewarded well'])
        return None

    def _is_baited(session, trial_number):
        """Return the Baited flag (0/1) of a given trial, or None if missing."""
        for trial in session['Trials']:
            if trial['Number'] == trial_number:
                b = trial.get('Baited')
                return None if b is None else int(b)
        return None

    def _band_of(age):
        for lo, hi, label in age_bands:
            if lo <= age <= hi:
                return label
        return None

    def _date_key(d):
        dd, mm, yy = d.split('_')
        return (int(yy), int(mm), int(dd))

    # --- collect one T1 and one P1 per animal-day ---
    rows = []
    n_dropped_baited_p1 = 0
    for litter in litters:
        path = os.path.join(working_folder, f"litter_{litter}_processed_cm.json")
        if not os.path.exists(path):
            continue
        with open(path, 'r') as f:
            data = json.load(f)

        for subject in data:
            name = subject['Name']
            by_date = defaultdict(dict)
            for s in subject['Sessions']:
                if s['Type'] == 'Probe' and exclude_rotation and s.get('Rotation', 0) == 1:
                    continue
                by_date[s['Date']][s['Type']] = s

            paired = [(d, p) for d, p in by_date.items()
                      if 'Training' in p and 'Probe' in p]
            if not paired:
                continue
            paired.sort(key=lambda item: _date_key(item[0]))

            entry_band = _band_of(paired[0][1]['Training']['Age'])
            if entry_band is None:
                continue

            day_set = set(days) if days is not None else None
            for day_num, (date, pair) in enumerate(paired, start=1):
                if day_set is not None:
                    if day_num not in day_set:
                        continue           # explicit day selection
                elif day_num > max_days:
                    break                  # default: first max_days days
                if require_p1_unbaited and _is_baited(pair['Probe'], 1) == 1:
                    n_dropped_baited_p1 += 1     # probe was baited -> not retention
                    continue
                t1 = _one_eff(pair['Training'], 1)
                p1 = _one_eff(pair['Probe'], 1)
                if t1 is None or p1 is None:      # need BOTH for a matched day
                    continue
                rows.append({'Subject': name, 'Litter': litter,
                             'Band': entry_band, 'T1': t1, 'P1': p1})

    if require_p1_unbaited:
        print(f"Dropped {n_dropped_baited_p1} paired day(s) where the first "
              f"probe trial (P1) was baited.")

    df_days = pd.DataFrame(rows)
    if df_days.empty:
        print("No usable paired days.")
        return None

    # --- collapse to one T1 and one P1 per animal per band ---
    agg = 'median' if use_median else 'mean'
    per_animal = (df_days.groupby(['Band', 'Subject'])[['T1', 'P1']]
                         .agg(agg).reset_index())

    # melt to long form so T1/P1 become a hue
    df_long = per_animal.melt(id_vars=['Band', 'Subject'],
                              value_vars=['T1', 'P1'],
                              var_name='Block', value_name='Efficiency')

    band_labels = [b[2] for b in age_bands]
    present = [b for b in band_labels if b in df_long['Band'].values]
    n_per_band = per_animal.groupby('Band')['Subject'].nunique()

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=figsize)

    if color_by_animal:
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        # per-animal colours, shaded by litter of origin (matches per-session fig)
        litter_cmaps = {9: 'Reds', 10: 'Blues', 11: 'Greens'}
        subj_litter = (df_days.drop_duplicates('Subject')
                              .set_index('Subject')['Litter'].to_dict())
        subject_colors = {}
        for litter in sorted(df_days['Litter'].unique()):
            names = sorted(df_days.loc[df_days['Litter'] == litter, 'Subject'].unique())
            cmap = plt.get_cmap(litter_cmaps.get(litter, 'Greys'))
            for name, shade in zip(names, cmap(np.linspace(0.45, 0.9, len(names)))):
                subject_colors[name] = shade

        # two neutral greys so T1 and P1 boxes stay distinguishable
        box_fill = {'T1': '#bdbdbd', 'P1': '#e0e0e0'}
        offset = {'T1': -0.2, 'P1': 0.2}
        rng = np.random.default_rng(0)

        for i, band in enumerate(present):
            sub = per_animal[per_animal['Band'] == band]
            for block in ('T1', 'P1'):
                xc = i + offset[block]
                vals = sub[block].to_numpy(dtype=float)
                bp = ax.boxplot([vals], positions=[xc], widths=0.34,
                                patch_artist=True, showfliers=False, zorder=2,
                                medianprops=dict(color='black', linewidth=1.2))
                for box in bp['boxes']:
                    box.set(facecolor=box_fill[block], edgecolor='black',
                            linewidth=0.8, alpha=0.9)
                # per-animal jittered dots, coloured like the per-session figure
                jit = rng.uniform(-0.07, 0.07, size=len(sub))
                ax.scatter(xc + jit, vals,
                           c=[subject_colors[s] for s in sub['Subject']],
                           s=42, alpha=0.9, edgecolor='black', linewidth=0.4,
                           zorder=3)

        ax.set_xticks(range(len(present)))
        ax.set_xticklabels([f"{b}\n(N={n_per_band.get(b, 0)})" for b in present])
        ax.set_xlim(-0.6, len(present) - 0.4)

        # two legends: which box is which, and who each colour is
        block_leg = ax.legend(
            handles=[Patch(facecolor=box_fill['T1'], edgecolor='black',
                           label='T1 (naive training)'),
                     Patch(facecolor=box_fill['P1'], edgecolor='black',
                           label='P1 (first probe)')],
            title='Trial', loc='upper left', fontsize=8)
        ax.add_artist(block_leg)
        subj_handles = [
            Line2D([0], [0], marker='o', linestyle='',
                   markerfacecolor=subject_colors[s], markeredgecolor='black',
                   markersize=6, label=s)
            for s in sorted(subject_colors, key=lambda s: (subj_litter[s], s))
            if s in per_animal['Subject'].values]
        ax.legend(handles=subj_handles, title='Animal',
                  bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7)
    else:
        sns.boxplot(data=df_long, x='Band', y='Efficiency', hue='Block',
                    order=present, hue_order=['T1', 'P1'],
                    palette={'T1': '#66c2a5', 'P1': '#fc8d62'},
                    showfliers=False, ax=ax, zorder=2)
        sns.stripplot(data=df_long, x='Band', y='Efficiency', hue='Block',
                      order=present, hue_order=['T1', 'P1'],
                      palette={'T1': '#2c7a5b', 'P1': '#c65a2e'},
                      dodge=True, size=6, alpha=0.85, edgecolor='black',
                      linewidth=0.4, ax=ax, zorder=3)

        # de-duplicate legend (box + strip both add entries)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:2], ['T1 (naive training)', 'P1 (first probe)'],
                  title='Trial', loc='best')

        ax.set_xticks(range(len(present)))
        ax.set_xticklabels([f"{b}\n(N={n_per_band.get(b, 0)})" for b in present])

    ax.set_ylabel(f"Path efficiency (real / ideal), per-animal {agg}")
    ax.set_xlabel('Developmental stage (entry cohort)')
    if days is None:
        _dayspan = f"days 1-{max_days}"
    elif len(days) == 1:
        _dayspan = f"day {list(days)[0]} only"
    else:
        _dayspan = f"days {sorted(days)}"
    default_title = ("Naive T1 vs first probe P1, by entry cohort "
                     f"(lower = better) — {_dayspan}")
    if require_p1_unbaited:
        default_title += "\nunbaited probes only (P1 unbaited)"
    ax.set_title(title or default_title)
    if ylim:
        ax.set_ylim(ylim)
    plt.tight_layout()
    plt.show()

    return fig, df_days

#%% Baited vs Unbaited
# ============================================================

#%% -- B vs U Boxplot (one litter)
def plot_baited_vs_unbaited_boxplot(working_folder, litter=11):
    """
    Creates a boxplot comparing overall efficiency of Baited vs Unbaited training trials.
    """
    # Load training trials
    converted_path = os.path.join(working_folder, f'litter_{litter}_processed_cm.json')
    training_trials, probe_trials = st.load_trials(converted_path, kind='Training', include_probes=True, number=1)

    plot_data = []

    for trial in probe_trials:
        df = trial[15]
        rewarded_well = trial[8]
        baited_status = trial[16] # 1 for baited, 0 for unbaited

        # Skip invalid/empty trials
        if df is None or df.empty:
            continue

        efficiency = st.calculate_efficiency(df, rewarded_well)

        # Label the condition
        condition = "Baited" if baited_status == 1 else "Unbaited"

        plot_data.append({
            'Efficiency': efficiency,
            'Condition': condition,
            'Age': trial[6],
            'Subject': trial[0]
        })

    df_plot = pd.DataFrame(plot_data)

    # Generate the plot
    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")

    sns.boxplot(
        data=df_plot,
        x='Condition',
        y='Efficiency',
        palette='Set2',
        showfliers=False # Set to True if you want to see extreme wandering paths
    )

    plt.title(f'Overall Training Efficiency: Baited vs Unbaited (Litter {litter})')
    plt.ylabel('Efficiency (Real/ Ideal Path)')
    plt.xlabel('Trial Condition')

    # Adjust this limit depending on how massive your unbaited outliers are
    plt.ylim(0, 30) 
    plt.tight_layout()
    plt.show()

    return df_plot

#%% -- B vs U Boxplot (across ages, one litter)
def plot_baited_vs_unbaited_across_ages(working_folder, litter=11):
    """
    Creates a boxplot comparing efficiency of Baited vs Unbaited training trials across ages.
    """
    # Load training trials
    converted_path = os.path.join(working_folder, f'litter_{litter}_processed_cm.json')
    training_trials, _ = st.load_trials(converted_path, kind='Training', include_probes=False)
    
    plot_data = []
    
    for trial in training_trials:
        df = trial[15]
        rewarded_well = trial[8]
        baited_status = trial[16] # 1 for baited, 0 for unbaited[cite: 2, 3]
        
        # Skip invalid/empty trials
        if df is None or df.empty:
            continue
            
        efficiency = st.calculate_efficiency(df, rewarded_well)
        
        # Label the condition
        condition = "Baited" if baited_status == 1 else "Unbaited"
        
        plot_data.append({
            'Efficiency': efficiency,
            'Condition': condition,
            'Age': trial[6], # Extracts the age[cite: 2, 3]
            'Subject': trial[0]
        })
        
    df_plot = pd.DataFrame(plot_data)
    
    # Sort by age to ensure the x-axis is in chronological order
    df_plot = df_plot.sort_values(by='Age')
    
    # Generate the plot
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    sns.boxplot(
        data=df_plot,
        x='Age',
        y='Efficiency',
        hue='Condition',
        palette='Set2',
        showfliers=False # Hides extreme outliers for a cleaner look at the median trends
    )
    
    plt.title(f'Training Efficiency Across Ages: Baited vs Unbaited (Litter {litter})')
    plt.ylabel('Efficiency (Real / Ideal Path)')
    plt.xlabel('Age (days)')
    
    # Place the legend outside the plot area
    plt.legend(title='Trial Condition', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust y-limit as needed based on your data spread
    plt.ylim(0, 15) 
    
    plt.tight_layout()
    plt.show()
    
    return df_plot

#%% -- Baited vs unbaited gap across age (new-paradigm litters)
def plot_baited_gap(
    working_folder,
    litters=(9, 10, 11),
    max_trial=None,
    min_valid=3,
    use_median=True,
    min_ideal_cm=8.0,
    normalise=True,
    min_baited_eff=1.2,
    show_group_median=True,
    min_median_n=3,
    jitter=0.12,
    figsize=(15, 7),
    ylim=None,
    title=None
):
    """
    Baited vs unbaited path efficiency, expressed as a per-animal-day gap
    plotted against postnatal age.

    Within each training session, trials are split by the `Baited` flag and
    summarised separately. The plotted value is either the raw gap
    (unbaited - baited) or, with `normalise=True`, the proportional gap
    (unbaited - baited) / baited.

    Efficiency is real/ideal, so a POSITIVE gap means unbaited trials were less
    efficient than baited ones -- consistent with the animal being helped by
    odour on baited trials. If pups shift from odour to distal-cue navigation
    with age, the gap should shrink toward zero.

    Only new-paradigm litters (>= 9) are meaningful here: under the old
    paradigm the first ten trials were baited and the last ten unbaited, so
    condition is confounded with position in the session.

    Parameters
    ----------
    working_folder : str
        Folder holding 'litter_{n}_processed_cm.json' files.
    litters : tuple, optional
        Litters to include. The default is (9, 10, 11).
    max_trial : int, optional
        Use only trials numbered <= this. The default is None (all trials).
    min_valid : int, optional
        Minimum usable trials in EACH condition per session. The default is 3.
    use_median : bool, optional
        Median (True) or mean (False) within a condition. The default is True.
    min_ideal_cm : float, optional
        Drop trials starting nearer than this to the reward. The default is 8.
    normalise : bool, optional
        Plot the proportional gap rather than the raw difference. Default True.
    min_baited_eff : float, optional
        Skip animal-days whose baited value falls below this, since a
        near-1 denominator inflates the ratio. The default is 1.2.
    min_median_n : int, optional
        Suppress the group median where fewer animals contribute. Default 3.

    Returns
    -------
    fig : Figure
    df_out : DataFrame, one row per animal-day.
    """
    from io import StringIO

    agg = np.median if use_median else np.mean
    stat = 'median' if use_median else 'mean'

    rows, dropped, denom_skipped = [], [], 0
    subjects_of = defaultdict(list)
    balance = []          # (n_baited, n_unbaited) per session, for the report

    for litter in litters:
        path = os.path.join(working_folder, f"litter_{litter}_processed_cm.json")
        if not os.path.exists(path):
            print(f"Missing file for litter {litter}, skipping.")
            continue
        with open(path, 'r') as f:
            data = json.load(f)

        for subject in data:
            name = subject['Name']
            if name not in subjects_of[litter]:
                subjects_of[litter].append(name)

            for session in subject['Sessions']:
                if session['Type'] != 'Training':
                    continue
                well = session['Rewarded well']
                b_eff, u_eff = [], []

                for trial in session['Trials']:
                    if max_trial is not None and trial['Number'] > max_trial:
                        continue
                    sf, ef = trial['Start frame'], trial['End frame']
                    if pd.isna(sf) or pd.isna(ef) or ef <= sf:
                        continue
                    df = pd.read_json(StringIO(trial['Position']))
                    if df.empty:
                        continue
                    ideal = _ideal_distance(df, well)
                    if not np.isfinite(ideal) or ideal < min_ideal_cm:
                        continue
                    eff = st.calculate_efficiency(df, well)
                    (b_eff if int(trial['Baited']) == 1 else u_eff).append(eff)

                balance.append((len(b_eff), len(u_eff)))

                if len(b_eff) < min_valid or len(u_eff) < min_valid:
                    dropped.append((name, session['Age'], len(b_eff), len(u_eff)))
                    continue

                b_val, u_val = agg(b_eff), agg(u_eff)
                if normalise and b_val < min_baited_eff:
                    denom_skipped += 1
                    continue

                rows.append({
                    'Subject': name, 'Litter': litter, 'Age': session['Age'],
                    'Environment': session['Environment'],
                    'Baited': b_val, 'Unbaited': u_val,
                    'n_baited': len(b_eff), 'n_unbaited': len(u_eff),
                    'Gap': u_val - b_val,
                    'RelGap': (u_val - b_val) / b_val
                })

    if balance:
        nb = np.mean([b for b, _ in balance])
        nu = np.mean([u for _, u in balance])
        print(f"Mean usable trials per session: {nb:.1f} baited, {nu:.1f} unbaited.")
    if dropped:
        print(f"Dropped {len(dropped)} animal-days with <{min_valid} usable "
              f"trials in a condition.")
    if denom_skipped:
        print(f"Skipped {denom_skipped} animal-days with baited efficiency "
              f"<{min_baited_eff} (unstable denominator).")
    if not rows:
        print("No usable animal-days.")
        return None

    df_out = pd.DataFrame(rows)
    ycol = 'RelGap' if normalise else 'Gap'

    litter_cmaps = {9: 'Blues', 10: 'Greens', 11: 'Reds'}
    subject_colors, subject_litter = {}, {}
    for litter in litters:
        names = sorted(subjects_of[litter])
        if not names:
            continue
        cmap = plt.get_cmap(litter_cmaps.get(litter, 'Greys'))
        for name, shade in zip(names, cmap(np.linspace(0.45, 0.9, len(names)))):
            subject_colors[name] = shade
            subject_litter[name] = litter

    markers = {'Square': 's', 'Circle': 'o'}
    ages = sorted(df_out['Age'].unique())
    age_to_x = {a: i for i, a in enumerate(ages)}

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=figsize)
    ax.axhline(0, color='black', linestyle='--', linewidth=1.5, zorder=1)

    rng = np.random.default_rng(0)
    for _, r in df_out.iterrows():
        xi = age_to_x[r['Age']] + rng.uniform(-jitter, jitter)
        ax.scatter(xi, r[ycol],
                   color=subject_colors.get(r['Subject'], 'gray'),
                   marker=markers.get(r['Environment'], 'o'),
                   s=70, edgecolor='black', linewidth=0.4, alpha=0.85,
                   zorder=3, label=f"{r['Subject']} (L{r['Litter']})")

    if show_group_median:
        med = df_out.groupby('Age')[ycol].median()
        cnt = df_out.groupby('Age')[ycol].size()
        keep = [a for a in med.index if cnt[a] >= min_median_n]
        if keep:
            ax.plot([age_to_x[a] for a in keep], [med[a] for a in keep],
                    color='black', linewidth=2.5, marker='D', markersize=7,
                    alpha=0.9, zorder=4,
                    label=f"Median across animals (n>={min_median_n})")

    ax.set_xlabel('Age (days)')
    ax.set_ylabel(
        f"Proportional gap (unbaited - baited) / baited, {stat}"
        if normalise else f"Gap in efficiency (unbaited - baited), {stat}")
    ax.set_title(title or
                 "Baited vs unbaited gap across age "
                 f"(litters {', '.join(map(str, litters))}; "
                 "positive = unbaited worse)")
    ax.set_xticks(np.arange(len(ages)))
    ax.set_xticklabels([f"P{a}" for a in ages], rotation=45)
    if ylim is not None:
        ax.set_ylim(ylim)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    order = sorted(by_label,
                   key=lambda l: (99 if l.startswith('Median') else
                                  subject_litter.get(l.split(' (')[0], 98), l))
    ax.legend([by_label[l] for l in order], order, title='Animal (litter)',
              bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)

    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return fig, df_out

#%% Summary Boxplots
def _get_session_block_data(trials):
    """
    Helper function to extract the First 5 and Last 5 trials from each session,
    calculate their path efficiency, and format them for the summary boxplots.
    """
    session_trials = defaultdict(list)
    plot_data = []

    # Group trials by Subject and Date
    for trial in trials:
        subject = trial[0]
        date = trial[5]
        session_trials[(subject, date)].append(trial)

    for (subject, date), s_trials in session_trials.items():
        # Sort chronologically by trial number
        s_trials.sort(key=lambda x: x[10])

        # Slice the blocks
        first_5 = s_trials[:5]
        last_5 = s_trials[-5:] if len(s_trials) >= 5 else s_trials

        # Calculate efficiency for each block
        for block, label in [(first_5, 'First 5'), (last_5, 'Last 5')]:
            for t in block:
                df = t[15]
                rewarded_well = t[8]
                
                # Skip broken trajectories
                if df is not None and not df.empty:
                    eff = st.calculate_efficiency(df, rewarded_well)
                    plot_data.append({
                        'Subject': subject,
                        'Efficiency': eff,
                        'Trial Group': label,
                        'Condition': label 
                    })
                    
    return plot_data
def plot_all_animals_boxplot(working_folder):
    """
    Generates a boxplot aggregating First 5 vs Last 5 across all combined sessions.
    """
    merged_data_path = os.path.join(working_folder, 'combined_litters_9_10_11_.json')
    all_trials, _ = st.load_trials(merged_data_path, kind='Training', include_probes=False)

    plot_data = _get_session_block_data(all_trials)
    df_plot = pd.DataFrame(plot_data)
    
    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")

    sns.boxplot(
        data=df_plot,
        x='Trial Group',
        y='Efficiency',
        hue='Condition',
        palette='Set2',
        order=['First 5', 'Last 5'],
        showfliers=False
    )

    plt.title('Overall Efficiency: First 5 vs. Last 5 Trials (Litters 9-11)')
    plt.xlabel('Trial Block')
    plt.ylabel('Efficiency (Real / Ideal Path)')
    plt.legend(title='Condition')
    plt.tight_layout()
    plt.show()


# --- Function 2: Per litter efficiency boxplot + summary

def plot_litter_summary_grid(working_folder, litters=[9, 10, 11]):
    """
    Generates a 2x2 grid of box plots for Litters 9, 10, 11 + a summary plot.
    Saves the combined figure to the working folder.
    """
    # Create a 2x2 grid of subplots (4 plots total)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    all_litters_plot_data = []

    # Plot each litter in the first 3 subplots
    for i, litter in enumerate(litters):
        converted_path = os.path.join(working_folder, f'litter_{litter}_processed_cm.json')
        if not os.path.exists(converted_path):
            continue
            
        all_trials, _ = st.load_trials(converted_path, kind='Training', include_probes=False)
        plot_data = _get_session_block_data(all_trials)
        all_litters_plot_data.extend(plot_data) # Accumulate for the summary

        df_plot = pd.DataFrame(plot_data)

        sns.set_style("whitegrid")
        sns.boxplot(
            data=df_plot,
            x='Trial Group',
            y='Efficiency',
            hue='Condition',
            palette='Set2',
            order=['First 5', 'Last 5'],
            showfliers=False, # Hides extreme outliers to keep focus on the medians
            ax=axes[i]
        )
        axes[i].set_title(f'Litter {litter}')
        axes[i].set_xlabel('Trial Block')
        axes[i].set_ylabel('Efficiency (Real / Ideal)')
        axes[i].legend(title='Condition', loc='upper right')

    # Plot the summary in the 4th subplot
    df_summary = pd.DataFrame(all_litters_plot_data)
    sns.boxplot(
        data=df_summary,
        x='Trial Group',
        y='Efficiency',
        hue='Condition',
        palette='Set2',
        order=['First 5', 'Last 5'],
        showfliers=False,
        ax=axes[3]
    )
    axes[3].set_title('Summary (Litters 9-11)')
    axes[3].set_xlabel('Trial Block')
    axes[3].set_ylabel('Efficiency (Real / Ideal)')
    axes[3].legend(title='Condition', loc='upper right')

    plt.tight_layout()
    
    # Save the combined figure
    output_path = os.path.join(working_folder, 'litter_summary_grid_L9_L11.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()


    
#-- Training Q1 vs Probe P1 boxplots with difference line (NP)


# ============================================================
#  ARCHIVE / other functions (not in callback list)
# ============================================================

def plot_trial_completion(filepath, subjects):
    """
    Plot the percentage of completed vs failed probe trials per subject.
    This function computes the number of completed and failed probe trials for
    each subject, expressed as a percentage. It then plots the breakdown
    using a grouped barplot. A trial is considered completed if both start
    and end frames are present; otherwise, it is marked as failed.

    Parameters
    ----------
    filepath : String
        Path to the JSON file for the litter.
    subjects : List
        List of subject names to plot.

    Returns
    -------
    fig : Figure
        Matplotlib Figure object containing the produced figure.

    """

    with open(filepath, 'r') as json_file:
        data = json.load(json_file)
        
    completed_counts = []
    failed_counts = []

    for subject in subjects:
        # Find subject entry from dataset
        subject_data = next((s for s in data if s["Name"] == subject), None)
        if not subject_data:
            # No data for this subject
            completed_counts.append(0)
            failed_counts.append(0)
            continue

        # Collect all probe trials across sessions
        probe_trials = []
        for session in subject_data["Sessions"]:
            if session["Type"] == "Probe":
                probe_trials.extend(session["Trials"])

        total_trials = len(probe_trials)
        if total_trials == 0:
            completed_counts.append(0)
            failed_counts.append(0)
            continue

        # Count as completed if start and end frames are not NaN
        completed = sum(
            1 for trial in probe_trials
            if not (pd.isna(trial["Start frame"]) or pd.isna(trial["End frame"]))
        )
        failed = total_trials - completed

        # Convert to percentages
        completed_counts.append((completed / total_trials) * 100)
        failed_counts.append((failed / total_trials) * 100)

    # Create grouped barplot
    # Create figure
    fig = plt.figure(figsize=(8, 5))
    x = np.arange(len(subjects))

    # Plot the first layer
    plt.bar(x, completed_counts,
            label="Completed",
            color="tab:blue")

    # Plot the second layer on top of the first
    plt.bar(x, failed_counts,
            bottom=completed_counts, # This is the "Stacking" logic
            label="Failed",
            color="tab:red")

    # FIX: Map the numbers in 'x' to the names in 'subjects'
    plt.xticks(x, subjects, rotation=0) 
    plt.xlabel("Subjects")
    plt.ylabel("Percentage of Trials")
    plt.title("Percentage of Completed and Failed Probe Trials per Subject")
    plt.legend(title='Trial Type', loc='center left', bbox_to_anchor = (1, 0.8))
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout() # Added to prevent label cutoff
    plt.show()

    
    return fig

def plot_efficiency_ratio_per_subject(trials):
    """
    Plots the trajectory efficiency over trials for each subject,
    with each line representing a different age.

    Parameters:
    - trials (list): List of trial tuples.
    - circle_radius_cm (float): Circle radius for computing distances.
    - square_side_cm (float): Square side for computing distances.
    """

    subject_efficiency = defaultdict(lambda: defaultdict(list))
    subject_environment = {}  

    # Compute efficiencies
    for trial in trials:
        subject, dob, sex, session_type, env, date, age, rotation, rewarded_well, session_center, trial_num, start_frame, end_frame, fx, fy, df, baited = trial
        if df is None or df.empty:
            continue
        if subject not in subject_environment:
            subject_environment[subject] = env  

        efficiency = st.calculate_efficiency(df, rewarded_well)
        subject_efficiency[subject][age].append((trial_num, efficiency))

    figs = []
    # Plot per subject
    for subject, age_trials in subject_efficiency.items():
        fig = plt.figure(figsize=(10, 6))

        all_ages = sorted(age_trials.keys())
        norm = mcolors.Normalize(vmin=min(all_ages), vmax=max(all_ages))
        cmap = mcolors.LinearSegmentedColormap.from_list("orange_purple", ["#ff7f0e", "#9467bd"])

        for age, trial_data in sorted(age_trials.items()):
            trial_data.sort()
            trial_nums = [t[0] for t in trial_data]
            efficiencies = [t[1] for t in trial_data]
            color = cmap(norm(age))
            plt.plot(trial_nums, efficiencies, marker="o", label=f"Age {age}", color=color)

        plt.axhline(1.0,
                    linestyle="--",
                    color="red",
                    label="Perfect efficiency")
        plt.xlabel("Trial Number")
        plt.ylabel("Efficiency (Ideal / Real Path)")
        plt.title(f"Trajectory Efficiency by Trial — Subject {subject} ({subject_environment[subject]})")
        all_trial_nums = [t[0] for age_vals in age_trials.values() for t in age_vals]
        plt.xticks(sorted(set(all_trial_nums)))
        plt.ylim(0, 20)
        plt.legend(title="Age", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.show()
        figs.append(fig)
        
    return figs

def make_boxplot(df, labels):
    """
    Helper function for boxplot_quantiles().
    Produces the figure using the data in df.

    Parameters
    ----------
    df : DataFrame
        Pandas DataFrame.
    labels : Dict
        Dictionary containing the plot labels and their order.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    label_keys = list(labels.keys())

    sns.set_style('white')
    fig = plt.figure(figsize=(12, 6))
    
    unique_quartiles = list(df['Quartile'].unique())

    def quartile_sort_key(q):
        # try:
        #     env_part, label_part = q.split(' - ', 1)
        # except ValueError:
        #     env_part, label_part = q, ''
        # return (env_part, labels.get(label_part, 99))
        
        if " - " in q:
            env_part, label_part = q.split(" - ", 1)
        else:
            env_part, label_part = "", q
        return (env_part, labels.get(label_part, 99))


    hue_order = sorted(unique_quartiles, key=quartile_sort_key)

    sns.boxplot(
        data=df,
        x='Age',
        y='Efficiency',
        hue='Quartile',
        hue_order=hue_order,
        palette='Set2'
    )

# Create display names for title - updated 
    display_names = {
        'T1': 'Training Q1',
        'T2': 'Training Q2',
        'T3': 'Training Q3',
        'T4': 'Training Q4',
        'P1': 'Probe Q1',
        'P2': 'Probe Q2'}
# Which title to pick for place 1 and 2 - updated      
    title_part1 = display_names.get(label_keys[0],
                                    label_keys[0]) #second one is backup
    title_part2 = display_names.get(label_keys[1],
                                    label_keys[1]) #second one is backup

    plt.xlabel('Age (days)')
    plt.ylabel('Path efficiency')
    plt.title(f"{title_part1} vs {title_part2}")
    plt.ylim(0, 1.1)
    plt.legend(title='Quantiles')
    plt.tight_layout()
    plt.show()
    
    return fig

# FIXME there are differences in some datapoints when env_split = True or False that shouldn't be there!

def boxplot_quantiles(population_trials,
                      q1,
                      q2,
                      min_age=None,
                      max_age=None,
                      env_split=True):
    """
    Produces a figure showing the path efficiency distribution for the
    specified quantiles of training and probe sessions for all subjects
    across age.
    The quantiles are quartiles for training sessions and halves for probe
    sessions, so that each quantile is 5 trials long.
    The quantiles are specified as 'xn', where 'x' is the session type
    (training or probe) and 'n' the quantile number.
    
    Example
    ----------
    fig = boxplot_quantiles(trial_tuples, 't4', 'p2')
        Plots the last 5 trials of training sessions vs the last 5 trials of
        probe sessions for all trials in trial_tuples.
    
    Parameters
    ----------
    population_trials : List.
        List of 15-element tuples.
    q1 : String
        What quantile to plot first.
    q2 : String
        What quantile to plot second.
    min_age : Int, optional
        Lower cutoff for subject age to plot. The default is None.
    max_age : Int, optional
        Higher cutoff for subject age to plot. The default is None.
    env_split : Bool, optional
        Whether to split boxplots for environments. The default is True.
    Returns
    -------
    fig : Figure
        Matplotlib Figure object.

    """
    # t1 is q1 for training
    # p1 is q1 for probe
    
    quant1 = list(q1)
    quant2 = list(q2)
    
    if len(quant1) > 2:
        raise ValueError('Parameter q1 is not correct!')
        
    elif len(quant2) > 2:
        raise ValueError('Parameter q2 is not correct!')

    quant1_kind = quant1[0]
    quant2_kind = quant2[0]
    
    session_trials = defaultdict(list)
    data_for_plot = []
    
    if env_split:
        # Group all trials by (subject, environment, date)
        for trial in population_trials:
            session_key = (trial[0], trial[4], trial [16], trial[5])
            session_trials[session_key].append(trial)
    
        for session_key, trials in session_trials.items():
            subject, environment, baited, date_str = session_key
            
            baited_label = 'Baited' if baited == 1 else 'Unbaited'
            
            quant1_trials = []
            quant2_trials = []

            # TRAINING
            if quant1_kind == 't' or quant2_kind == 't':
            
                trials_to_get = [t for t in trials if t[3] == 'Training']
                trials_to_get = sorted(trials_to_get, key=lambda t: t[9])
                
                if quant1_kind == 't':
                    qtile = int(quant1[1])
                    
                    if qtile == 1:
                        quant1_trials = trials_to_get[:5]  # first 5 trials only
                    elif qtile == 2:
                        quant1_trials = trials_to_get[5:10]
                    elif qtile == 3:
                        quant1_trials = trials_to_get[10:15]
                    elif qtile == 4:
                        quant1_trials = trials_to_get[-5:]  # last 5 trials only
                    else:
                        raise ValueError('Quantile number for q1 not correct!')
                
                if quant2_kind == 't':
                    qtile = int(quant2[1])
                    
                    if qtile == 1:
                        quant2_trials = trials_to_get[:5]  # first 5 trials only
                    elif qtile == 2:
                        quant2_trials = trials_to_get[5:10]
                    elif qtile == 3:
                        quant2_trials = trials_to_get[10:15]
                    elif qtile == 4:
                        quant2_trials = trials_to_get[-5:]  # last 5 trials only
                    else:
                        raise ValueError('Quantile number for q2 not correct!')
    
            # TODO implement handling of rotation
            # PROBE (rotation == 0)
            if quant1_kind == 'p' or quant2_kind == 'p':
    
                trials_to_get = [t for t in trials if t[3] == 'Probe' and t[7] == 0]
                trials_to_get = sorted(trials_to_get, key=lambda t: t[9])
                
                if quant1_kind == 'p':
                    qtile = int(quant1[1])
                    
                    if qtile == 1:
                        quant1_trials = trials_to_get[:5]  # first 5 trials only
                    elif qtile == 2:
                        quant1_trials = trials_to_get[5:10] # last 5 trials only
                    else:
                        raise ValueError('Quantile number for q1 not correct!')
                
                if quant2_kind == 'p':
                    qtile = int(quant2[1])
                    
                    if qtile == 1:
                        quant2_trials = trials_to_get[:5]  # first 5 trials only
                    elif qtile == 2:
                        quant2_trials = trials_to_get[5:10] # last 5 trials only
                    else:
                        raise ValueError('Quantile number for q2 not correct!')
    
            # Make legend
            quartile_sets = [(quant1_trials, q1.upper(),
                              'Training' if quant1_kind == 't' else 'Probe'),
                             (quant2_trials, q2.upper(),
                              'Training' if quant2_kind == 't' else 'Probe'),]
    
            for trial_set, label, source in quartile_sets:
                if len(trial_set) <= 0:
                    continue
    
                efficiencies = []
                for trial in trial_set:
                    age = trial[6]
    
                    # Apply age filters
                    if min_age is not None and age < min_age:
                        continue
                    if max_age is not None and age > max_age:
                        continue
    
                    start_frame = trial[11]
                    end_frame = trial[12]
                    df = trial[15]
                    rewarded_well = trial[8]
    
                    if pd.isna(start_frame) or pd.isna(end_frame) or df is None or df.empty:
                        continue
    
                    efficiency = st.calculate_efficiency(df, rewarded_well)
                    efficiencies.append(efficiency)
    
                if efficiencies:
                    # Age logic: shift probe trials to previous training age
                    raw_age = trial_set[0][6]
                    adjusted_age = raw_age - 1 if source == 'Probe' else raw_age
    
                    data_for_plot.append({
                        'Age': adjusted_age,
                        'Efficiency': np.mean(efficiencies),
                        'Environment': environment,
                        'Quartile': f'{environment} ({baited_label}) - {label}'
                    })
    else:
        # Group all trials by (subject, date)
        for trial in population_trials:
            session_key = (trial[0], trial[16], trial[5])
            session_trials[session_key].append(trial)
    
        for session_key, trials in session_trials.items():
            subject, baited, date_str = session_key
            baited_label = 'Baited' if baited == 1 else 'Unbaited'
            
            quant1_trials = []
            quant2_trials = []
            
            # TRAINING
            if quant1_kind == 't' or quant2_kind == 't':
            
                trials_to_get = [t for t in trials if t[3] == 'Training']
                trials_to_get = sorted(trials_to_get, key=lambda t: t[9])
                
                if quant1_kind == 't':
                    qtile = int(quant1[1])
                    
                    if qtile == 1:
                        quant1_trials = trials_to_get[:5]  # first 5 trials only
                    elif qtile == 2:
                        quant1_trials = trials_to_get[5:10]
                    elif qtile == 3:
                        quant1_trials = trials_to_get[10:15]
                    elif qtile == 4:
                        quant1_trials = trials_to_get[-5:]  # last 5 trials only
                    else:
                        raise ValueError('Quantile number for q1 not correct!')
                
                if quant2_kind == 't':
                    qtile = int(quant2[1])
                    
                    if qtile == 1:
                        quant2_trials = trials_to_get[:5]  # first 5 trials only
                    elif qtile == 2:
                        quant2_trials = trials_to_get[5:10]
                    elif qtile == 3:
                        quant2_trials = trials_to_get[10:15]
                    elif qtile == 4:
                        quant2_trials = trials_to_get[-5:]  # last 5 trials only
                    else:
                        raise ValueError('Quantile number for q2 not correct!')
    
            # TODO implement handling of rotation
            # PROBE (rotation == 0)
            if quant1_kind == 'p' or quant2_kind == 'p':
    
                trials_to_get = [t for t in trials if t[3] == 'Probe' and t[7] == 0]
                trials_to_get = sorted(trials_to_get, key=lambda t: t[9])
                
                if quant1_kind == 'p':
                    qtile = int(quant1[1])
                    
                    if qtile == 1:
                        quant1_trials = trials_to_get[:5]  # first 5 trials only
                    elif qtile == 2:
                        quant1_trials = trials_to_get[5:10] # last 5 trials only
                    else:
                        raise ValueError('Quantile number for q1 not correct!')
                
                if quant2_kind == 'p':
                    qtile = int(quant2[1])
                    
                    if qtile == 1:
                        quant2_trials = trials_to_get[:5]  # first 5 trials only
                    elif qtile == 2:
                        quant2_trials = trials_to_get[5:10] # last 5 trials only
                    else:
                        raise ValueError('Quantile number for q2 not correct!')
    
            # Make legend
            quartile_sets = [(quant1_trials, q1.upper(),
                              'Training' if quant1_kind == 't' else 'Probe'),
                             (quant2_trials, q2.upper(),
                              'Training' if quant2_kind == 't' else 'Probe'),]
    
            for trial_set, label, source in quartile_sets:
                if len(trial_set) <= 0:
                    continue
    
                efficiencies = []
                for trial in trial_set:
                    age = trial[6]
    
                    # Apply age filters
                    if min_age is not None and age < min_age:
                        continue
                    if max_age is not None and age > max_age:
                        continue
    
                    start_frame = trial[11]
                    end_frame = trial[12]
                    df = trial[15]
                    rewarded_well = trial[8]
                    if pd.isna(start_frame) or pd.isna(end_frame) or df is None or df.empty:
                        continue
    
                    efficiency = st.calculate_efficiency(df, rewarded_well)
                    efficiencies.append(efficiency)
    
                if efficiencies:
                    # Age logic: shift probe trials to previous training age
                    raw_age = trial_set[0][6]
                    adjusted_age = raw_age - 1 if source == 'Probe' else raw_age
    
                    data_for_plot.append({
                        'Age': adjusted_age,
                        'Efficiency': np.mean(efficiencies),
                        'Environment': 'All', #added this in
                        'Quartile': label
                    })
    
    df = pd.DataFrame(data_for_plot)
    
    # Ordering the plot labels
    label_order_map = {
        q1.upper(): 1,
        q2.upper(): 2
        }
    
    fig = make_boxplot(df, label_order_map)
    plt.ylim(0, 1.1)
    
    return fig

#%% 3* 7 trajectory grid (per age band)
def plot_trajectory_grids(
    working_folder,
    litters=(9, 10, 11),
    band_representatives=None,     # {'Pre-weaning': name, 'Peri-weaning': name, 'Post-weaning': name}
    age_bands=((17, 20, 'Pre-weaning'),
               (21, 24, 'Peri-weaning'),
               (25, 32, 'Post-weaning')),
    n_days=7,
    trial_number=1,
    figsize=(22, 9)
):
    """
    Two 3x7 trajectory grids across one or more litters: rows are one
    representative animal per ENTRY age band, columns are that animal's first
    `n_days` sessions. One figure = first TRAINING trial per day, the other =
    first PROBE trial per day.

    NOTE: a row's band label reflects the animal's ENTRY age; the panels show
    later sessions, so the per-panel P{age} is the true age displayed and may
    exceed the entry band (e.g. an animal that started pre-weaning is older by
    the time its probes run). Pin `band_representatives` for report figures
    rather than trusting auto-selection.
    """
    from io import StringIO

    # --- Load all requested litters into one pool ---
    by_name = {}
    litter_of = {}
    for litter in litters:
        path = os.path.join(working_folder, f"litter_{litter}_processed_cm.json")
        if not os.path.exists(path):
            print(f"Missing litter {litter}, skipping.")
            continue
        with open(path, 'r') as f:
            for subj in json.load(f):
                by_name[subj['Name']] = subj
                litter_of[subj['Name']] = litter

    if not by_name:
        print("No litters loaded.")
        return None

    def _date_key(d):
        dd, mm, yy = d.split('_')
        return (int(yy), int(mm), int(dd))

    def _band_of(age):
        for lo, hi, label in age_bands:
            if lo <= age <= hi:
                return label
        return None

    # --- Entry age of each animal (age of its earliest training session) ---
    entry = {}
    for name, subj in by_name.items():
        train = [s for s in subj['Sessions'] if s['Type'] == 'Training']
        if train:
            entry[name] = min(train, key=lambda s: _date_key(s['Date']))['Age']

    # --- Auto-select representatives per band, across the whole pool ---
    if band_representatives is None:
        band_representatives = {}
        for lo, hi, blabel in age_bands:
            cand = sorted([n for n, a in entry.items() if lo <= a <= hi],
                          key=lambda n: entry[n])
            if cand:
                band_representatives[blabel] = cand[0]
        print("Auto-selected representatives (entry age):",
              {b: f"{n} (litter {litter_of[n]}, entry P{entry[n]})"
               for b, n in band_representatives.items()})

    def _draw(ax, df, rewarded_well, env, baited):
        path_color = 'red' if baited else 'blue'           # <-- the actual change
        reward_x, reward_y = well_locations_cm[rewarded_well - 1]
        wx = [x for x, y in well_locations_cm]
        wy = [y for x, y in well_locations_cm]
        ax.scatter(wx, wy, color="gray", s=18, zorder=2)
        ax.scatter(reward_x, reward_y, color="red", s=60, zorder=3)
        if df is not None and not df.empty:
            ax.plot(df["x"], df["y"], color=path_color, alpha=0.7,
                    linewidth=1.2, zorder=4)
            sx, sy = df.iloc[0]["x"], df.iloc[0]["y"]
            ax.scatter(sx, sy, color="lime", edgecolor="black", linewidth=1,
                       marker="*", s=90, zorder=5)
            
        if env == "Circle":
            ax.add_patch(plt.Circle((0, 0), circle_radius_cm, color="black",
                                    fill=False, linewidth=1.2))
        elif env == "Square":
            side = square_side_cm + 7
            ax.add_patch(plt.Rectangle((-side/2, -side/2), side, side,
                                       color="black", fill=False, linewidth=1.2))
        ax.set_xlim(-40, 40); ax.set_ylim(-40, 40)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

    def _panel_data(subj, session_type, day_idx):
        sess = sorted([s for s in subj['Sessions']
                       if s['Type'] == session_type
                       and not (session_type == 'Probe' and s.get('Rotation', 0) == 1)],
                      key=lambda s: _date_key(s['Date']))
        if day_idx >= len(sess):
            return None
        s = sess[day_idx]
        for trial in s['Trials']:
            if trial['Number'] != trial_number:
                continue
            df = pd.read_json(StringIO(trial['Position']))
            if df.empty:
                return None
            baited = int(trial.get('Baited', 0))          # <-- new
            return df, s['Rewarded well'], s['Environment'], s['Age'], baited
        return None

    band_order = [b[2] for b in age_bands]

    def _build_figure(session_type):
        fig, axes = plt.subplots(len(band_order), n_days, figsize=figsize)
        if len(band_order) == 1:
            axes = axes[np.newaxis, :]
        for r, blabel in enumerate(band_order):
            rep = band_representatives.get(blabel)
            for c in range(n_days):
                ax = axes[r][c]
                if rep is None or rep not in by_name:
                    ax.axis('off')
                    continue
                info = _panel_data(by_name[rep], session_type, c)
                if info is None:
                    ax.axis('off')
                    ax.set_title(f"Day {c+1}: no data", fontsize=7, color='gray')
                    continue
                df, well, env, age, baited = info          # <-- fifth name added
                _draw(ax, df, well, env, baited)           # <-- pass it through
                ax.set_title(f"Day {c+1} · P{age} · {env}", fontsize=8)
            # Row label: animal + entry band, so the label can't overclaim the age
            if rep is not None and rep in entry:
                axes[r][0].set_ylabel(
                    f"{rep} (litter {litter_of[rep]})\nentry {blabel} (P{entry[rep]})",
                    fontsize=9, rotation=90, labelpad=12)
            else:
                axes[r][0].set_ylabel(blabel, fontsize=9, rotation=90, labelpad=12)
        pool = ', '.join(map(str, litters))
        fig.suptitle(f"Litters {pool}: {session_type} trial {trial_number} per day "
                     f"— red=baited, blue=unbaited", fontsize=13)
        plt.tight_layout(); plt.subplots_adjust(top=0.92)
        plt.show()
        return fig

    return _build_figure('Training'), _build_figure('Probe')

#%% Speed of learning L9-L11

#%% T1/Tx
def plot_fold_improvement_from_naive_by_band(
    working_folder,
    litters=(9, 10, 11),
    reference_trial=1,
    comparator_trials=tuple(range(2, 11)),   # T2..T10 vs T1
    age_bands=((17, 20, 'Pre-weaning (P17-20)'),
               (21, 24, 'Peri-weaning (P21-24)'),
               (25, 32, 'Post-weaning (P25-32)')),
    session_types=('Training',),
    min_ideal_cm=8.0,
    exclude_rotation=True,
    use_median=True,
    band_colors=None,
    figsize=(11, 6),
    ylim=None,
    title=None
):
    """
    Fold-improvement from naive: within each session, the ratio of naive-trial
    efficiency to each comparator (T1 / Tx), expressed as log2 fold-change.
    0 = no change from naive; +1 = twice as efficient as naive; +2 = 4x; etc.
    Because it's a RATIO, it measures PROPORTIONAL improvement and is comparable
    across bands regardless of how bad the naive trial was -- removing the
    "pre-weaning had more absolute room to fall" confound of the difference plot.

    log2(T1/Tx) computed per session; averaged within animal; summarised across
    animals per band (median + IQR default). One line per band.

    Returns
    -------
    fig : Figure
    per_animal : one row per animal per band per comparator.
    """
    from io import StringIO

    if band_colors is None:
        band_colors = ['tab:green', 'tab:blue', 'tab:orange']

    def _band_of(age):
        for lo, hi, label in age_bands:
            if lo <= age <= hi:
                return label
        return None

    def _eff_of_trial(session, number):
        for trial in session['Trials']:
            if trial['Number'] != number:
                continue
            sf, ef = trial['Start frame'], trial['End frame']
            if pd.isna(sf) or pd.isna(ef) or ef <= sf:
                return None
            df = pd.read_json(StringIO(trial['Position']))
            if df.empty:
                return None
            ideal = _ideal_distance(df, session['Rewarded well'])
            if not np.isfinite(ideal) or ideal < min_ideal_cm:
                return None
            e = st.calculate_efficiency(df, session['Rewarded well'])
            # efficiency is real/ideal >= 1, so always positive; guard anyway
            return e if (e is not None and e > 0) else None
        return None

    rows = []
    for litter in litters:
        path = os.path.join(working_folder, f"litter_{litter}_processed_cm.json")
        if not os.path.exists(path):
            continue
        with open(path, 'r') as f:
            data = json.load(f)

        for subject in data:
            name = subject['Name']
            for s in subject['Sessions']:
                if s['Type'] not in session_types:
                    continue
                if s['Type'] == 'Probe' and exclude_rotation and s.get('Rotation', 0) == 1:
                    continue
                band = _band_of(s['Age'])
                if band is None:
                    continue

                ref = _eff_of_trial(s, reference_trial)   # T1
                if ref is None:
                    continue

                for x in comparator_trials:
                    comp = _eff_of_trial(s, x)
                    if comp is None:
                        continue
                    # log2 fold-improvement: T1 / Tx
                    rows.append({'Subject': name, 'Band': band,
                                 'Comparator': x,
                                 'LogFold': np.log2(ref / comp)})

    if not rows:
        print("No usable sessions.")
        return None
    df_sessions = pd.DataFrame(rows)

    per_animal = (df_sessions.groupby(['Band', 'Subject', 'Comparator'])['LogFold']
                             .mean().reset_index())

    band_labels = [b[2] for b in age_bands]
    color_map = {lbl: band_colors[i % len(band_colors)]
                 for i, lbl in enumerate(band_labels)}
    n_per_band = per_animal.groupby('Band')['Subject'].nunique()

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=figsize)
    ax.axhline(0, color='black', linestyle='--', linewidth=1.5, zorder=1,
               label='No change from naive')

    for band in band_labels:
        bdata = per_animal[per_animal['Band'] == band]
        if bdata.empty:
            continue
        grp = bdata.groupby('Comparator')['LogFold']
        centre = grp.median() if use_median else grp.mean()
        x = centre.index.to_numpy()
        if use_median:
            lo = grp.quantile(0.25); hi = grp.quantile(0.75)
            yerr = np.vstack([centre.values - lo.values, hi.values - centre.values])
        else:
            sem = grp.sem()
            yerr = np.vstack([sem.values, sem.values])
        n = n_per_band.get(band, 0)
        ax.errorbar(x, centre.values, yerr=yerr, marker='o', markersize=6,
                    linewidth=2, capsize=4, color=color_map[band],
                    label=f"{band} (N={n})", zorder=3)

    ax.set_xlabel(f"Trial (compared to naive T{reference_trial})")
    ax.set_ylabel("Fold-improvement from naive, log2(T1 / Tx)\n"
                  "0 = no change, +1 = 2x better, +2 = 4x better")
    ax.set_xticks(list(comparator_trials))
    ax.set_title(title or
                 "Fold-improvement from naive across trials, by age band")
    ax.legend(title='Age band', loc='best', fontsize=8)
    if ylim is not None:
        ax.set_ylim(ylim)
    plt.tight_layout()
    plt.show()

    return fig, per_animal

#%% Testing, will be removed later

def boxplot_training_probe_Q4_vs_Q1(population_trials, min_age=None, max_age=None):
    """
    Boxplot comparing efficiency for:
    - Q4 from Training (last 5 trials)
    - Q1 from corresponding Probe session (first 5 trials, rotation == 0)
    Probe trials are plotted at the age of the preceding training day.

    Parameters:
    - population_trials (list): List of processed trial tuples with metadata and position traces.
    - min_age, max_age (int, optional): Age filter in days.
    """
    session_trials = defaultdict(list)
    data_for_plot = []

    # Group all trials by (subject, environment, date)
    for trial in population_trials:
        session_key = (trial[0], trial[4], trial[5])
        session_trials[session_key].append(trial)

    for session_key, trials in session_trials.items():
        subject, environment, date_str = session_key

        # TRAINING
        training_trials = [t for t in trials if t[3] == "Training"]
        training_trials = sorted(training_trials, key=lambda t: t[9])
        training_Q4 = training_trials[-5:]  # last 5 trials only

        # PROBE (rotation == 0)
        probe_trials = [t for t in trials if t[3] == "Probe" and t[7] == 0]
        probe_trials = sorted(probe_trials, key=lambda t: t[9])
        probe_Q1 = probe_trials[:5]  # first 5 trials only

        quartile_sets = [
            (training_Q4, "Q4 (Training)", "Training"),
            (probe_Q1, "Q1 (Probe)", "Probe"),
        ]

        for trial_set, label, source in quartile_sets:
            if len(trial_set) <= 0:
                continue

            efficiencies = []
            for trial in trial_set:
                age = trial[6]

                # Apply age filters
                if min_age and age < min_age:
                    continue
                if max_age and age > max_age:
                    continue

                start_frame = trial[11]
                end_frame = trial[12]
                df = trial[15]
                rewarded_well = trial[8]

                if pd.isna(start_frame) or pd.isna(end_frame) or df is None or df.empty:
                    continue

                start_x, start_y = df["x"].iloc[0], df["y"].iloc[0]
                reward_x, reward_y = well_locations_cm[rewarded_well - 1]
                ideal_distance = np.sqrt((reward_x - start_x) ** 2 + (reward_y - start_y) ** 2)

                x = df["x"].values
                y = df["y"].values
                real_distance = np.nansum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))

                if real_distance == 0:
                    continue
                if real_distance < ideal_distance:
                    real_distance = ideal_distance

                efficiency = ideal_distance / real_distance
                efficiencies.append(efficiency)

            if efficiencies:
                # Age logic: shift probe trials to previous training age
                raw_age = trial_set[0][5]
                adjusted_age = raw_age - 1 if source == "Probe" else raw_age

                data_for_plot.append({
                    "Age": adjusted_age,
                    "Efficiency": np.mean(efficiencies),
                    "Environment": environment,
                    "Quartile": f"{environment} - {label}"
                })

    # Plotting
    df = pd.DataFrame(data_for_plot)

    sns.set_style("white")
    plt.figure(figsize=(12, 6))

    # Ordering: Q4 Training first, then Q1 Probe
    label_order_map = {
        "Q4 (Training)": 1,
        "Q1 (Probe)": 2
    }
    unique_quartiles = list(df["Quartile"].unique())

    def quartile_sort_key(q):
        try:
            env_part, label_part = q.split(" - ", 1)
        except ValueError:
            env_part, label_part = q, ""
        return (env_part, label_order_map.get(label_part, 99))

    hue_order = sorted(unique_quartiles, key=quartile_sort_key)

    sns.boxplot(
        data=df,
        x="Age",
        y="Efficiency",
        hue="Quartile",
        hue_order=hue_order,
        palette="Set2"
    )

    plt.xlabel("Age (days)")
    plt.ylabel("Efficiency (Ideal / Real Path, cm)")
    plt.title("Training Q4 vs Probe Q1")
    plt.ylim(0, 1.1)
    plt.legend(title="Quartiles")
    plt.tight_layout()
    plt.show()
    
def boxplot_training_probe_Q4_vs_Q1_noenv(population_trials, min_age=None, max_age=None):
    """
    Boxplot comparing efficiency for:
    - Q4 from Training (last 5 trials)
    - Q1 from corresponding Probe session (first 5 trials, rotation == 0)
    Probe trials are plotted at the age of the preceding training day.

    Environment grouping removed — only quartiles are compared.
    """
    session_trials = defaultdict(list)
    data_for_plot = []

    # Group all trials by (subject, date)
    for trial in population_trials:
        session_key = (trial[0], trial[4])  # (subject, date)
        session_trials[session_key].append(trial)

    for session_key, trials in session_trials.items():
        subject, date_str = session_key

        # TRAINING
        training_trials = [t for t in trials if t[2] == "Training"]
        training_trials = sorted(training_trials, key=lambda t: t[9])
        training_Q4 = training_trials[-5:]  # last 5 trials only

        # PROBE (rotation == 0)
        probe_trials = [t for t in trials if t[2] == "Probe" and t[6] == 0]
        probe_trials = sorted(probe_trials, key=lambda t: t[9])
        probe_Q1 = probe_trials[:5]  # first 5 trials only

        quartile_sets = [
            (training_Q4, "Q4 (Training)", "Training"),
            (probe_Q1, "Q1 (Probe)", "Probe"),
        ]

        for trial_set, label, source in quartile_sets:
            if len(trial_set) <= 0:
                continue

            efficiencies = []
            for trial in trial_set:
                age = trial[5]

                # Apply age filters
                if min_age and age < min_age:
                    continue
                if max_age and age > max_age:
                    continue

                start_frame = trial[10]
                end_frame = trial[11]
                df = trial[14]
                rewarded_well = trial[7]

                if pd.isna(start_frame) or pd.isna(end_frame) or df is None or df.empty:
                    continue

                start_x, start_y = df["x"].iloc[0], df["y"].iloc[0]
                reward_x, reward_y = well_locations_cm[rewarded_well - 1]
                ideal_distance = np.sqrt((reward_x - start_x) ** 2 + (reward_y - start_y) ** 2)

                x = df["x"].values
                y = df["y"].values
                real_distance = np.nansum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))

                if real_distance == 0:
                    continue
                if real_distance < ideal_distance:
                    real_distance = ideal_distance

                efficiency = ideal_distance / real_distance
                efficiencies.append(efficiency)

            if efficiencies:
                # Age logic: shift probe trials to previous training age
                raw_age = trial_set[0][5]
                adjusted_age = raw_age - 1 if source == "Probe" else raw_age

                data_for_plot.append({
                    "Age": adjusted_age,
                    "Efficiency": np.mean(efficiencies),
                    "Quartile": label
                })

    # Plotting
    df = pd.DataFrame(data_for_plot)

    sns.set_style("white")
    plt.figure(figsize=(12, 6))

    # Fix quartile order
    hue_order = ["Q4 (Training)", "Q1 (Probe)"]

    sns.boxplot(
        data=df,
        x="Age",
        y="Efficiency",
        hue="Quartile",
        hue_order=hue_order,
        palette="Set2"
    )

    plt.xlabel("Age (days)")
    plt.ylabel("Efficiency (Ideal / Real Path, cm)")
    plt.title("Training Q4 vs Probe Q1")
    plt.ylim(0, 1.1)
    plt.legend(title="Quartiles")
    plt.tight_layout()
    plt.show()
    
def boxplot_training_avg4_efficiency_quartiles(population_trials, start_date=None, end_date=None, quartiles_to_plot=None):
    """
    Creates a boxplot comparing average trajectory efficiency across quartile bins
    (1st to 4th group of 5 trials each) for training sessions, grouped by age and environment.

    Efficiency is calculated as: ideal_path_length / actual_path_length

    Parameters:
    - population_trials (list): List of processed trial tuples
    - start_date, end_date (str, optional): Filter sessions by date ("dd/mm/YYYY")
    - quartiles_to_plot (list, optional): Subset of quartiles to plot, e.g. ["Q1", "Q4"].
      Defaults to all ["Q1", "Q2", "Q3", "Q4"].
    """
    
    from datetime import datetime

    if quartiles_to_plot is None:
        quartiles_to_plot = ["Q1", "Q2", "Q3", "Q4"]

    session_trials = defaultdict(list)
    data_for_plot = []

    # Group trials by session
    for trial in population_trials:
        if trial[2] != "Training":
            continue
        session_key = (trial[0], trial[3], trial[4])
        session_trials[session_key].append(trial)

    # Process each session
    for session_key, trials in session_trials.items():
        subject_name, environment, date_str = session_key
        session_date = datetime.strptime(date_str, "%d_%m_%Y")

        if start_date and session_date < datetime.strptime(start_date, "%d/%m/%Y"):
            continue
        if end_date and session_date > datetime.strptime(end_date, "%d/%m/%Y"):
            continue

        sorted_trials = sorted(trials, key=lambda t: t[9])

        # Define 4 groups of 5 trials each (if possible)
        quartile_sets = [
            (sorted_trials[:5], "Q1"),
            (sorted_trials[5:10], "Q2"),
            (sorted_trials[-10:-5], "Q3"),
            (sorted_trials[-5:], "Q4")
        ]

        for trial_set, label in quartile_sets:
            if label not in quartiles_to_plot:
                continue
            if len(trial_set) < 5:
                continue

            efficiencies = []

            for trial in trial_set:
                start_frame = trial[11]
                end_frame = trial[12]
                df = trial[15]
                rewarded_well = trial[8]

            if pd.isna(start_frame) or pd.isna(end_frame) or df is None or df.empty:
                efficiency = st.calculate_efficiency(df, rewarded_well)
                efficiencies.append(efficiency)

            if efficiencies:
                avg_eff = np.mean(efficiencies)
                age = trial_set[0][5]
                data_for_plot.append({
                    "Age": age,
                    "Efficiency": avg_eff,
                    "Environment": environment,
                    "Trial Position": f"{environment} - {label}"
                })

    # Create DataFrame and plot
    df = pd.DataFrame(data_for_plot)

    sns.set_style("white")
    fig = plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=df,
        x="Age",
        y="Efficiency",
        hue="Trial Position",
        palette="Set2"
    )
    plt.xlabel("Age (days)")
    plt.ylabel("Efficiency (Ideal / Real Path, cm)")
    plt.title("Avg Efficiency in Quartiles")
    plt.ylim(0, 1.1) 
    plt.legend(title="Trial Position")
    plt.tight_layout()
    plt.show()
    
    return fig

def plot_training_and_probes_test(trials_data,
                             probe_trials, 
                             colors=None,
                             separate_color_scale=True):
    #TO DO fix dahsed line
    """
    Plots all training and probe trial trajectories on the same figure with a color-scale legend.
    """
    num_train = len(trials_data)
    num_probe = len(probe_trials)
    num_trials = num_train + num_probe
    
    if num_trials == 0:
        raise ValueError('No trials selected for plotting.')

    # 1. Color Generation Logic
    if colors is None:
        if separate_color_scale:
            train_map = plt.cm.Blues  
            probe_map = plt.cm.Reds
            
            train_colors = [train_map(x) for x in np.linspace(0.4, 0.9, num_train)] if num_train > 0 else []
            probe_colors = [probe_map(x) for x in np.linspace(0.4, 0.9, num_probe)] if num_probe > 0 else []
            
            colors = train_colors + probe_colors #vstack?
        else:
            combined_map = plt.cm.rainbow
            colors = [combined_map(x) for x in np.linspace(0, 1, num_trials)]

    fig = plt.figure(figsize=standard_figsize)

    # Plot static well locations
    well_x = [x for x, y in well_locations_cm]
    well_y = [y for x, y in well_locations_cm]
    plt.scatter(well_x, well_y, color="gray", label="Wells", s=50)

    # Determine environment type from the first trial available
    _, _, _, env, *_ = trials_data[0]

    # 2. Draw Training Trials
    # compare with previous code - order of plot 
    for i, trial in enumerate(trials_data):
        (_, _, _, _, _, _, rotation, rewarded_well, _, trial_number, _, _, _, _, df, baited) = trial
        reward_x, reward_y = well_locations_cm[rewarded_well - 1]
        
        # Target well marker
        plt.scatter([reward_x], [reward_y], color="red", s=150, zorder=2)
        
        # Start Star
        if not df.empty:
            plt.scatter(df["x"].iloc[0], df["y"].iloc[0], color=colors[i], 
                        edgecolors="black", linewidths=1.5, marker="*", s=200, zorder=3)
        
        # Path
        plt.plot(df["x"], df["y"], color=colors[i], alpha=0.7, zorder=4)
        

    # 3. Draw Probe Trials
    # compare with previous code - order of plot
    for j, trial in enumerate(probe_trials):
        (_, _, _, _, _, _, rotation, rewarded_well, _, trial_number, _, _, _, _, df, baited) = trial
        probe_index = num_train + j 
        #TO DO maybe skip multiple colors instead of starting where training left of
        linestyle = "--" if rotation == 1 else "-"
       
        # rotated target well
        if rotation == 1:
            reward_x, reward_y = well_locations_cm[rewarded_well - 1]
            plt.scatter([reward_x], [reward_y], color="blue", s=150, zorder=2)
        
        # Start Star
        if not df.empty:
            plt.scatter(df["x"].iloc[0], df["y"].iloc[0], color=colors[probe_index], 
                        edgecolors="black", linewidths=1.5, marker="*", s=200, zorder=3)
        # Path
        plt.plot(df["x"], df["y"], linestyle=linestyle, linewidth=1.5, 
                 color=colors[probe_index], alpha=0.8, zorder=4)
  

    # Boundary Drawing Logic
    ax = plt.gca()
    if env == "Circle":
        circle = plt.Circle((0, 0),
                            circle_radius_cm,
                            color="black",
                            fill=False,
                            linewidth=1.5)
        ax.add_patch(circle)

    elif env == "Square":
        square_side = square_side_cm + square_pad
        half_side = square_side / 2
        square = plt.Rectangle((-half_side, -half_side),
                               square_side,
                               square_side,
                               color="black",
                               fill=False,
                               linewidth=1.5)
        ax.add_patch(square)

    else:
        raise ValueError('Environment type not correct!')

    # 4. BUILD THE COLOR SCALE LEGEND
    legend_elements = [] # difference in code 
    legend_labels = []

    # Static Markers
    legend_elements.append(mlines.Line2D([], 
                                         [], 
                                         color="none", 
                                         marker="o",
                                         markerfacecolor="gray", 
                                         markersize=10))
    legend_labels.append("Wells")
    
    legend_elements.append(mlines.Line2D([], 
                                         [],
                                         color="none",
                                         marker="*", 
                                         markeredgecolor="black", 
                                         markerfacecolor="gray", 
                                         markersize=12))
    legend_labels.append("Start Position")
    
    legend_elements.append(mlines.Line2D([],
                                         [],
                                         color="none", 
                                         marker="o",
                                         markerfacecolor="red",
                                         markersize=10))
    legend_labels.append("Target Well")
    
    if any(trial[6] == 1 for trial in probe_trials):
        legend_elements.append(mlines.Line2D([],
                                             [],
                                             color="none",
                                             marker="o",
                                             markerfacecolor="blue",
                                             markersize=10))
        legend_labels.append("Rotated Target Well")

    # Training Scale Legend
    if num_train > 0:
        # Create a line for training trials
        train_all_colors = colors[0:num_train]
        train_handle = tuple(mlines.Line2D([], [], color=c, linestyle="-", lw=2) 
                             for c in train_all_colors)
        
        legend_elements.append(train_handle)
        legend_labels.append("Training Trials")

    # Probe Scale Legend (All Trials)
    if num_probe > 0:
        # Create a line for probe trial
        probe_all_colors = colors[num_train : num_train + num_probe]
        
        # Standard Probe
        probe_handle = tuple(mlines.Line2D([], [], color=c, linestyle="-", lw=2) 
                             for c in probe_all_colors)
        legend_elements.append(probe_handle)
        legend_labels.append("Probe Trials")
        
        # Rotated Probe
        if any(trial[6] == 1 for trial in probe_trials):
            # dashes=[2, 1] means 2 points on, 1 point off (tight dash pattern)/doesn't seem to work?
            rotated_handle = tuple(mlines.Line2D([], [], color=c, dashes=[2, 1], lw=2) 
                                   for c in probe_all_colors)
            legend_elements.append(rotated_handle)
            legend_labels.append("Rotated Probe Trials (dashed)")

    # 5. FINAL LEGEND CALL
    plt.legend(handles=legend_elements,
               labels=legend_labels,
               loc="center left",
               bbox_to_anchor=(1, 0.5),
               fontsize=10,
               handlelength=6.0,# IMPORTANT: increase length so dashed line shows up
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0)})

    plt.xlabel("X Position (cm)")
    plt.ylabel("Y Position (cm)")
    plt.title("Training and Probe Trials")
    plt.xlim(-40, 40)
    plt.ylim(-40, 40)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()

    return fig, colors

def draw_static_environment(env_type="Circle", ax=None):
    """
    Draws the wells and boundaries of the environment.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6)) # Uses standard_figsize if defined
    
    # 1. Plot static well locations
    # (Assuming well_locations_cm is defined globally as in your snippet)
    well_x = [x for x, y in well_locations_cm]
    well_y = [y for x, y in well_locations_cm]
    ax.scatter(well_x, well_y, color="gray", label="Wells", s=50, zorder=2)

    # 2. Draw boundary
    if env_type == "Circle":
        circle = plt.Circle((0, 0), circle_radius_cm, color="black", 
                            fill=False, linewidth=1.5)
        ax.add_patch(circle)
    elif env_type == "Square":
        pad = 7
        side = square_side_cm + pad
        square = plt.Rectangle((-side/2, -side/2), side, side,
                               color="black", fill=False, linewidth=1.5)
        ax.add_patch(square)
    
    # 3. Formatting
    ax.set_xlabel("X Position (cm)")
    ax.set_ylabel("Y Position (cm)")
    ax.set_xlim(-45, 45)
    ax.set_ylim(-45, 45)
    ax.set_aspect("equal")
    return ax

import matplotlib.pyplot as plt

def plot_avg_efficiency_across_ages(trials):
    """
    Plots average trajectory efficiency, skipping ages with no data.
    """
    # 1. Group data (No collections module)
    data_map = {}
    all_ages_in_data = set()
    
    for trial in trials:
        subject, age, rewarded_well, df = trial[0], trial[5], trial[7], trial[14]
        if df is None or df.empty:
            continue
            
        efficiency = st.calculate_efficiency(df, rewarded_well)
        if efficiency > 120:
          continue
        all_ages_in_data.add(age)
        
        if subject not in data_map:
            data_map[subject] = {}
        if age not in data_map[subject]:
            data_map[subject][age] = []
        data_map[subject][age].append(efficiency)

    # 2. Create a sorted list of ONLY the ages that actually have data
    sorted_unique_ages = sorted(list(all_ages_in_data))
    
    # Create a mapping so we know which index each age belongs to
    # Age 18 -> Index 0, Age 19 -> Index 1 ... Age 124 -> Index 17
    age_to_x_index = {age: i for i, age in enumerate(sorted_unique_ages)}

    # 3. Setup the plot
    plt.figure(figsize=(12, 6))
    
    for subject in sorted(data_map.keys()):
        subject_data = data_map[subject]
        
        # We only plot ages that exist for THIS subject
        subj_ages = sorted(subject_data.keys())
        
        # The X-coordinates are the INDICES from our sorted list, not the age itself
        x_coords = [age_to_x_index[age] for age in subj_ages]
        
        # Calculate means
        y_values = [sum(subject_data[age]) / len(subject_data[age]) for age in subj_ages]
        
        plt.plot(x_coords, y_values, marker="o", label=subject)

    # 4. Formatting to fix the labels and the gap
    plt.title("Avg Efficiency per Subject Across Ages (Training)")
    plt.xlabel("Age (days)")
    plt.ylabel("Efficiency (Ideal / Real Path, cm)")

    # This is the magic part: 
    # Set the ticks to the indices (0, 1, 2...) 
    # but label them with the actual age values (18, 19, 124...)
    plt.xticks(range(len(sorted_unique_ages)), sorted_unique_ages)
    
    # Note: I removed plt.ylim(0, 1.1) because your latest data 
    # has a massive outlier near 140. 
    # If you want to see that point, keep this commented out:
    # plt.ylim(0, 1.1)

    plt.legend(title="Subject", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    
def test_plot_training_and_probes(trials_data,
                             probe_trials,
                             sex_map=None, 
                             filter_sex=None,
                             colors=None,
                             separate_color_scale=True):
    """
    Plots all training and probe trial trajectories on the same figure.

    Parameters
    ----------
    trials_data : List
        List of training trial tuples (structured from load_trials()).
    probe_trials : List
        List of probe trial tuples (structured from load_trials()).
    Sex : filter by sex....
    colors : List, optional
        List of RGB colors used to plot each trial. The default is None.
    Seperate color scale : bool, optional
        True means new version - two color scales 
    Returns
    -------
    fig : Figure
        Matplotlib Figure object containing the produced figure.
    colors : List
        List of RGB colors used to plot each trial.

    """
    # added new filtering logic 
    if sex_map is not None and filter_sex is not None:
        trials_data = [t for t in trials_data if sex_map.get(t[0]) == filter_sex]
        probe_trials = [t for t in trials_data if sex_map.get(t[0]) == filter_sex]
    
    num_train = len(trials_data)
    num_probe = len(probe_trials)
    num_trials =  num_train + num_probe
    
    if num_trials == 0:
        raise ValueError('No trials selected for plotting.')

    if colors is None:
        if separate_color_scale:
            train_map = plt.cm.YlGn  
            probe_map = plt.cm.RdPu   
            
            # Create two lists and join them
            train_colors = [train_map(x) for x in np.linspace(0.4, 0.9, num_train)] if num_train > 0 else []
            probe_colors = [probe_map(x) for x in np.linspace(0.4, 0.9, num_probe)] if num_probe > 0 else []
            colors = train_colors + probe_colors
            
        else:
            combined_map = plt.cm.turbo
            colors = [combined_map(x) for x in np.linspace(0, 1, num_trials)]

    fig = plt.figure(figsize=standard_figsize)

    # Plot static well locations
    well_x = [x for x, y in well_locations_cm]
    well_y = [y for x, y in well_locations_cm]
    plt.scatter(well_x, well_y, color="gray", label="Wells", s=50)

    # Determine environment type from the first trial available
    _, _, _, env, *_ = trials_data[0]

    # Draw training trials
    for i, trial in enumerate(trials_data):
        (_, _, _, _, _, _, rotation, rewarded_well, _, trial_number, _, _, _, _, df, baited) = trial
        reward_x, reward_y = well_locations_cm[rewarded_well - 1]
        plt.scatter([reward_x],
                    [reward_y],   
                    color="red",
                    s=150,
                    label="Rewarded Well" if i == 0 else "")  
        plt.plot(df["x"],
                 df["y"],
                 color=colors[i],
                 alpha=0.7,
                 label=f"Training {trial_number}")

        if not df.empty:
            start_x, start_y = df.iloc[0]
            plt.scatter([start_x],
                        [start_y],
                        color=colors[i],
                        edgecolors="black",
                        linewidths=1.5,
                        marker="*",
                        s=200,
                        label="Start Position" if i == 0 else "")

    # Draw probe trials
    for j, trial in enumerate(probe_trials):
        (_, _, _, _, _, _, rotation, rewarded_well, _, trial_number, _, _, _, _, df, baited) = trial
        probe_index = len(trials_data) + j
        linestyle = "--" if rotation == 1 else "-"
        linewidth = 1.5 if rotation == 0 else 1.5
        label = "Probe Trial Rotation" if rotation == 1 else "Probe Trial"
        plt.plot(df["x"],
                 df["y"],
                 linestyle=linestyle,
                 linewidth=linewidth,
                 color=colors[probe_index],
                 label=label)

        if not df.empty:
            start_x, start_y = df.iloc[0]
            plt.scatter([start_x],
                        [start_y],
                        color=colors[probe_index],
                        edgecolors="black",
                        linewidths=1.5,
                        marker="*", s=200)

        if rotation == 1:
            reward_x, reward_y = well_locations_cm[rewarded_well - 1]
            plt.scatter([reward_x], [reward_y], color="blue", s=150, label="Rewarded Well Rotation")

    # Draw boundary of the environment
    ax = plt.gca()
    if env == "Circle":
        circle = plt.Circle((0, 0),
                            circle_radius_cm,
                            color="black",
                            fill=False,
                            linewidth=1.5)
        ax.add_patch(circle)

    elif env == "Square":
        pad = 7
        square_side = square_side_cm + pad
        half_side = square_side / 2
        square = plt.Rectangle((-half_side, -half_side),
                               square_side,
                               square_side,
                               color="black",
                               fill=False,
                               linewidth=1.5)
        ax.add_patch(square)

    else:
        raise ValueError('Environment type not correct!')

    # Build a clean and consistent legend
    # Define Static Markers/legend entries that never change 
    legend_elements = [
    
        # Use Line2D with linestyle="None" for scatter-like markers in legends
        mlines.Line2D([], [],
                      color="none",
                      marker="o", 
                      markerfacecolor="gray", 
                      markersize=10, 
                      label="Wells"),
        mlines.Line2D([], [],
                      color="none",
                      marker="*",
                      markeredgecolor="black", 
                      markerfacecolor="gray",
                      markersize=12,
                      label="Start Position"),
        mlines.Line2D([], [],
                      color="none",
                      marker="o",
                      markerfacecolor="red", 
                      markersize=10,
                      label="Target Well"),
        ]

# Add Grouped Trial Types (instead of every individual trial)
    # Training Group
    if len(trials_data) > 0:
        legend_elements.append(
            mlines.Line2D([], [],
                          color=colors[0],
                          linestyle="-",
                          lw=2,
                          label="Training Trials"))

    # Probe Group (Standard)
    if len(probe_trials) > 0:
        legend_elements.append(
            mlines.Line2D([], [],
                          color=colors[len(trials_data)],
                          linestyle="-",
                          lw=2,
                          label="Probe Trials"))
        
        # Check if any probe trials were rotated to add the dashed line to legend
        if any(trial[6] == 1 for trial in probe_trials):
            legend_elements.append(
                mlines.Line2D([], [],
                              color=colors[-1],
                              linestyle="--",
                              lw=2,
                              label="Probe (Rotated)"))
            legend_elements.append(
                mlines.Line2D([], [],
                              color="none",
                              marker="o",
                              markerfacecolor="blue",
                              markersize=10,
                              label="Rotated Target"))

    # Final plot details
    plt.legend(handles=legend_elements,
               loc="center left",
               bbox_to_anchor=(1, 0.5),
               fontsize=12)
    plt.xlabel("X Position (cm)")
    plt.ylabel("Y Position (cm)")
    plt.title("Training and Probe Trials")
    plt.xlim(-45, 45)
    plt.ylim(-45, 45)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()

    #added title 
    title_str =  "Training and Probe Trials"
    if filter_sex:
        title_str += f"({'Males' if filter_sex == 'M' else 'Females'})"
    plt.title(title_str)

    return fig, colors

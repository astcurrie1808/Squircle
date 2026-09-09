# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 15:27:46 2025

@author: JoseTeixeira
"""


#%% To-Do's

"""
    Change the whole 15-element tuple logic into dataclass (eg Dicts)
    
"""

#%% Imports

import os
import math
from datetime import datetime, timedelta
from io import StringIO
import pandas as pd
import numpy as np
import json
import random

from squircle.design import well_locations, well_locations_cm, circle_radius, square_boundary, cm_per_pixel



#%% Functions

# TODO - implement rotation and decide what/how to return
def alignment(center_coords, unit = 'px', top_coords=well_locations[0], kind='Translation'):
    """
    Takes the coordinates for the center well and returns a translation vector.
    At the moment, rotation is not implemented.

    Parameters
    ----------
    center_coords : Tuple
        Tuple with the (x,y) coordinates of the center well (well #17).
    unit : String, optional
        In which unit to calculate the alignment. The default is 'px' (pixels)
    top_coords : Tuple, optional
        Tuple with the (x,y) coordinates of the top well (well #1).
        The default is well_locations[0].
    kind : String, optional
        What alignment operation to perform.
        Supported operations are:
            'Translation'
            'Rotation'
            'Both'
        The default is 'Translation'.

    Returns
    -------
    Tuple
        Alignment vector.

    """

    operations = ['Translation', 'Rotation', 'Both']

    match unit:
        case 'px':
            center_ref = well_locations[-1]
            top_ref = well_locations[0]
            
        case 'cm':
            center_ref = well_locations_cm[-1]
            top_ref = well_locations_cm[0]

        case _:
            info_string = '\n'.join((
                '\'unit\' is not supported!',
                'Supported units are:',
                'px',
                'cm'
                ))

            raise ValueError(info_string)

    match kind:
        case 'Translation':
            translationXY = tuple(x - y for x, y in zip(center_ref, center_coords))

        case 'Rotation':
            # TODO - fix the rotation! Idea: compare docs for atan2 across math and matlab, I have a feeling one is in radians and the other is not
            ref_vector = tuple(x - y for x, y in zip(top_ref, center_ref))
            curr_vector = tuple(x - y for x, y in zip(top_coords, center_ref))
            angle = math.atan2(ref_vector[1],
                               ref_vector[0]) - math.atan2(curr_vector[1],
                                                           curr_vector[0])
            rotation_matrix = np.array([[math.cos(angle), -math.sin(angle)],
                                        [math.sin(angle), math.cos(angle)]])

        case 'Both':
            pass

        # any other case
        case _:

            info_string = '\n'.join((
                '\'kind\' is not one of the supported operations!',
                'Supported operations are:',
                *operations
                ))

            raise ValueError(info_string)

    return translationXY    # temporary while we are only concerned about translation and not rotation
    
def create_json(litter_folder, savedir=None, verbose=False):
    """
    Creates a structured JSON file from raw tracking data and a summary .csv for the whole litter.
    This function filters for litter number, reads individual subject session data,
    extracts relevant metadata, reads corresponding trial tracking data (in pixels),
    converts positions to centimeters, and saves all sessions in structured JSON format.

    Parameters
    ----------
    litter_folder : String
        Directory containing subject session subfolders and tracking files.
    savedir : String, optional
        Directory where to save output file. The default is None (current working directory)
    verbose : Bool, optional
        Whether to log progress into console. The default is False.
    Returns
    -------
    outpath : String
        Path to the created json file.
    
    """

    summary_csv = os.path.join(litter_folder, 'Summary.csv')
    # litter_n = int(os.path.split(litter_folder)[1][-1]) 
    litter_n = int(os.path.split(litter_folder)[1].replace("Litter", ""))

    # Load and preprocess summary file
    summary_df = pd.read_csv(summary_csv, sep=';', encoding='utf-8-sig')
    summary_df["Litter"] = summary_df["Litter"].astype(str).str.strip()  # Ensure Litter is clean string
    summary_df["Codename"] = summary_df["Codename"].astype(str).str.strip()
    summary_df["Sex"] = summary_df["Sex"].astype(str).str.strip()
    summary_df = summary_df[summary_df["Litter"] == str(litter_n)]  # Filter only Litter N animals
    
    # Get list of subjects and a dictionary mapping subject -> DOB
    subjects = summary_df["Codename"].tolist()
    subject_dobs = dict(zip(summary_df["Codename"], summary_df["DOB"]))
    subject_sex = dict(zip(summary_df["Codename"], summary_df["Sex"]))
    
    data = []
    
    for subject in subjects:
        subject = subject.strip()
        # Reformat DOB and convert to datetime for age calculation
        dob_str = str(subject_dobs[subject]).strip().replace("/", "_").replace("-", "_")
        dob = datetime.strptime(dob_str, "%d_%m_%Y")
        # Convert ID into Boolean: True if Male, False if Female
        sex_str = str(subject_sex[subject]).strip().upper() == 'M'
        
        subject_data = {
            "Name": subject,
            "DOB": dob_str,
            "Sex": sex_str,
            "Sessions": []
        }
        
        subject_csv = os.path.join(litter_folder, f"{subject}.csv")
        if not os.path.exists(subject_csv):
            raise NameError(f"Subject file not found: {subject_csv}")
        elif verbose:
            print(f"Processing subject file: {subject_csv}")

        df = pd.read_csv(subject_csv, sep=';', decimal=',', encoding='utf-8-sig')
        df["Date"] = df["Date"].astype(str).str.strip().str.replace("/", "_").str.replace("-", "_")  # Format dates consistently
        

        # Group trials by unique session (Type, Env, Date, Rotation)
        sessions = df.groupby(["Session", "Environment", "Date", "Rotation"])

        for (session_type, env, date_str, rotation), trials in sessions:
            # Convert to folder naming format (strip leading zeros)
            day, month, year = date_str.split("_")
            folder_date = f"{int(day)}_{int(month)}_{year}"

            # Convert session date string to datetime object
            session_date = datetime(int(year), int(month), int(day))

            # Compute age in days for this session
            age_in_days = (session_date - dob).days

            # Collect all metadata for this session
            session_data = {
                "Type": session_type,
                "Environment": env,
                "Date": date_str,
                "Age": int(age_in_days),
                "Rotation": int(rotation),
                "Rewarded well": int(trials["Rewarded_well"].iloc[0]),
                "Center x": trials["CenterX"].iloc[0],
                "Center y": trials["CenterY"].iloc[0],
                "Cues": trials["Cues"].iloc[0].split(","),
                "Trials": []
            }

            # Build path to the expected tracking data folder
            env_folder = env if int(rotation) == 0 else f"{env}_Rotation"
            session_path = os.path.join(litter_folder, session_type)
            subject_path = os.path.join(session_path, subject)
            env_path = os.path.join(subject_path, env_folder)
            date_path = os.path.join(env_path, folder_date)

            # Skip session if path is invalid
            if not os.path.exists(session_path):
                print(f"Session folder not found: {session_path}")
                continue
            if not os.path.exists(subject_path):
                print(f"Subject folder not found: {subject_path}")
                continue
            if not os.path.exists(env_path):
                print(f"Environment folder not found: {env_path}")
                continue
            if not os.path.exists(date_path):
                print(f"Date folder not found: {date_path}")
                continue
            
            if verbose:
                print(f"Looking inside: {date_path}")   # Troubleshooting

            # Loop through each trial in the session
            for _, row in trials.iterrows():
                trial_number = row["Trial"]
               # tracking_file = f"NewTracking_trial{trial_number}.csv"
                tracking_file =  f"NewTracking_trial{int(row['Trial'])}.csv"
                file_path = os.path.join(date_path, tracking_file)
               

                if os.path.exists(file_path):
                    df_positions = pd.read_csv(file_path, header=None, names=["x", "y"])
                    df_positions["x"]
                    df_positions["y"]
                    position_data = df_positions.to_json()

                else:
                    print(f"Tracking file not found: {file_path}")
                    position_data = "{}"

                # Collect trial-specific metadata
                trial_data = {
                    "Number": trial_number,
                    "Start frame": row["Start_frame"],
                    "End frame": row["End_frame"],
                    "Frame offset x": row["FrameOffsetX"],
                    "Frame offset y": row["FrameOffsetY"],
                    "Baited": int(row["Baited"]),
                    "Completed": str(row["Completed"]).strip() if "Completed" in row else "Completed",
                    "Position": position_data
                }

                session_data["Trials"].append(trial_data)

            subject_data["Sessions"].append(session_data)

        data.append(subject_data)

    # Write the final structured dataset to JSON
    if savedir is None:
        savedir = os.getcwd()
    
    filepath = 'litter_' + str(litter_n) + '_pixel.json'
    
    outpath = os.path.join(savedir, filepath)

    with open(outpath, "w") as json_file:
        json.dump(data,
                  json_file,
                  indent=4,
                  default=lambda o: int(o) if isinstance(o, (np.integer)) else o)

    print("JSON file for litter %d created successfully." % litter_n)
    
    return outpath

def merge_litters(json_files):
    """
    Merge multiple litter JSON files into one combined JSON.

    Parameters
    ----------
    json_files : String | List
        JSON files to combine as either a list of file paths, or a directory.
        In the latter case, it combines all (preprocessed) files in the folder.

    Returns
    -------
    outpath : String
        Path to the generated JSON file.

    """

    combined_data = []
    filepath = 'combined_litters_'

    if type(json_files) == list:

        filelist = json_files
            
    elif type(json_files) == str:
        if os.path.isdir(json_files):
            filelist = [os.path.join(json_files, f) \
                        for f in os.listdir(json_files) \
                        if os.path.isfile(os.path.join(json_files, f)) \
                        if f.endswith('processed_cm.json')]
        
        else:
            raise ValueError('Input not accepted! Must be directory or list of files')
    
    else:
        raise ValueError('Input not accepted! Must be directory or list of files')
    
    for json_file in filelist:
        with open(json_file, "r") as f:
            litter_data = json.load(f)

        if not isinstance(litter_data, list):
            raise ValueError(f"{json_file} does not contain a list as expected")

        combined_data.extend(litter_data)
        
        # Grab litter number for naming output file
        ht = os.path.split(json_file)
        input_file = ht[1]
        fileparts = input_file.split('_')
        filepath += str(fileparts[1]) + '_'
     
    filepath +='.json'
    
    if len(ht[0]) == 0:
        savedir = os.getcwd()
    
    else:
        savedir = ht[0]
    
    savedir = 'C:\\Users\\astcu\\OneDrive\\Documents\\2025-2026\\Neurobiologie\\Internship'
    outpath = os.path.join(savedir, filepath)

    with open(outpath, "w") as f:
        json.dump(combined_data, f, indent=2)

    print(f"Combined JSON saved to {outpath}")

    return outpath

def process_trial(df, session_center, environment, frame_offset_x=0, frame_offset_y=0):
    """
    Helper function for preprocess_json_positions().
    Processes a single trial's tracking data by aligning position data to
    common reference frame and correcting for tracking errors.
    
    Parameters
    ----------
    df : DataFrame
        Raw tracking data (in pixels) with 'x' and 'y' columns.
    session_center : Tuple
        Tuple of (center_x, center_y) for alignment reference.
    environment : String
        Environment type; either 'Circle' or 'Square'.
    frame_offset_x : Float, optional
        Horizontal offset to apply. The default is 0.
    frame_offset_y : Float, optional
        Vertical offset to apply. The default is 0.

    Returns
    -------
    df : DataFrame
        Corrected tracking data.

    """
    
    if session_center is None:
        raise ValueError("Session center coordinates are missing.")

    # Apply frame offsets to correct potential recording misalignment
    df['x'] += frame_offset_x
    df['y'] += frame_offset_y

    # Align session center to (0, 0) using translation
    translationXY = alignment(session_center)
    df += translationXY
    df.columns = ['x', 'y']

    # Invert y-axis 
    df['y'] *= -1

    # Set filtering threshold values
    distance_threshold = 15  # max jump allowed between frames (in pixels)
    nan_index = []           # collect indices to remove
    last_valid_index = None  # track last valid position

    for i in range(len(df)):
        x1, y1 = df.iloc[i]

        # Skip NaNs
        if pd.isna(x1) or pd.isna(y1):
            nan_index.append(i)
            continue

        # Reject positions outside arena boundary
        if environment == 'Circle':
            distance_from_center = np.sqrt(x1**2 + y1**2)
            if distance_from_center > circle_radius:
                nan_index.append(i)
                continue
            
        elif environment == 'Square':
            if abs(x1) > square_boundary or abs(y1) > square_boundary:
                nan_index.append(i)
                continue

        # Reject large jumps in trajectory
        if last_valid_index is not None:
            x2, y2 = df.iloc[last_valid_index]
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            if distance > distance_threshold:
                nan_index.append(i)
                continue

        last_valid_index = i

    # Replace invalid positions with NaN
    for idx in nan_index:
        df.at[idx, 'x'] = np.nan
        df.at[idx, 'y'] = np.nan

    # Interpolate missing data linearly
    df['x'] = df['x'].interpolate(method='linear', limit_direction='both')
    df['y'] = df['y'].interpolate(method='linear', limit_direction='both')

    return df

def preprocess_json_positions(input_path):
    """
    Loads a structured JSON dataset and corrects position values caused by
    tracking errors. Creates a new JSON file in the same folder.

    Parameters
    ----------
    input_path : String
        Path to the JSON file with raw (pixel) position data.

    Returns
    -------
    outpath : String
        Path to the created json file.

    """

    # Load the original structured JSON file
    with open(input_path, "r") as f:
        data = json.load(f)

    # Iterate over all subjects and sessions
    for subject in data:
        for session in subject["Sessions"]:
            session_center = (session["Center x"], session["Center y"])
            environment = session["Environment"]

            for trial in session["Trials"]:
                # Read position data into a DataFrame
                df = pd.read_json(StringIO(trial["Position"]))
                if df.empty:
                    continue  # Skip failed trials with no tracking data

                # Get frame offsets if available
                fx = trial.get("Frame offset x", 0)
                fy = trial.get("Frame offset y", 0)

                # Run correction and alignment on the raw tracking data
                df_processed = process_trial(
                    df.copy(),
                    session_center=session_center,
                    environment=environment,
                    frame_offset_x=fx,
                    frame_offset_y=fy
                )

                # Save the cleaned DataFrame back as JSON string
                trial["Position"] = df_processed.to_json()

    # Write updated trial data back to output JSON
    
    ht = os.path.split(input_path)
    out_folder = ht[0]
    input_file = ht[1]
    fileparts = input_file.split('_')
    
    filepath = fileparts[0] + '_' + fileparts[1] + '_processed_' + fileparts[2]
    
    output_path = os.path.join(out_folder, filepath)
    with open(output_path, "w") as f:
        json.dump(data,
                  f,
                  indent=4,
                  default=lambda o: int(o) if isinstance(o, (np.integer)) else o)

    print(f"Processed JSON; saved to {output_path}")
    
    return output_path
    
def convert_positions_to_cm(input_path):
    """
    Converts all position coordinates in a JSON file from pixels to centimeters.

    Parameters
    ----------
    input_path : String
        Path to the JSON file with preprocessed (pixel) position data.

    Returns
    -------
    outpath : String
        Path to the created JSON file.

    """

    # Load the JSON dataset
    with open(input_path, "r") as f:
        data = json.load(f)

    # Iterate over each subject and session
    for subject in data:
        for session in subject["Sessions"]:
            for trial in session["Trials"]:
                try:
                    # Read position data into DataFrame
                    df = pd.read_json(StringIO(trial["Position"]))
                    if df.empty:
                        continue  # Skip failed trials with no data

                    # Scale coordinates from pixels to centimeters
                    df["x"] *= cm_per_pixel
                    df["y"] *= cm_per_pixel

                    # Save updated coordinates back to JSON string
                    trial["Position"] = df.to_json()
                except Exception as e:
                    # Log and skip any trial with issues
                    print(f"Skipping one trial due to error: {e}")

    # Save the modified dataset to output path
    
    ht = os.path.split(input_path)
    out_folder = ht[0]
    input_file = ht[1]
    
    output_path = os.path.join(out_folder, input_file.replace('pixel', 'cm'))
    
    with open(output_path, "w") as f:
        json.dump(data,
                  f,
                  indent=4,
                  default=lambda o: int(o) if isinstance(o, (np.integer)) else o)

    print(f"Position data converted to cm and saved to {output_path}")
    
    return output_path

def load_population_trials(data,
                           name=None,
                           kind=None,
                           environment=None,
                           date=None,
                           number=None,
                           rotation=None,
                           baited=None):
    """
    Loads trial-level data across the entire population based on various filters.

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    name : String, optional
        Subject to extract. The default is None.
    kind : String, optional
        Session kind ('Training' or 'Probe'). The default is None.
    environment : String, optional
        Environment type ('Circle' or 'Square'). The default is None.
    date : String, optional
        Session date in the format 'dd_mm_yyyy'. The default is None.
    number : int | list[int] | set[int], optional
        Specific trial number(s) to include. The default is None.
    rotation : Int, optional
        Rotation condition (0 for no rotation, 1 otherwise). The default is None.
    baited : Int, optional
        Baited condition (0 for unbaited, 1 for baited). The default is None. 

    Returns
    -------
    selected_trials : List
        List contining the trials that match the specified criteria.

    """

    selected_trials = []

    for subject_data in data:
        # Filter by subject name if specified
        if name is not None and subject_data["Name"] != name:
            continue
        subject_sex = subject_data.get("Sex", "Unknown")

        for session in subject_data["Sessions"]:
            # Apply session-level filters
            if kind is not None and session["Type"] != kind:
                continue
            if environment is not None and session["Environment"] != environment:
                continue
            if date is not None and session["Date"] != date:
                continue
            if rotation is not None and session["Rotation"] != rotation:
                continue

            session_center = (session["Center x"], session["Center y"])

            for trial in session["Trials"]:
                # Filter by trial number
                if number is not None:
                    if isinstance(number, (list, tuple, set)):
                        if trial["Number"] not in number:
                            continue
                    elif trial["Number"] != number:
                        continue
                if baited is not None and int(trial["Baited"]) != int(baited):
                    continue

                df = pd.read_json(StringIO(trial["Position"]))

                # Skip trials with no tracking data
                if df.empty:
                    print(f"Skipping trial {trial['Number']} for {subject_data['Name']} on {session['Date']} (empty positions).")
                    continue

                # Package all relevant trial data
                trial_info = (
                    subject_data["Name"],
                    subject_data["DOB"],
                    subject_sex,
                    session["Type"],
                    session["Environment"],
                    session["Date"],
                    session["Age"],
                    session["Rotation"],
                    session["Rewarded well"],
                    session_center,
                    trial["Number"],
                    trial["Start frame"],
                    trial["End frame"],
                    trial["Frame offset x"],
                    trial["Frame offset y"],
                    df,
                    trial["Baited"],
                    trial.get("Completed", "Completed")
                )

                selected_trials.append(trial_info)

    return selected_trials

def load_trials(filepath,
                name=None,
                kind=None,
                environment=None,
                date=None,
                number=None,
                rotation=None,
                baited=None,
                include_probes=False,
                same_day=False):
    """
    Loads trial data from a JSON dataset according to the specified arguments.
    Each trial is in a 17-tuple format:
        name - Subject's name\n
        dob - Subject's date of birth\n
        sex - Subject's sex\n
        kind - Trial type (Training or Probe)\n
        env - Environment (Circle or Square)\n
        date - Session date\n
        age - Subject's age on the day of the Session\n
        rotation - Whether this is a rotation Session\n
        rewarded_well - Number of the rewarded well\n
        session_center - Coordinates (x,y) of the center of the environment\n
        trial_number - Trial number\n
        start_frame - First frame in the trial's video\n
        end_frame - Last frame in the trial's video\n
        fx - x coordinate of the trial video's cropping\n
        fy - y coordinate of the trial video's cropping\n
        df - Positions of the animal in the trial\n
        
    Parameters
    ----------
    filepath : String
        Path to the preprocessed JSON file.
    name : String, optional
        Subject to extract. The default is None.
    kind : String, optional
        Session kind ('Training' or 'Probe'). The default is None.
    environment : String, optional
        Environment type ('Circle' or 'Square'). The default is None.
    date : String, optional
        Session date in the format 'dd_mm_yyyy'. The default is None.
    number : int | list[int] | set[int], optional
        Specific trial number(s) to include. The default is None.
    rotation : Int, optional
        Rotation condition (0 for no rotation, 1 otherwise). The default is None.
    include_probes : Bool, optional
        Whether to include probe trials from the next session. The default is False.
    same_day : Bool, optional
        Whether to take grab probe trials with the same date as training trials.
        The default is False.
    baited : Bool, optional. The default is None

    Returns
    -------
    selected_trials : List
        List contining the trials that match the specified criteria.
    probe_trials : List
        If include_probes=True, returns a list containing those;
        otherwise returns an empty list.
        Note: if kind == 'Probe', then the probe trials will be assigned to
        the selected_trials output variable.

    """

    with open(filepath, 'r') as json_file:
        litter_data = json.load(json_file)
        
    selected_trials = load_population_trials(data=litter_data,
                                             name=name,
                                             kind=kind,
                                             environment=environment,
                                             date=date,
                                             number=number,
                                             rotation=rotation,
                                             baited=baited)
    
    probe_trials = []
    if include_probes:
       if same_day:
            probe_date=date
            
       else:
            # Parse the date string for later comparisons
           date_obj = datetime.strptime(date, "%d_%m_%Y") if date else None
           probe_date = (date_obj + timedelta(days=1)).strftime("%d_%m_%Y") if date_obj else None
        
       probe_trials = load_population_trials(data=litter_data,
                                          name=name,
                                          kind='Probe',
                                          date=probe_date,
                                          number=number,
                                          rotation=rotation,
                                          baited=baited)

    return selected_trials, probe_trials

def calculate_efficiency(df, reward_location):
    """
    Computes the path efficiency ratio in a given trial.

    Parameters
    ----------
    df : DataFrame
        Pandas DataFrame containing the x,y positions of the subject in the trial.
    reward_location : Int
        Number of the rewarded well (1-based).

    Returns
    -------
    efficiency : Float
        The computed path efficiency ratio.

    """

    start_x, start_y = df.iloc[0]["x"], df.iloc[0]["y"]
    reward_x, reward_y = well_locations_cm[reward_location - 1]
    ideal_distance = np.sqrt((reward_x - start_x)**2 + (reward_y - start_y)**2)

    x = df["x"].values
    y = df["y"].values
    real_distance = np.nansum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))

    # Handle edge cases
    if real_distance < ideal_distance:
        real_distance = ideal_distance

    try:
        efficiency = real_distance/ideal_distance
        
#  Original (ideal / real): You are asking: "What percentage of the rat's movement was actually productive?"
#   Score 0.5: "Only 50% of the distance the rat walked helped it get to the goal."
# Current as of 8-5-26: (real / ideal): You are asking: "How many times longer than necessary was the path taken?"
#   Score 2.0: "The rat walked twice as far as it actually needed to."
#   Score 5.0: "The rat walked five times the distance of a straight line."

    except ZeroDivisionError:
        Warning('Trying to divide by zero! Check real_distance for this trial')

    return efficiency

#%% Generate 30 * Baited/unbaited + coordinates 
# List of all possible entry points (27 in total) with abbreviated names
entry_points = [
    "N-o", "N-m", "N-i",
    "W-o", "W-m", "W-i",
    "S-o", "S-m", "S-i",
    "E-o", "E-m", "E-i",
    "S-o", "S-m", "S-i",
    "NW-o", "NW-m", "NW-i",
    "NE-o", "NE-m", "NE-i",
    "SW-o", "SW-m", "SW-i",
    "SE-o", "SE-m", "SE-i",
]
# Initialize run lengths and last trials for each of the 2 columns
run_lengths = [0, 0, 0]
last_trials = [None, None, None]
# Generate 30 rows with 2 columns each
for _ in range(30):
    row = []
    for col in range(2):
        # Generate BAITED/UNBAITED for this column
        if run_lengths[col] == 4:
            choice = "UNBAITED" if last_trials[col] == "BAITED" else "BAITED"
        else:
            choice = random.choice(["BAITED", "UNBAITED"])
        # Update run length and last trial for this column
        if choice == last_trials[col]:
            run_lengths[col] += 1
        else:
            run_lengths[col] = 1
            last_trials[col] = choice
        # Generate random coordinate for this column
        coord = random.choice(entry_points)
        row.append(f"{choice},{coord}")
    # Print the row with all 2 columns
    print(",".join(row))
    

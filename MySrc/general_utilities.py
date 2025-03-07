from pynwb import NWBHDF5IO
import numpy as np
from scipy.ndimage import gaussian_filter1d
import os
import datetime

# Prototype function
def load_nwb(file_path):
    """Load an NWB file and return the NWBFile object."""
    io = NWBHDF5IO(file_path, mode='r')
    nwbfile = io.read()
    return nwbfile, io

def get_units_tables(nwbfile):
    """
    Extracts all unit tables from an NWB file stored under nwbfile.processing['units'].

    Parameters:
    - nwbfile: NWBFile object (loaded from NWBHDF5IO)

    Returns:
    - units_tables (dict): A dictionary where keys are table names and values are unit tables.
    - table_names (list): A list of table names for reference.
    """
    units_tables = {}
    table_names = []

    # Ensure that 'units' processing module exists
    if 'units' not in nwbfile.processing:
        print("No 'units' processing module found in this NWB file.")
        return units_tables, table_names  # Return empty structures

    # Access the processing module correctly
    units_module = nwbfile.processing['units']

    # Iterate through DataInterfaces and store both table names and tables
    for name, data_interface in units_module.data_interfaces.items():
        units_tables[name] = data_interface  # Store the table
        table_names.append(name)  # Store the name separately

    return units_tables, table_names

def get_trial_timing(nwbfile):
    """
    Extracts 'start_time' and 'stop_time' from the NWB trials table.

    Parameters:
    - nwbfile: NWBFile object (loaded from NWBHDF5IO)

    Returns:
    - dict: A dictionary containing trial event timings with keys:
      - 'start_time': Trial start time.
      - 'stim1_ON_time': First stimulus onset time.
      - 'stim2_ON_time': Second stimulus onset time.
      - 'choiceTarget_ON_time': Time when choice target appears.
      - 'fixTarget_OFF_time': Time when fixation target disappears.
      - 'responseStart_time': Time when the subject initiates a response and gaze go out of fixation point.
      - 'fix_ChoiceTarget_time': Time when gaze fixes at Choice target.
      - 'rewardStart_time': Time when the subject receives a reward.
      - 'choiceTarget_OFF_time': Time when the choice target disappears.
      - 'trialEnd_time': Trial enter ending phase.
      - 'stop_time': Trial stop the trial ending time.
    """
    # Ensure the NWB file contains trial data
    if not hasattr(nwbfile, 'trials') or nwbfile.trials is None:
        print("No trial data found in this NWB file.")
        return None  # Return None if no trial data is available

    # Convert NWB trials table to a Pandas DataFrame
    trials_df = nwbfile.trials.to_dataframe()
    # Extract 'start_time' and 'stop_time' columns
    start_time = nwbfile.trials['start_time'].data[:]
    
    stim1_ON_time = nwbfile.trials['stim1_ON_time'].data[:]
    stim1_OFF_time = nwbfile.trials['stim1_OFF_time'].data[:]
    stim2_ON_time = nwbfile.trials['stim2_ON_time'].data[:]
    stim2_OFF_time = nwbfile.trials['stim2_OFF_time'].data[:]

    choiceTarget_ON_time = nwbfile.trials['choiceTarget_ON_time'].data[:]
    fixTarget_OFF_time = nwbfile.trials['fixTarget_OFF_time'].data[:]
    responseStart_time = nwbfile.trials['responseStart_time'].data[:]
    fix_ChoiceTarget_time = nwbfile.trials['fix_ChoiceTarget_time'].data[:]
    rewardStart_time = nwbfile.trials['rewardStart_time'].data[:]
    choiceTarget_OFF_time = nwbfile.trials['choiceTarget_OFF_time'].data[:]
    trialEnd_time = nwbfile.trials['trialEnd_time'].data[:]
    
    stop_time = nwbfile.trials['stop_time'].data[:]

    if len(start_time) != len(stop_time):
        raise ValueError(f"❌ Mismatch in count: {len(start_time)} start_times vs {len(stop_time)} stop_times.")

    # **Sanity Check 2 & 3: Ensure each start time comes before its corresponding stop time and trials are sequential**
    for i in range(len(start_time)):
        if start_time[i] >= stop_time[i]:  # Start should be strictly before stop
            raise ValueError(f"❌ Timing issue at index {i}: start_time ({start_time[i]}) >= stop_time ({stop_time[i]}).")

        if i > 0 and start_time[i] <= stop_time[i - 1]:  # Ensure start → stop → start → stop sequence
            raise ValueError(f"❌ Overlapping trials at index {i}: start_time ({start_time[i]}) is before previous stop_time ({stop_time[i-1]}).")

    return {'start_time':start_time,'stim1_ON_time':stim1_ON_time, 'stim1_OFF_time':stim1_OFF_time,
            'stim2_ON_time':stim2_ON_time, 'stim2_OFF_time':stim2_OFF_time,
            'choiceTarget_ON_time':choiceTarget_ON_time,'fixTarget_OFF_time':fixTarget_OFF_time,'responseStart_time':responseStart_time,
            'fix_ChoiceTarget_time':fix_ChoiceTarget_time,'rewardStart_time':rewardStart_time,'choiceTarget_OFF_time':choiceTarget_OFF_time,
            'trialEnd_time':trialEnd_time,'stop_time':stop_time}

def compute_average_firing_rate(unit_spike_times, align_times, window, bin_size = 0.05, smooth = True, smooth_sigma  = 2):
    time_bins = np.arange(window[0], window[1], bin_size)
    trial_spike_counts = np.zeros((len(align_times), len(time_bins) - 1))
    
    for trial_idx, align_time in enumerate(align_times):
        aligned_spikes = unit_spike_times - align_time  # Align spikes to event
        filtered_spikes = aligned_spikes[(aligned_spikes >= window[0]) & (aligned_spikes <= window[1])]
        hist, _ = np.histogram(filtered_spikes, bins=time_bins)  # Bin spike counts
        trial_spike_counts[trial_idx, :] = hist / bin_size  # Convert to rate (Hz)
    
    mean_firing_rate = np.mean(trial_spike_counts, axis=0)  # Average across trials
    if smooth:
        smoothed_firing_rate = gaussian_filter1d(mean_firing_rate, sigma=smooth_sigma)  # Apply Gaussian smoothing
    else:
        smoothed_firing_rate = mean_firing_rate
    return smoothed_firing_rate, time_bins

def check_quality(metric_name, metric_value, quality_thresholds = None):
    """
    Check if a given metric value falls within the acceptable threshold range.
    
    Args:
        metric_name (str): Name of the metric.
        metric_value (float): The value of the metric to check.
    
    Returns:
        bool: True if the value is within the good range or if the metric has no predefined threshold. 
              False if the value is outside the defined threshold range.
    """
    if quality_thresholds == None:
        quality_thresholds = {
            "presence_ratio": (0.9, 1.0),
            "snr": (3.5, float('inf')),  # SNR should be above 3.5
            "isi_violations_ratio": (0, 0.05),  # Low ISI violations indicate better quality
            "l_ratio": (0, 0.2),  # L-ratio should be small for good quality
            "nn_hit_rate": (0.8, 1.0),  # Example: Assume a good hit rate should be high
            "sd_ratio": (0.66, 1.5)  # Example: Within a reasonable range
        }
    if metric_name in quality_thresholds:
        min_val, max_val = quality_thresholds[metric_name]
        return min_val <= metric_value <= max_val
    elif metric_name == "quality":
        return metric_value == "good"
    return True  # If no threshold is defined, return True by default

def fetch_key_metrices(units_tables, table_names,electrode_id, unit_index, metrices_names = None):
    if metrices_names == None:
        metrices_names = ["quality","presence_ratio","snr","isi_violations_ratio","l_ratio","nn_hit_rate","sd_ratio"]
    fetched_metrices = dict()
    for met in metrices_names:
        fetched_metrices[met] = units_tables[table_names[electrode_id]][met][unit_index]
    
    return fetched_metrices, metrices_names
    

    import os
import datetime

def save_figure(fig, handle=".jpg", name=None, output_folder=None):
    """
    Saves the given matplotlib figure with a timestamped filename.

    Parameters:
        fig (matplotlib.figure.Figure): The figure object to save.
        handle (str): The file extension (e.g., ".jpg", ".png", ".pdf", ".svg"). Default is ".jpg".
        name (str or None): The base name for the file. If None, uses only the timestamp.
        output_folder (str or None): The directory to save the figure. 
                                     If None, saves in `example_plots/` in the current script's location.
    """
    # Ensure the handle is a valid extension
    valid_extensions = (".jpg", ".jpeg", ".png", ".pdf", ".svg",".eps")
    if not handle.lower().startswith(valid_extensions):
        raise ValueError(f"Invalid file extension '{handle}'. Use one of {valid_extensions}")

    # Generate a timestamp for uniqueness
    timestamp = datetime.datetime.now().strftime("%y%m%d%H%M%S")

    # Construct filename: [name]_YYYYMMDDHHMMSS.[extension] or just timestamp if no name given
    filename = f"{name}_{timestamp}{handle}" if name else f"{timestamp}{handle}"

    # Determine output directory
    if output_folder is None:
        # Get the current working directory (where the script/notebook is running)
        script_dir = os.getcwd()
        output_folder = os.path.join(script_dir, "example_plots")

    # Create `example_plots/` or target folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Determine full file path
    file_path = os.path.join(output_folder, filename)

    # Save the figure
    fig.savefig(file_path, dpi=300, bbox_inches="tight")
    
    print(f"✅ Figure saved at: {file_path}")

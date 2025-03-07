import numpy as np
import matplotlib.pyplot as plt
from MySrc.general_utilities import load_nwb,compute_average_firing_rate, get_units_tables,get_trial_timing, fetch_key_metrices, check_quality, save_figure # Import the function
import matplotlib.colors as mcolors
from pathlib import Path

def plot_neuron_general_metrices(
    filepath, electrode_id, unit_id,
    trial_window_marks = ["stim1_ON_time", "stim2_ON_time", "stim2_OFF_time", "choiceTarget_ON_time","responseStart_time"],
    trial_window_expanding=[-1, +2], smooth=False, resize = 1, silent_plot = False
):
    """
    Plots neuron activity including raster plots, firing rates, ISI distribution, 
    waveform plots, and quality metrics.

    Parameters:
        units_tables (dict): Dictionary containing spike sorting results.
        table_names (dict): Mapping of electrode IDs to table names.
        electrode_id (int): ID of the electrode.
        unit_id (int): ID of the neuron/unit.
        trials_timings (dict): Dictionary with 'start_time' and 'stop_time' arrays.
        start_window (tuple): Time window around trial start (default: (-1,6)).
        stop_window (tuple): Time window around trial stop (default: (-6,1)).
        smooth (bool): Whether to apply smoothing when computing firing rates (default: True).
        resize (float): resize the figure proportionally
    """
    
    # Fetch unit data from tables
    nwbfile, io = load_nwb(filepath)
    units_tables, table_names = get_units_tables(nwbfile)
    #print(table_names)
    trials_timings = get_trial_timing(nwbfile)

    unit_spike_times = units_tables[table_names[electrode_id]]["spike_times"][unit_id]
    waveform_mean = units_tables[table_names[electrode_id]]["waveform_mean"][unit_id]
    unit_spike_amp = units_tables[table_names[electrode_id]]["spike_amplitudes"][unit_id]

    # Filter valid trials where choiceTarget_ON_time > 0
    valid_trials_mask = trials_timings['choiceTarget_ON_time'] > 0

    # Set trial start time based on the first mark
    trial_center = trials_timings[trial_window_marks[0]][valid_trials_mask]

    # Compute mean intervals between successive marks
    mark_intervals = []
    for i in range(len(trial_window_marks) - 1):
        mark1, mark2 = trial_window_marks[i], trial_window_marks[i + 1]
        interval = trials_timings[mark2][valid_trials_mask] - trials_timings[mark1][valid_trials_mask]
        mean_interval = np.mean(interval)  # Compute mean interval between mark1 → mark2
        mark_intervals.append(mean_interval)

    # Compute trial stop time by adding all mean intervals sequentially
    window_start = trial_window_expanding[0]
    window_stop = np.sum(mark_intervals) + trial_window_expanding[1]
    trial_window = [window_start, window_stop]


    # Compute trial-averaged firing rates
    smoothed_firing_rate_start, time_bins_start = compute_average_firing_rate(unit_spike_times, trial_center, window=trial_window, smooth=smooth, bin_size=0.02)
    num_lines = len(mark_intervals) + 1  # +1 for the first line at 0
    cm = mcolors.LinearSegmentedColormap.from_list(
        "red_purple_blue", ["red", "purple", "blue"]
    )

    # Compute ISI (Inter-Spike Intervals)
    ISIs = np.diff(unit_spike_times) * 1000  # Convert to ms
    ISIs = ISIs[ISIs > 0]  # Remove zero or negative values

    # Sample waveform data (Replace with actual data)
    x_offset = waveform_mean.shape[1] + 5  # Offset for horizontal separation

    # Compute trial-wise mean firing rate and amplitude
    trial_firing_rates = []
    trial_mean_amp = []
    trial_duration = trial_window[1] - trial_window[0]
    for trial_idx in range(len(trial_center)):
        # Count total num of spiking and calculate FR in trial window surround each trial center
        trial_spikes = unit_spike_times[(unit_spike_times >= trial_center[trial_idx]+trial_window[0]) & 
                                        (unit_spike_times <= trial_center[trial_idx]+trial_window[1])]
        
        trial_firing_rates.append(len(trial_spikes) / trial_duration if trial_duration > 0 else 0)

        # fetch all spiking amps and calculate mea amp of that trial
        trial_spikes_amps = abs(unit_spike_amp[(unit_spike_times >= trial_center[trial_idx]+trial_window[0]) & 
                                        (unit_spike_times <= trial_center[trial_idx]+trial_window[1])])
        trial_mean_amp.append(np.mean(trial_spikes_amps))

    # Create 3×4 figure layout
    fig, axes = plt.subplots(
        ncols=4, 
        nrows=3,  # ✅ Only create first 2 rows here
        figsize=(12*resize, 16*resize), 
        gridspec_kw={'width_ratios': [4, 4, 1,1], 'height_ratios': [2, 1, 1]},  # ✅ No row 3 yet!
        sharey='row'
    )  

    # **Row 1: Raster Plots**
    # Raster plot centered around trial start
    gs = axes[0, 0].get_gridspec()  # Get grid layout
    for ax in [axes[0, 0], axes[0, 1]]:  # Remove the individual subplots
        ax.remove()

    # Create a new subplot that spans both columns
    ax_raster = fig.add_subplot(gs[0, 0:2])  

    # Convert list to string
    event_names_str = ", ".join(trial_window_marks)

    # Break into multiple lines every 50 characters
    max_line_length = 50  # Max characters per line
    wrapped_lines = [event_names_str[i:i+max_line_length] for i in range(0, len(event_names_str), max_line_length)]
    # Join lines with newline characters
    formatted_title = "Raster Plot Around Events (zero at 1st):\n" + "\n".join(wrapped_lines)
    # Set title with automatic line wrapping
    ax_raster.set_title(formatted_title, fontsize=14)  # Adjust fontsize as needed


    for trial_idx, start_time in enumerate(trial_center):
        aligned_spikes = unit_spike_times - start_time
        filtered_spikes = aligned_spikes[(aligned_spikes >= trial_window[0]) & (aligned_spikes <= trial_window[1])]
        ax_raster.eventplot(filtered_spikes, lineoffsets=trial_idx, colors='black')

    # Draw first vertical line at 0 with the first color
    line = 0
    ax_raster.axvline(line, color=cm(0), linestyle='--')  
    # Loop through mark intervals and draw vertical lines with different colors
    for i, line_step in enumerate(mark_intervals):
        line += line_step
        ax_raster.axvline(line, color=cm((i + 1) / num_lines), linestyle='--')  # Dynamically select color

    ax_raster.set_xlim(trial_window)
    ax_raster.set_xlabel("Time (s) relative to trial start")
    ax_raster.set_ylabel("Trial Number")

    # **Row 1: Firing Rate and Amplitude**
    axes[0, 2].barh(range(len(trial_firing_rates)), trial_firing_rates, color='blue', height=1)
    axes[0, 2].set_title("Mean Firing Rate")

    axes[0, 3].barh(range(len(trial_mean_amp)), trial_mean_amp, color='blue', height=1)
    axes[0, 3].set_title("Mean Amplitude")

    # **Row 2: Smoothed Trial-Averaged Activity**
    # Trial-averaged activity centered on trial start
    gs = axes[1, 0].get_gridspec()  # Get grid layout
    for ax in [axes[1, 0], axes[1, 1]]:  # Remove the individual subplots
        ax.remove()
    # Create a new subplot that spans both columns
    ax_TrialMean = fig.add_subplot(gs[1, 0:2])  

    ax_TrialMean.fill_between(time_bins_start[1:], smoothed_firing_rate_start, color='black', alpha=0.3)
    ax_TrialMean.plot(time_bins_start[1:], smoothed_firing_rate_start, color='black',linewidth = 0.5)
    # Draw first vertical line at 0 with the first color
    line = 0
    ax_TrialMean.axvline(line, color=cm(0), linestyle='--')  
    # Loop through mark intervals and draw vertical lines with different colors
    for i, line_step in enumerate(mark_intervals):
        line += line_step
        ax_TrialMean.axvline(line, color=cm((i + 1) / num_lines), linestyle='--')  # Dynamically select color
    ax_TrialMean.set_xlim(trial_window)
    ax_TrialMean.set_ylim([0,None])
    ax_TrialMean.set_xlabel("Time (s) relative to trial start")
    ax_TrialMean.set_ylabel("Mean Firing Rate (Hz)")
    ax_TrialMean.set_title("Trial-Averaged Activity (Start)")

    for ax in [axes[1, 2], axes[1, 3], axes[2, 0], axes[2, 1], axes[2, 2], axes[2, 3]]:
        ax.axis("off")

    ### **Manually Create Row 3 (Independent Y-Axes)**
    # **Waveform Plot (1st column)**
    ax_waveform = fig.add_subplot(3, 4, 9)  # ✅ Insert new subplot manually
    ax_waveform.set_title("Mean Waveforms")
    for ch in range(waveform_mean.shape[0]):
        x_values = np.arange(waveform_mean.shape[1]) + ch * x_offset  # Shift each waveform
        ax_waveform.plot(x_values, waveform_mean[ch, :], label=f'Ch {ch+1}', color='black')

    ax_waveform.set_xlabel("Time (samples)")
    ax_waveform.set_ylabel("Amplitude")
    ax_waveform.grid(True, linestyle="--", alpha=0.5)

    # **ISI Histogram (2nd column)**
    ax_isi = fig.add_subplot(3, 4, 10)  # ✅ Insert new subplot manually
    ax_isi.set_title("ISI Histogram (Zoomed to 30ms)")
    ax_isi.hist(ISIs, bins=np.linspace(0, 30, 30), color='blue', alpha=0.7, edgecolor='black', log=False)
    ax_isi.axvline(2, color='red', linestyle="--", label="2ms Refractory Period")
    ax_isi.set_xlabel("Inter-Spike Interval (ms)")
    ax_isi.set_ylabel("counts")
    ax_isi.set_xlim([0,30])

    # **Extract and Display Neuron Metrics**
    fetched_metrices, metrices_names = fetch_key_metrices(units_tables, table_names, electrode_id, unit_id)
    
    text_x, text_y = 0.78, 0.47
    line_spacing = (text_y - 0.05) / max(1, len(metrices_names))

    for i, met_name in enumerate(metrices_names):
        metric_value = fetched_metrices[met_name]
        text_color = "red" if not check_quality(met_name, metric_value) else "black"
        text_line = f"{met_name} = {metric_value:.3f}" if isinstance(metric_value, (int, float)) else f"{met_name} = {metric_value}"
        fig.text(text_x, text_y - i * line_spacing, text_line, fontsize=12, fontweight="bold", color=text_color)

    # **Grand Title**
    i = unit_id
    true_unit_ids = units_tables[table_names[electrode_id]].unit_name[:]
    max_electrode = units_tables[table_names[electrode_id]].max_electrode[:]

    BodyPart_1 = max_electrode['BodyPart_1']
    BodySide_1 = max_electrode['BodySide_1']
    Modality_1 = max_electrode['Modality_1']
    location = max_electrode['location']
    rel_id = max_electrode['rel_id']
    fig.suptitle(
        f'Session: {Path(filepath).stem}, {location.iloc[i]}; Electrode: {table_names[electrode_id]} ({rel_id.iloc[i]}), Unit: {true_unit_ids[i]} \n {BodySide_1.iloc[i]} {BodyPart_1.iloc[i]} {Modality_1.iloc[i]}',
        fontsize=15, fontweight="bold", y=1.)

    plt.tight_layout()
    if not silent_plot:
        plt.show()
    fig_filename = f"{Path(filepath).stem}_{location.iloc[i]}_Electrode-{table_names[electrode_id]}-{rel_id.iloc[i]}_Unit-{true_unit_ids[i]}"

    io.close()
    return fig, fig_filename

def plot_general_metrices_all_units(filepath, output_folder=None, resize=0.9, smooth=False, silent_plot = True):
    """
    Loops through all electrodes and their units in an NWB file, 
    generates visualizations, and saves them.

    Parameters:
        filepath (str): Path to the NWB file.
        output_folder (str, optional): Folder to save the figures (default=None, saves in the current directory).
        resize (float, optional): Resize factor for the plots (default=0.9).
        smooth (bool, optional): Whether to apply smoothing in plots (default=False).
    """

    # Load NWB file
    nwbfile, _ = load_nwb(filepath)

    # Get unit tables and electrode names
    units_tables, table_names = get_units_tables(nwbfile)

    # Get the number of electrodes
    Num_elec = len(table_names)

    # Loop through all electrodes
    for electrode_id in range(Num_elec):
        
        # Get number of units in this electrode
        Num_unit = len(units_tables[table_names[electrode_id]].unit_name[:])

        # Loop through all units in this electrode
        for unit_id in range(Num_unit):
            
            # Generate figure and filename
            fig, fig_filename = plot_neuron_general_metrices(
                filepath, electrode_id=electrode_id, unit_id=unit_id,
                resize=resize, smooth=smooth, silent_plot = silent_plot
            )

            # Save the figure using the specified output folder
            save_figure(fig, handle=".jpg", name=fig_filename, output_folder=output_folder)

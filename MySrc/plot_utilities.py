import numpy as np
import matplotlib.pyplot as plt
from MySrc.general_utilities import compute_average_firing_rate, fetch_key_metrices, check_quality # Import the function

def plot_neuron_general_metrices(
    units_tables, table_names, electrode_id, unit_id, trials_timings, 
    start_window=(-1, 6), stop_window=(-6, 1), smooth=True, resize = 1
):
    """
    Plots neuron activity including raster plots, firing rates, ISI distribution, 
    waveform plots, and quality metrics.

    Parameters:
        units_tables (dict): Dictionary containing spike sorting results.
        table_names (dict): Mapping of electrode IDs to table names.
        electrode_id (int): ID of the electrode.
        unit_id (int): ID of the neuron/unit.
        trials_timings (dict): Dictionary with 'start_times' and 'stop_times' arrays.
        compute_average_firing_rate (function): Function to compute firing rates.
        fetch_key_metrices (function): Function to fetch neuron quality metrics.
        check_quality (function): Function to validate neuron quality.
        start_window (tuple): Time window around trial start (default: (-1,6)).
        stop_window (tuple): Time window around trial stop (default: (-6,1)).
        smooth (bool): Whether to apply smoothing when computing firing rates (default: True).
    """
    
    # Fetch unit data from tables
    unit_spike_times = units_tables[table_names[electrode_id]]["spike_times"][unit_id]
    waveform_mean = units_tables[table_names[electrode_id]]["waveform_mean"][unit_id]
    unit_spike_amp = units_tables[table_names[electrode_id]]["spike_amplitudes"][unit_id]

    # Fetch trial start/stop times
    start_times = trials_timings['start_times']
    stop_times = trials_timings['stop_times']

    # Compute trial-averaged firing rates
    smoothed_firing_rate_start, time_bins_start = compute_average_firing_rate(unit_spike_times, start_times, window=start_window, smooth=smooth, bin_size=0.02)
    smoothed_firing_rate_stop, time_bins_stop = compute_average_firing_rate(unit_spike_times, stop_times, window=stop_window, smooth=smooth, bin_size=0.02)

    # Compute ISI (Inter-Spike Intervals)
    ISIs = np.diff(unit_spike_times) * 1000  # Convert to ms
    ISIs = ISIs[ISIs > 0]  # Remove zero or negative values

    # Sample waveform data (Replace with actual data)
    x_offset = waveform_mean.shape[1] + 5  # Offset for horizontal separation

    # Compute trial-wise mean firing rate and amplitude
    trial_firing_rates = []
    trial_mean_amp = []
    for trial_idx in range(len(start_times)):
        trial_spikes = unit_spike_times[(unit_spike_times >= start_times[trial_idx]) & (unit_spike_times <= stop_times[trial_idx])]
        trial_duration = stop_times[trial_idx] - start_times[trial_idx]
        trial_firing_rates.append(len(trial_spikes) / trial_duration if trial_duration > 0 else 0)

        trial_spikes_amps = abs(unit_spike_amp[(unit_spike_times >= start_times[trial_idx]) & (unit_spike_times <= stop_times[trial_idx])])
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
    for trial_idx, start_time in enumerate(start_times):
        aligned_spikes = unit_spike_times - start_time
        filtered_spikes = aligned_spikes[(aligned_spikes >= start_window[0]) & (aligned_spikes <= start_window[1])]
        axes[0, 0].eventplot(filtered_spikes, lineoffsets=trial_idx, colors='black')

    axes[0, 0].set_title("Raster Plot (Start)")
    axes[0, 0].axvline(0, color='red', linestyle='--')
    axes[0, 0].set_xlim(start_window)
    
    for trial_idx, stop_time in enumerate(stop_times):
        aligned_spikes = unit_spike_times - stop_time
        filtered_spikes = aligned_spikes[(aligned_spikes >= stop_window[0]) & (aligned_spikes <= stop_window[1])]
        axes[0, 1].eventplot(filtered_spikes, lineoffsets=trial_idx, colors='black')

    axes[0, 1].set_title("Raster Plot (Stop)")
    axes[0, 1].axvline(0, color='red', linestyle='--')
    axes[0, 1].set_xlim(stop_window)

    # **Row 1: Firing Rate and Amplitude**
    axes[0, 2].barh(range(len(trial_firing_rates)), trial_firing_rates, color='blue', height=1)
    axes[0, 2].set_title("Mean Firing Rate")

    axes[0, 3].barh(range(len(trial_mean_amp)), trial_mean_amp, color='blue', height=1)
    axes[0, 3].set_title("Mean Amplitude")

    # **Row 2: Smoothed Trial-Averaged Activity**
    axes[1, 0].fill_between(time_bins_start[1:], smoothed_firing_rate_start, color='black', alpha=0.3)
    axes[1, 0].axvline(0, color='red', linestyle='--')
    axes[1, 0].set_title("Trial-Averaged Activity (Start)")
    axes[1, 0].set_xlim(start_window)
    axes[1, 0].set_ylim([0,None])

    axes[1, 1].fill_between(time_bins_stop[1:], smoothed_firing_rate_stop, color='black', alpha=0.3)
    axes[1, 1].axvline(0, color='red', linestyle='--')
    axes[1, 1].set_title("Trial-Averaged Activity (Stop)")
    axes[1, 1].set_xlim(stop_window)
    axes[1, 1].set_ylim([0,None])

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
    fig.suptitle(f"Neuron Metrics for Electrode: {table_names[electrode_id]}, Unit: {unit_id}", fontsize=16, fontweight="bold", y=1.)

    plt.tight_layout()
    plt.show()
    return fig

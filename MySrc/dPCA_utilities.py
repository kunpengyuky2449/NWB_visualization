import numpy as np
from scipy.ndimage import gaussian_filter1d
from itertools import product

def compute_TrNTi_tensor(nwbfile, units_tables, selected_neurons, 
                         event="stim1_ON_time", window_start=-1.0, window_end=6.0, 
                         bin_size=0.01, normalize=True, gaussian_smooth=True, sigma=1):
    """
    Computes a trial-aligned spike activity tensor for selected neurons.

    Parameters:
        nwbfile (NWBFile): NWB file object containing trial and spike data.
        units_tables (dict): Dictionary mapping electrode names to unit tables.
        selected_neurons (list of tuples): List of (electrode_name, unit_name) pairs reported by neuron inpector UI or manually formated in this way.
        event (str, optional): Trial event to align spikes to (default: "stim1_ON_time").
        window_start (float, optional): Time before event to include (default: -1.0).
        window_end (float, optional): Time after event to include (default: 6.0).
        bin_size (float, optional): Bin size for spike counts in seconds (default: 0.01).
        normalize (bool, optional): If True, normalize by subtracting mean activity per neuron (default: True).
        gaussian_smooth (bool, optional): If True, apply Gaussian smoothing (default: True).
        sigma (int, optional): Smoothing factor for Gaussian filter (default: 1).

    Returns:
        trial_tensor (numpy.ndarray): 3D array of shape (num_trials, num_neurons, num_time_bins).
        Or
        normalized_trial_tensor (numpy.ndarray or None): Normalized version of trial_tensor if `normalize=True`, else None.
    """

    # ✅ Define time bins
    time_bins = np.arange(window_start, window_end + bin_size, bin_size)
    num_time_frames = len(time_bins) - 1  # Number of bins

    # ✅ Filter valid trials (ignore early-stopped trials)
    valid_trials_mask = nwbfile.trials["choiceTarget_ON_time"].data[:] > 0
    valid_event_times = nwbfile.trials[event].data[:][valid_trials_mask]
    num_trials = len(valid_event_times)

    # ✅ Initialize tensor (trials × neurons × time_bins)
    trial_tensor = np.zeros((num_trials, len(selected_neurons), num_time_frames))

    # Loop through neurons
    for neuron_idx, (electrode_name, unit_name) in enumerate(selected_neurons):
        # Find correct electrode and unit
        corrected_electrode_name = electrode_name.rsplit('-', 1)[0]
        unit_table = units_tables.get(corrected_electrode_name, None)
        if unit_table is None:
            print(f"⚠ Warning: Electrode {corrected_electrode_name} not found, skipping neuron {unit_name}.")
            continue
        
        true_unit_ids = unit_table.unit_name[:]
        index = np.where(true_unit_ids.astype(str) == str(unit_name))[0]
        if index.size == 0:
            print(f"⚠ Warning: Unit {unit_name} not found in {corrected_electrode_name}, skipping.")
            continue
        index = index[0]

        # Retrieve spike times
        unit_spike_times = unit_table["spike_times"][index]

        # Loop through valid trials
        for trial_idx, event_time in enumerate(valid_event_times):
            # Extract spikes within the observation window
            spike_mask = (unit_spike_times >= event_time + window_start) & (unit_spike_times < event_time + window_end)
            aligned_spikes = unit_spike_times[spike_mask] - event_time  # Align to event

            # Compute spike histogram
            spike_counts, _ = np.histogram(aligned_spikes, bins=time_bins)

            # Apply Gaussian smoothing if enabled
            if gaussian_smooth:
                spike_counts = gaussian_filter1d(spike_counts.astype(float), sigma=sigma)

            # Normalize by subtracting mean if smoothing was applied
            if gaussian_smooth:
                spike_counts -= np.mean(spike_counts)

            # Store in tensor
            trial_tensor[trial_idx, neuron_idx, :] = spike_counts

    # ✅ Compute mean across trials & time for each neuron (shape: (num_neurons, 1, 1))
    if normalize:
        neuron_means = np.mean(trial_tensor, axis=(0, 2), keepdims=True)
        normalized_trial_tensor = trial_tensor - neuron_means
        return normalized_trial_tensor

    return trial_tensor  # No normalization applied

# Example usage:
# trial_tensor, normalized_trial_tensor = compute_trial_tensor(nwbfile, units_tables, selected_neurons)

def compute_dpca_tensor(TrNTi_tensor, categories):
    """
    Constructs a dPCA tensor by averaging trial data across category combinations.

    Parameters:
        TrNTi_tensor (numpy.ndarray): Normalized trial tensor with shape (trials, neurons, time_bins).
        categories (dict): Dictionary mapping category names to arrays of category labels.
                          Each category should have a shape (trials,).

    Example category structure:
        categories["stim_strength"] = np.vectorize(lambda x: value_to_category[x])(temp)
        categories["high_low"] = np.where(np.isin(valid_event_times["selectedChoiceTarget_ID"], [1, 2]), 0, 1)
        categories["left_right"] = np.where(np.isin(valid_event_times["selectedChoiceTarget_ID"], [1, 3]), 0, 1)

    Returns:
        dpca_tensor (numpy.ndarray): dPCA tensor of shape (neurons, time_bins, *category_sizes).
        category_names (list): List of categorical dimension names beyond neurons and time (the first two dims).
    """

    # Extract category names and sizes
    category_names = list(categories.keys())
    category_sizes = [len(np.unique(categories[name])) for name in category_names]

    # Initialize dPCA tensor
    dpca_tensor = np.zeros((TrNTi_tensor.shape[1], TrNTi_tensor.shape[2], *category_sizes))

    # Generate all possible category combinations
    category_combinations = list(product(*[range(size) for size in category_sizes]))

    # Create category label array (trials × num_categories)
    category_array = np.column_stack([categories[name] for name in category_names])

    # Fill dPCA tensor by averaging trials for each category combination
    for combination in category_combinations:
        mask = np.all(category_array == np.array(combination), axis=1)
        matching_trials = TrNTi_tensor[mask]

        if matching_trials.shape[0] > 0:
            dpca_tensor[:, :, *combination] = np.mean(matching_trials, axis=0)
        else:
            print(f"⚠ No data found for category combination: {combination}")

    return dpca_tensor, category_names

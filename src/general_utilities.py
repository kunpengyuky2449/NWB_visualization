from pynwb import NWBHDF5IO

# Prototype function
def load_nwb(file_path):
    """Load an NWB file and return the NWBFile object."""
    io = NWBHDF5IO(file_path, mode="r")
    nwbfile = io.read()
    return nwbfile

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
    - trials_timing_df (pd.DataFrame): A DataFrame with 'start_time' and 'stop_time' columns.
    """
    # Ensure the NWB file contains trial data
    if not hasattr(nwbfile, 'trials') or nwbfile.trials is None:
        print("No trial data found in this NWB file.")
        return None  # Return None if no trial data is available

    # Convert NWB trials table to a Pandas DataFrame
    trials_df = nwbfile.trials.to_dataframe()
    trials_df = nwbfile.trials.to_dataframe()
    # Extract 'start_time' and 'stop_time' columns
    start_times = nwbfile.trials['start_time'].data[:]
    stop_times = nwbfile.trials['stop_time'].data[:]

    if len(start_times) != len(stop_times):
        raise ValueError(f"❌ Mismatch in count: {len(start_times)} start_times vs {len(stop_times)} stop_times.")

    # **Sanity Check 2 & 3: Ensure each start time comes before its corresponding stop time and trials are sequential**
    for i in range(len(start_times)):
        if start_times[i] >= stop_times[i]:  # Start should be strictly before stop
            raise ValueError(f"❌ Timing issue at index {i}: start_time ({start_times[i]}) >= stop_time ({stop_times[i]}).")

        if i > 0 and start_times[i] <= stop_times[i - 1]:  # Ensure start → stop → start → stop sequence
            raise ValueError(f"❌ Overlapping trials at index {i}: start_time ({start_times[i]}) is before previous stop_time ({stop_times[i-1]}).")


    return {'start_times':start_times,'stop_times':stop_times}
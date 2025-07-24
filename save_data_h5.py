import numpy as np
import glob
import re
from datetime import datetime
import pandas as pd
import h5py
import os

# extracting data using a directory:
# directory_path = r"G:\My Drive\ACCL_L3B_3180"
# directory_path = r"/mccfs2/u1/lcls/physics/rf_lcls2/fault_data/ACCL_L3B_3180"
directory_path = r"G:\.shortcut-targets-by-id\1kjgZjwGRIE-5anoMitTfYFQ6bScG9PbZ\Summer_2025\Leila\ACCL_L3B_3180"

# # checks if the directory is found
# print("Exists:", os.path.exists(directory_path)) 
# print("Is Dir:", os.path.isdir(directory_path))

# # create a list of directories to loop through and save to one file for each cryomodule
# directory_list = [
#     r"G:\.shortcut-targets-by-id\1kjgZjwGRIE-5anoMitTfYFQ6bScG9PbZ\Summer_2025\Leila\ACCL_L3B_3110"
#     r"G:\.shortcut-targets-by-id\1kjgZjwGRIE-5anoMitTfYFQ6bScG9PbZ\Summer_2025\Leila\ACCL_L3B_3120"
#     r"G:\.shortcut-targets-by-id\1kjgZjwGRIE-5anoMitTfYFQ6bScG9PbZ\Summer_2025\Leila\ACCL_L3B_3130"
#     r"G:\.shortcut-targets-by-id\1kjgZjwGRIE-5anoMitTfYFQ6bScG9PbZ\Summer_2025\Leila\ACCL_L3B_3140"
# ]

# print all of the file names with "_QUENCH" in the folder using a loop (using glob module)
results = glob.glob(directory_path + '/**/*QUENCH.txt', recursive=True) # force this to have a number before _QUENCH
quench_files = [f for f in results if re.search(r"\d+_QUENCH", f)]
print(quench_files)

# quench_files = []
# for directory_path in directory_list:
#     results = glob.glob(directory_path + '/**/*QUENCH.txt', recursive=True)
#     matched = [f for f in results if re.search(r"\d+_QUENCH", f)]
#     quench_files.extend(matched)
# print(f"Found {len(quench_files)} quench files from directory.")

# creating a function to extract the waveform data and timestamps from each file
def extracting_data(path_name, faultname): 
    with open(path_name, 'r') as file:
        for line in file: 
            if f"{faultname}" in line and f"{faultname}." not in line: 
                data = pd.Series(line.split())
                target_timestamp = line.split()[1]
                values = data[2:].astype(float).values
                return values, target_timestamp
    return None, None

# saving waveform and metadata to an HDF5 file
output_filename = f"cavity_quench_data_L3B_3180.h5"

# this block of code is for saving waveform data to an HDF5 file
with h5py.File(output_filename, 'w') as h5file: 
    for i, file in enumerate(quench_files):
        print("\nProcessing file: " + file)
        
        # getting PV and timestamp information from the file
        filename = os.path.basename(file)   # COMPATABLE WITH OTHER DIRECTORY PATHS
        parts = filename.replace('.txt', '').split('_')
        pv_base = parts[0] + ":" + parts[1] + ":" + parts[2]
        timestamp_raw = parts[3] + "_" + parts[4]
        timestamp = datetime.strptime(timestamp_raw, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d_%H:%M:%S.")
        # SORT THE TIMESTAMP LIST BY YEAR-MONTH-DATE_HOUR:MINUTE:SECOND

        # constructing PV label strings
        cavity_faultname = pv_base + ':CAV:FLTAWF'
        forward_pow = pv_base + ':FWD:FLTAWF'
        reverse_pow = pv_base + ':REV:FLTAWF'
        decay_ref = pv_base + ':DECAYREFWF'  
        time_range = pv_base + ':CAV:FLTTWF'

        # extracting all data for quench waveform using defined function
        cavity_data, cavity_time = extracting_data(file, cavity_faultname)
        forward_data, forward_time = extracting_data(file, forward_pow)
        reverse_data, reverse_time = extracting_data(file, reverse_pow)
        decay_data, decay_time = extracting_data(file, decay_ref)
        time_data, time_timestamp = extracting_data(file, time_range)

        # making them all the same length in case the length varies
        forward_data = forward_data[:len(cavity_data)]
    
        # making each file have a unique group name
        group_name = f"quench_{i:03d}"
        group = h5file.create_group(group_name)

        # saving waveform data into the group
        group.create_dataset('time_seconds', data=time_data)
        group.create_dataset('cavity_amplitude_MV', data=cavity_data)
        group.create_dataset('forward_power_W2', data=forward_data)
        group.create_dataset('reverse_power_W2', data=reverse_data)
        group.create_dataset('decay_reference_MV', data=decay_data)

        # saving metadata as attributes
        group.attrs['filename'] = f"{filename}"
        group.attrs['timestamp'] = cavity_time
        group.attrs['faultname'] = cavity_faultname
        group.attrs['cavity_number'] = parts[2][2]
        group.attrs['cryomodule'] = parts[2][:2] 
print(f"Data from {len(quench_files)} successfully saved to {output_filename}.")
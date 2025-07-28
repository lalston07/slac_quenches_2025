import numpy as np
import glob
import re
from datetime import datetime
import pandas as pd
import h5py
import os
import time

# extracting data using a single directory:
# directory_path = r"G:\My Drive\ACCL_L3B_3180"
# directory_path = r"/mccfs2/u1/lcls/physics/rf_lcls2/fault_data/ACCL_L3B_3180"
# directory_path = r"G:\.shortcut-targets-by-id\1kjgZjwGRIE-5anoMitTfYFQ6bScG9PbZ\Summer_2025\Leila\ACCL_L3B_3180"

CM_num = 31                         # CHANGES FOR EACH FILE/CRYOMODULE
LOADED_Q_CHANGE_FOR_QUENCH = 0.6    # fixed value to determine threshold

# searching for all quench files in the cryomodule
quenches = []
# base_directory = r"/Users/nneveu/Google Drive/My Drive/students/Summer_2025/Leila/" # CHANGE THIS TO THE DIRECTORY WHERE THE FILES ARE STORED
base_directory = r"/mccfs2/u1/lcls/physics/rf_lcls2/fault_data"
CM_matches = glob.glob(base_directory + rf'/ACCL_L*B_{CM_num}*/**/*QUENCH.txt', recursive=True)
matched = [f for f in CM_matches if re.search(r"\d+_QUENCH.txt", f)]
print(f"Matched files from CM{CM_num}:")
quenches.extend(matched)
print(f"Found {len(quenches)} quench files from cryomodule.")

# putting the files in order by timestamp
quench_files = []
for file in quenches:
    filename = os.path.basename(file)
    parts = filename.replace('.txt', '').split('_')
    timestamp_raw = parts[3] + "_" + parts[4]
    timestamp_obj = datetime.strptime(timestamp_raw, "%Y%m%d_%H%M%S")
    quench_files.append((filename, parts, timestamp_raw, timestamp_obj, file))
quench_files.sort(key=lambda x: x[0])

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

def validate_quench(fault_data, time_data, saved_loaded_q, frequency):
    pre_quench_amp = fault_data[0]

    exponential_term = np.polyfit(time_data, np.log(pre_quench_amp / fault_data), 1)[0]
    loaded_q = (np.pi * frequency) / exponential_term

    thresh_for_quench = LOADED_Q_CHANGE_FOR_QUENCH * saved_loaded_q

    is_real = loaded_q < thresh_for_quench

    return saved_loaded_q, loaded_q, is_real

# defining a function to imcrement the quench count
def increment_quench_count(group):
    # if 'quench_count' already exists then we increment it
    # if it doesn't exist yet then we set the value to one
    if "quench_count" in group.attrs:
        group.attrs["quench_count"] += 1
    else:
        group.attrs["quench_count"] = 1

# creating a dictionary to search the cavity number
cavity_num = {
    f"{CM_num}10": "CAV1", 
    f"{CM_num}20": "CAV2",
    f"{CM_num}30": "CAV3",
    f"{CM_num}40": "CAV4",
    f"{CM_num}50": "CAV5",
    f"{CM_num}60": "CAV6",
    f"{CM_num}70": "CAV7",
    f"{CM_num}80": "CAV8",
}

# saving waveform and metadata to an HDF5 file
output_filename = f"quench_data_CM{CM_num}.h5"

# this block of code is for saving waveform data and metadata to an HDF45 File
with h5py.File(output_filename, 'w') as h5file: 
    for i, (filename, parts, timestamp_raw, timestamp_obj, file) in enumerate(quench_files):
        print("\nProcessing file: " + file)
        
        # getting PV and timestamp information from the file
        pv_base = parts[0] + ":" + parts[1] + ":" + parts[2]
        timestamp = timestamp_obj.strftime("%Y-%m-%d_%H:%M:%S.").replace('.','')
        timestamp = timestamp.split('_', 1)[-1] # gives only the HOUR:MINUTE:SECOND

        # formatting date components
        year = str(timestamp_obj.year)
        month = f"{timestamp_obj.month:02d}"
        day = f"{timestamp_obj.day:02d}"

        # GROUP HIERARCHY : CM# (HDF5 file) > CAV# > YEAR > MONTH > DAY > TIMESTAMP
        cavity = cavity_num.get(parts[2])               
        cavity_group = h5file.require_group(cavity)     # '.require_group()' only creates a group if it doesn't already exist
        increment_quench_count(cavity_group)            # if the group already exists then this line returns a reference to the existing group

        year_group = cavity_group.require_group(year) 
        increment_quench_count(year_group)              # incrementing the number of quenches at each level (cavity, year, month, etc)

        month_group = year_group.require_group(month)
        increment_quench_count(month_group)

        day_group = month_group.require_group(day)
        increment_quench_count(day_group)

        quench_group = day_group.create_group(timestamp)

        # constructing PV label strings
        cavity_faultname = pv_base + ':CAV:FLTAWF'
        forward_pow = pv_base + ':FWD:FLTAWF'
        reverse_pow = pv_base + ':REV:FLTAWF'
        decay_ref = pv_base + ':DECAYREFWF'  
        time_range = pv_base + ':CAV:FLTTWF'
        q_value = pv_base + ":QLOADED"          
        freq_value = pv_base + ":FREQ" 

        # extracting all data for quench waveform using defined function
        cavity_data, cavity_time = extracting_data(file, cavity_faultname)
        forward_data, forward_time = extracting_data(file, forward_pow)
        reverse_data, reverse_time = extracting_data(file, reverse_pow)
        decay_data, decay_time = extracting_data(file, decay_ref)
        time_data, time_timestamp = extracting_data(file, time_range)
        q_data, q_time = extracting_data(file, q_value)
        freq_data, freq_time = extracting_data(file, freq_value) 

        saved_loaded_q, calculated_q, classification = validate_quench(cavity_data, time_data, saved_loaded_q=q_data[0], frequency=freq_data[0])

        # making them all the same length in case the length varies
        forward_data = forward_data[:len(cavity_data)]

        # saving waveform data into the group
        quench_group.create_dataset('time_seconds', data=time_data)
        quench_group.create_dataset('cavity_amplitude_MV', data=cavity_data)
        quench_group.create_dataset('forward_power_W2', data=forward_data)
        quench_group.create_dataset('reverse_power_W2', data=reverse_data)
        quench_group.create_dataset('decay_reference_MV', data=decay_data)

        # saving metadata for each quench as attributes
        quench_group.attrs['filename'] = f"{filename}"
        quench_group.attrs['timestamp'] = cavity_time
        quench_group.attrs['faultname'] = cavity_faultname
        quench_group.attrs['quench_classification'] = classification
        quench_group.attrs['saved_q_value'] = saved_loaded_q
        quench_group.attrs['calculated_q_value'] = calculated_q
        quench_group.attrs['cavity_number'] = parts[2][2]
        quench_group.attrs['cryomodule'] = parts[2][:2] 

print(f"Data from {len(quench_files)} successfully saved to {output_filename}.")

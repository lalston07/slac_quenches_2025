import numpy as np
import glob
import re
from datetime import datetime
import pandas as pd
import h5py

# to extract data using a directory:
directory_path = r"G:\My Drive\ACCL_L3B_3180"

# print all of the file names with "_QUENCH" in the folder using a loop (using glob module)
results = glob.glob(directory_path + '/**/*QUENCH.txt', recursive=True) # force this to have a number before _QUENCH
print(f"Found {len(results)} matching QUENCH text files:")
quench_files = [f for f in results if re.search(r"\d+_QUENCH", f)]
print(f"Found {len(quench_files)} matching '##_QUENCH' text files:")
print(quench_files)

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
output_filename = 'cavity_quench_data_google_drive.h5'

# this block of code is for saving waveform data to an HDF5 file
with h5py.File(output_filename, 'w') as h5file: 
    for i, file in enumerate(quench_files):
        print("\nProcessing file: " + file)
        
        # getting PV and timestamp information from the file
        filename = file.split("\\", 4)[-1].replace('.txt','') 
        parts = filename.split('_')
        pv_base = parts[0] + ":" + parts[1] + ":" + parts[2]
        timestamp_raw = parts[3] + "_" + parts[4]
        timestamp = datetime.strptime(timestamp_raw, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d_%H:%M:%S.")

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
        group.attrs['filename'] = f"{filename}.txt"
        group.attrs['timestamp'] = cavity_time
        group.attrs['faultname'] = cavity_faultname
        group.attrs['cavity_number'] = parts[2][2]
        group.attrs['cryomodule'] = parts[2][:2] 
print(f"Data from {len(quench_files)} successfully saved to {output_filename}.")

"""
# this function prints the quench file number and prints the dataset names and attributes for each file
with h5py.File("cavity_quench_data_google_drive.h5", "r") as file: 
    def print_h5_structure(name, object):
        print(name)
        if object.attrs:
            print("  Attributes:")
            for key, value in object.attrs.items():
                print(f"    {key}: {value}")
    file.visititems(print_h5_structure)
"""
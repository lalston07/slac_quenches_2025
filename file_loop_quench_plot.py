import matplotlib.pyplot as plt
import numpy as np
import io
import glob
import os
import re
import time
from datetime import datetime
import pandas as pd

# to extract data using a directory:
directory_path = r"G:\My Drive\ACCL_L3B_3180"

# checking if the directory is found
# if os.path.isdir(directory_path):
#   print("Google Drive directory found.")
# else: 
#   print("Directory not found.")

# print all of the file names with "_QUENCH" in the folder using a loop (using glob module)
results = glob.glob(directory_path + '/**/*QUENCH.txt', recursive=True) # force this to have a number before _QUENCH
print(f"Found {len(results)} matching QUENCH text files:")
# print(results) # results is a list
quench_files = [f for f in results if re.search(r"\d+_QUENCH", f)]
print(f"Found {len(quench_files)} matching '##_QUENCH' text files:")
print(quench_files)

data_long=[]

for file in quench_files: 
    print("\nProcessing file: " + file)
    
    # getting PV and timestamp information from {file}
    # line below splits the file in to 4 parts (after the '\') and gets the last part (filename)
    filename = file.split("\\", 4)[-1].replace('.txt','') 
    print(filename)
    parts = filename.split('_') # splits the filename into parts at each '_'
    pv_base = parts[0] + ":" + parts[1] + ":" + parts[2]    # ex: pt(0): ACCL, pt(1): L3B, pt(2):3180
    timestamp_raw = parts[3] + "_" + parts[4]               # ex: pt(3): 20221028, pt(4): 235218
    # line below formats the timestamp to match the file layout
    timestamp = datetime.strptime(timestamp_raw, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d_%H:%M:%S.")
    print("PV label: " + pv_base)
    print("Timestamp: " + timestamp + "\n")

    # constructing PV label strings
    cavity_faultname = pv_base + ':CAV:FLTAWF'  # ex: ACCL:L3B:3180:CAV:FLTAWF
    forward_pow = pv_base + ':FWD:FLTAWF'       # ex: ACCL:L3B:3180:FWD:FLTAWF
    reverse_pow = pv_base + ':REV:FLTAWF'       # ex: ACCL:L3B:3180:REV:FLTAWF
    decay_ref = pv_base + ':DECAYREFWF'         # ex: ACCL:L3B:3180:DECAYREFWF    
    time_range = pv_base + ':CAV:FLTTWF'         # ex: ACCL:L3B:3180:CAV:FLTTWF

    # creating a function to extract the waveform data and timestamps from each file
    def extracting_data(path_name, faultname): 
        with open(path_name, 'r') as file:
            for line in file: 
                if f"{faultname}" in line and f"{faultname}." not in line: 
                    data = pd.Series(line.split())
                    target_timestamp = line.split()[1]  # searching for timestamp in case it varies
                    values = data[2:].astype(float).values

                    print(f"{faultname} Information:")
                    print(f"Length of data: {len(values)}")
                    print(f"First value: {values[0]}, Last value: {values[-1]}")
                    print(f"Min value: {np.min(values)}, Max value: {np.max(values)}\n")

                    return values, target_timestamp
        return None, None   # added in case the PV line is not found
    
    # extract each waveform using defined function
    cavity_data, cavity_time = extracting_data(file, cavity_faultname)
    forward_data, forward_time = extracting_data(file, forward_pow)
    reverse_data, reverse_time = extracting_data(file, reverse_pow)
    decay_data, decay_time = extracting_data(file, decay_ref)
    time_data, time_timestamp = extracting_data(file, time_range)
 
    # used for the following files because there is more forward_data than cavity_data which leads to error: 
    # ['ACCL_L3B_3180_20240509_143717_QUENCH.txt', 
    # 'ACCL_L3B_3180_20240509_144621_QUENCH.txt', 
    # 'ACCL_L3B_3180_20240509_143712_QUENCH.txt', 
    # 'ACCL_L3B_3180_20240509_143606_QUENCH.txt', 
    # 'ACCL_L3B_3180_20240509_143700_QUENCH.txt']

    if len(forward_data) > len(cavity_data):
        data_long.append(f"{filename}.txt")
    forward_data = forward_data[:len(cavity_data)] 

    # plotting the data together
    if "ACCL_L3B_3180_20240509_143717_QUENCH.txt" in file or "ACCL_L3B_3180_20241002_151904_QUENCH.txt" in file:
        plt.figure(figsize=(14,6))
        plt.plot(time_data, cavity_data, label="Cavity (MV)", color='blue', linewidth=3)
        plt.plot(time_data, forward_data, label="Forward Power (W)", color='green', linewidth=3)
        plt.plot(time_data, reverse_data, label="Reverse Power (W)", color='red', linewidth=3)
        plt.plot(time_data, decay_data, label="Decay Reference (MV)", color='cyan', linewidth=1)
        plt.xlabel("Time in Seconds")
        plt.ylabel("Amplitude")
        plt.title(f"Cavity Quench Waveform for {cavity_faultname} {cavity_time}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

print(f"Files where forward data is greater than cavity data: {data_long}")






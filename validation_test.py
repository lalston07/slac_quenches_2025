import matplotlib.pyplot as plt
import numpy as np
import io
import glob
import os
import re
import time
from datetime import datetime
import pandas as pd
# import statement for LOADED_Q_CHANGE_FOR_QUENCH in valdate_quench function from Lisa's code
# from applications.quench_processing.quench_utils import LOADED_Q_CHANGE_FOR_QUENCH
# LOADED_Q_CHANGE_FOR_QUENCH = 0.6

# to extract data using a directory:
directory_path = r"G:\My Drive\ACCL_L3B_3180"
LOADED_Q_CHANGE_FOR_QUENCH = 0.6

# print all of the file names with "_QUENCH" in the folder using a loop (using glob module)
file_results = glob.glob(directory_path + '/**/*QUENCH.txt', recursive=True) # force this to have a number before _QUENCH
print(f"Found {len(file_results)} matching QUENCH text files:")
quench_files = [f for f in file_results if re.search(r"\d+_QUENCH", f)]
print(f"Found {len(quench_files)} matching '##_QUENCH' text files:")
# print(quench_files)

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

# re-writing Lisa's function to validate the quenches (real vs fake)
def validate_quench(fault_data, time_data, saved_loaded_q, frequency, wait_for_update: bool=False, logger=None):
    """
    Parsing the fault waveforms to calculate the loaded Q to try to determine
    if a quench was real.

    DERIVATION NOTES
    A(t) = A0 * e^((-2 * pi * cav_freq * t)/(2 * loaded_Q)) = A0 * e ^ ((-pi * cav_freq * t)/loaded_Q)

    ln(A(t)) = ln(A0) + ln(e ^ ((-pi * cav_freq * t)/loaded_Q)) = ln(A0) - ((pi * cav_freq * t)/loaded_Q)
    polyfit(t, ln(A(t)), 1) = [-((pi * cav_freq)/loaded_Q), ln(A0)]
    polyfit(t, ln(A0/A(t)), 1) = [(pi * f * t)/Ql]

    https://education.molssi.org/python-data-analysis/03-data-fitting/index.html

    :param wait_for_update: bool
    :return: bool representing whether quench was real
    """

    if wait_for_update:
        print(f"Waiting 0.1s to give {fault_data} waveforms a chance to update")
        time.sleep(0.1)
    
    time_0 = 0
    for time_0, timestamp in enumerate(time_data):
        if timestamp >= 0:
            break
    
    fault_data = fault_data[time_0:]
    time_data = time_data[time_0:]

    end_decay = len(fault_data) - 1

    # to find where amplitude decays to "zero"
    for end_decay, amp in enumerate(fault_data):
        if amp < 0.002:
            break

    fault_data = fault_data[:end_decay]
    time_data = time_data[:end_decay]

    pre_quench_amp = fault_data[0]  # pre-quench amplitude

    exponential_term = np.polyfit(time_data, np.log(pre_quench_amp / fault_data), 1)[0] # PLOT WHAT WE ARE FITTING AND EXP TERM
    loaded_q = (np.pi * frequency) / exponential_term

    thresh_for_quench = LOADED_Q_CHANGE_FOR_QUENCH * saved_loaded_q
    print(f"Saved Q: {saved_loaded_q:.2e}")
    print(f"Last recorded amplitude: {fault_data[0]}")
    print(f"Threshold: {thresh_for_quench:.2e}")
    print(f"Calculated Loaded Q: {loaded_q:.2e}")

    is_real = loaded_q < thresh_for_quench
    print(f"Validation: {is_real}")
    
    return is_real

# initializations before file loop
real_quenches = []
results = []
count_false = 0
count_true = 0

for file in quench_files: 
    print("\nProcessing file: " + file)
    
    # getting PV and timestamp information from {file}
    # line below splits the file in to 4 parts (after the '\') and gets the last part (filename)
    filename = file.split("\\", 4)[-1].replace('.txt','') 
    print(filename)
    parts = filename.split('_') # splits the filename into parts at each '_'
    pv_base = parts[0] + ":" + parts[1] + ":" + parts[2]    # ex: pt(1): ACCL, pt(2): L3B, pt(3):3180
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
    time_range = pv_base + ':CAV:FLTTWF'        # ex: ACCL:L3B:3180:CAV:FLTTWF
    q_value = pv_base + ":QLOADED"              # ex: ACCL:L3B:3180:QLOADED 
    freq_value = pv_base + ":FREQ"              # ex: ACCL:L3B:3180:FREQ
    
    # extract each waveform using defined function
    cavity_data, cavity_time = extracting_data(file, cavity_faultname)
    forward_data, forward_time = extracting_data(file, forward_pow)
    reverse_data, reverse_time = extracting_data(file, reverse_pow)
    decay_data, decay_time = extracting_data(file, decay_ref)
    time_data, time_timestamp = extracting_data(file, time_range)
    q_data, q_time = extracting_data(file, q_value)                 # used to find saved_loaded_q from validate_quench
    freq_data, freq_time = extracting_data(file, freq_value)        # used to find frequency from validate_quench

    # running validation
    # frequency is fixed value (1.3 GHz) and saved_loaded_q varies for each file
    is_real = validate_quench(cavity_data, time_data, saved_loaded_q=q_data[0], frequency=freq_data[0])  
    # results.append({"filename": f"{filename}.txt", "timestamp": cavity_time, "real_quench": is_real})
    results.append({"filename": f"{filename}.txt", "waveform_data": cavity_data, "Q_0": q_data, "real quench": is_real})

    if is_real:
        count_true += 1
        real_quenches.append(filename)
    else:
        count_false += 1
    
# converting results list to DataFrame
validation_results = pd.DataFrame(results)

# # saving results from all files as tab-separated .txt file
# validation_results.to_csv("quench_validation_results.txt", sep='\t', index=False)
# print(f"Number of fake quenches: {count_false}, and number of real quenches: {count_true}")
# print("Saved results to quench_validation_results.txt")

# saving results from all files as column separated .csv file
# validation_results.to_csv("quench_validation_results.csv", index=False)
print(f"Number of fake quenches: {count_false}, and number of real quenches: {count_true}")
print(f"Real Quench Files: {real_quenches}")
print("Saved results to quench_validation_results.csv")

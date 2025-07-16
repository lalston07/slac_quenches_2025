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
LOADED_Q_CHANGE_FOR_QUENCH = 0.6

# print all of the file names with "_QUENCH" in the folder using a loop (using glob module)
file_results = glob.glob(directory_path + '/**/*QUENCH.txt', recursive=True)
print(f"Found {len(file_results)} matching QUENCH text files:")
quench_files = [f for f in file_results if re.search(r"\d+_QUENCH", f)]
print(f"Found {len(quench_files)} matching '##_QUENCH' text files:")

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

def validate_quench(file_name, fault_data, time_data, saved_loaded_q, frequency):
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

    # using the numpy.polyfit() fitting method
    # option #1 - use ln(A0 / A(t))
    log_function = np.log(pre_quench_amp / fault_data)
    exponential_term, intercept = np.polyfit(time_data, log_function, 1)
        # np.polyfit(...) fits a straight line and returns the slope and intercept (for first order/degree) in an array
        # np.polyfit(...) is a method to finding the best fit polynomial coefficients
        # example returning (slope, intercept) = [5.12e+06 -0.45] means that the best fit line is y = (5.12*10^6)*t - 0.45
    print(f"Exponential term is {exponential_term}")
    print(f"Best fit polynomial is y={exponential_term:.2e}*t + ({intercept})")
    loaded_q = (np.pi * frequency) / exponential_term

    # option #2 - use ln(A(t))
    exponential_term, intercept = np.polyfit(time_data, np.log(fault_data), 1)
    print(f"Exponential term is {exponential_term}")
    print(f"Best fit polynomial is y={exponential_term:.2e}*t + ({intercept})")
    loaded_q = (np.pi * frequency) / (- exponential_term)   # this function returns negative slope

    thresh_for_quench = LOADED_Q_CHANGE_FOR_QUENCH * saved_loaded_q
    print(f"Saved Q: {float(saved_loaded_q):.2e}")
    print(f"Last recorded amplitude: {fault_data[0]}")
    print(f"Threshold: {float(thresh_for_quench):.2e}")
    print(f"Calculated Loaded Q: {float(loaded_q):.2e}")

    is_real = loaded_q < thresh_for_quench
    print(f"Validation: {is_real}")

    # exponential decay fit - plots A(t) = A_0 * e^(-pi*f*t / loaded_q)
    # plots the exponential decay curve that is fitted to the data
    A_fit = pre_quench_amp * np.exp((-np.pi * frequency * time_data) / loaded_q)   
    plt.figure(figsize=(8, 5))      
    plt.plot(time_data, fault_data, 'bo', label = 'Original Amplitude Data')    # blue dots are the raw data
    plt.plot(time_data, A_fit, 'r--', label='Fitted Exponential Curve')         # red dashed line is the fitting curve
    plt.xlabel("Time in Seconds")       
    plt.ylabel("Amplitude")
    plt.title(f"Exponential Fit to Fault Decay for {file_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return is_real, exponential_term

for file in quench_files: 
    print(f"Processing {file} for fitting curve plot:")
    
    filename = file.split("\\", 4)[-1].replace('.txt','') 
    print(filename)
    parts = filename.split('_')
    pv_base = parts[0] + ":" + parts[1] + ":" + parts[2]
    timestamp_raw = parts[3] + "_" + parts[4]
    timestamp = datetime.strptime(timestamp_raw, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d_%H:%M:%S.")
    print("PV label: " + pv_base)
    print("Timestamp: " + timestamp + "\n")

    cavity_faultname = pv_base + ':CAV:FLTAWF'
    time_range = pv_base + ':CAV:FLTTWF'
    q_value = pv_base + ":QLOADED"
    freq_value = pv_base + ":FREQ"

    cavity_data, cavity_time = extracting_data(file, cavity_faultname)
    time_data, time_timestamp = extracting_data(file, time_range)
    q_data, q_time = extracting_data(file, q_value)
    freq_data, freq_time = extracting_data(file, freq_value)

    validate_quench(f"{filename}.txt", cavity_data, time_data, q_data, freq_data)

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
LOADED_Q_CHANGE_FOR_QUENCH = 0.6

# print all of the file names with "_QUENCH" in the folder using a loop (using glob module)
file_results = glob.glob(directory_path + '/**/*QUENCH.txt', recursive=True)
print(f"Found {len(file_results)} matching QUENCH text files:")
quench_files = [f for f in file_results if re.search(r"\d+_QUENCH", f)]
print(f"Found {len(quench_files)} matching '##_QUENCH' text files:")

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

def validate_quench(file_name, fault_data, time_data, saved_loaded_q, frequency):
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

    # using the numpy.polyfit() fitting method
    # option #1 - use ln(A0 / A(t))
    log_function = np.log(pre_quench_amp / fault_data)
    exponential_term, intercept = np.polyfit(time_data, log_function, 1)
        # np.polyfit(...) fits a straight line and returns the slope and intercept (for first order/degree) in an array
        # np.polyfit(...) is a method to finding the best fit polynomial coefficients
        # example returning (slope, intercept) = [5.12e+06 -0.45] means that the best fit line is y = (5.12*10^6)*t - 0.45
    print(f"Exponential term is {exponential_term}")
    print(f"Best fit polynomial is y={exponential_term:.2e}*t + ({intercept})")
    loaded_q = (np.pi * frequency) / exponential_term

    # option #2 - use ln(A(t))
    exponential_term, intercept = np.polyfit(time_data, np.log(fault_data), 1)
    print(f"Exponential term is {exponential_term}")
    print(f"Best fit polynomial is y={exponential_term:.2e}*t + ({intercept})")
    loaded_q = (np.pi * frequency) / (- exponential_term)   # this function returns negative slope

    thresh_for_quench = LOADED_Q_CHANGE_FOR_QUENCH * saved_loaded_q
    print(f"Saved Q: {float(saved_loaded_q):.2e}")
    print(f"Last recorded amplitude: {fault_data[0]}")
    print(f"Threshold: {float(thresh_for_quench):.2e}")
    print(f"Calculated Loaded Q: {float(loaded_q):.2e}")

    is_real = loaded_q < thresh_for_quench
    print(f"Validation: {is_real}")

    # exponential decay fit - plots A(t) = A_0 * e^(-pi*f*t / loaded_q)
    # plots the exponential decay curve that is fitted to the data
    A_fit = pre_quench_amp * np.exp((-np.pi * frequency * time_data) / loaded_q)   
    plt.figure(figsize=(8, 5))      
    plt.plot(time_data, fault_data, 'bo', label = 'Original Amplitude Data')    # blue dots are the raw data
    plt.plot(time_data, A_fit, 'r--', label='Fitted Exponential Curve')         # red dashed line is the fitting curve
    plt.xlabel("Time in Seconds")       
    plt.ylabel("Amplitude")
    plt.title(f"Exponential Fit to Fault Decay for {file_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return is_real, exponential_term

for file in quench_files: 
    print(f"Processing {file} for fitting curve plot:")
    
    filename = file.split("\\", 4)[-1].replace('.txt','') 
    print(filename)
    parts = filename.split('_')
    pv_base = parts[0] + ":" + parts[1] + ":" + parts[2]
    timestamp_raw = parts[3] + "_" + parts[4]
    timestamp = datetime.strptime(timestamp_raw, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d_%H:%M:%S.")
    print("PV label: " + pv_base)
    print("Timestamp: " + timestamp + "\n")

    cavity_faultname = pv_base + ':CAV:FLTAWF'
    time_range = pv_base + ':CAV:FLTTWF'
    q_value = pv_base + ":QLOADED"
    freq_value = pv_base + ":FREQ"

    cavity_data, cavity_time = extracting_data(file, cavity_faultname)
    time_data, time_timestamp = extracting_data(file, time_range)
    q_data, q_time = extracting_data(file, q_value)
    freq_data, freq_time = extracting_data(file, freq_value)

    validate_quench(f"{filename}.txt", cavity_data, time_data, q_data, freq_data)

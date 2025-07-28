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
def validate_quench(fault_data, time_data, saved_loaded_q, frequency):
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

    pre_quench_amp = fault_data[0]

    exponential_term = np.polyfit(time_data, np.log(pre_quench_amp / fault_data), 1)[0]
    loaded_q = (np.pi * frequency) / exponential_term

    thresh_for_quench = LOADED_Q_CHANGE_FOR_QUENCH * saved_loaded_q
    print(f"Saved Q: {saved_loaded_q:.2e}")
    print(f"Last recorded amplitude: {fault_data[0]}")
    print(f"Threshold: {thresh_for_quench:.2e}")
    print(f"Calculated Loaded Q: {loaded_q:.2e}")

    is_real = loaded_q < thresh_for_quench
    print(f"Validation: {is_real}")
    
    """
    CALCULATION OF FITTING ERROR
    Calculates how well the exponential fit A(t) = A0 * exp(-pi*f*t / Q) matches the waveform data.
    Returns error metrics like MSE, RMSE, and residuals.
    """
    # fitted_amplitude is created using the dacay rate (exponential term) that we got from np.polyfit
    fitted_amplitude = pre_quench_amp * np.exp(-exponential_term * time_data)

    # error metrics using sklearn methods (import sklearn.metrics)
    # rmse = np.sqrt(sklearn.metrics.mean_squared_error(fault_data, fitted amplitude))
    # rmse = sklearn.metrics.root_mean_squared_error(fault_data, fitted_amplitude)
    # r2 = sklearn.metrics.r2_score(fault_data, fitted_amplitude)

    # error metrics using numpy
    rmse = np.sqrt(np.mean((fault_data - fitted_amplitude)**2))
    r2 = 1 - (np.sum((fault_data - fitted_amplitude)**2) / np.sum((fault_data - np.mean(fault_data))**2))
    
    # error metrics using np.corrcoef
    # correlation_matrix = np.corrcoef(fault_data, fitted_amplitude)
    # correlation = correlation_matrix[0, 1]
    # r2 = correlation**2

    print("\nFit Error Metrics From Polyfit Method: ")
    print(f"RMSE = {rmse}")
    print(f"R^2 = {r2}")

    # # plotting the fit over the raw cavity amplitude data
    # plt.figure(figsize=(8, 5))
    # plt.plot(time_data, fault_data, label='Raw Amplitude Data', marker='o')
    # plt.plot(time_data, fitted_amplitude, label='Linear Exponential Fit', linestyle='--')
    # plt.xlabel("Time in Seconds")
    # plt.ylabel("Amplitude")
    # plt.title("Exponential Fit vs Raw Amplitude")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    # plt.close()

    return is_real, rmse, r2

# initializations before file loop
real_quenches = []
results = []
count_false = 0
count_true = 0
rsme_values = []
r2_values = []
filename_list = []

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
    # SORT THE TIMESTAMP LIST BY YEAR-MONTH-DATE_HOUR:MINUTE:SECOND
    
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
    classification, rmse, r2 = validate_quench(cavity_data, time_data, saved_loaded_q=q_data[0], frequency=freq_data[0])
    rsme_values.append(rmse)
    r2_values.append(r2)
    
    # results.append({"filename": f"{filename}.txt", "timestamp": cavity_time, "real_quench": is_real})
    results.append({"filename": f"{filename}.txt", "waveform_data": cavity_data, "Q_0": q_data, "real quench": classification, "root_mean_sqaured_error": rmse, "r_squared_score": r2})

    if classification:
        count_true += 1
        real_quenches.append(filename)
    else:
        count_false += 1

    filename_list.append(f"{timestamp} - {classification}")
    
# converting results list to DataFrame
validation_results = pd.DataFrame(results)

# convering filename_list to a datatime object
timestamp_label = []
classification_label = []
for ts_label in filename_list:
    ts_part, class_part = ts_label.split(" - ")
    ts_part = ts_part.rstrip(".")
    timestamp_label.append(datetime.strptime(ts_part, "%Y-%m-%d_%H:%M:%S"))
    classification_label.append(class_part)
    

# # saving results from all files as tab-separated .txt file
# validation_results.to_csv("quench_validation_results.txt", sep='\t', index=False)
# print(f"Number of fake quenches: {count_false}, and number of real quenches: {count_true}")
# print("Saved results to quench_validation_results.txt")

# saving results from all files as column separated .csv file
# validation_results.to_csv("quench_validation_error.csv", index=False)
print(f"Number of fake quenches: {count_false}, and number of real quenches: {count_true}")
print(f"Real Quench Files: {real_quenches}")
# print("Saved results to quench_validation_error.csv")

# plotting RMSE values 
fig, ax1 = plt.subplots(figsize=(14,6))
ax1.set_xlabel('Quench Timestamp and Classification', fontsize=14)
ax1.set_ylabel('RMSE Value', color='blue', fontsize=14)
ax1.plot(filename_list, rsme_values, marker='o', color='blue', label='RMSE Values')
ax1.tick_params(axis='y', labelcolor='blue')
plt.xticks(rotation=90)
ax1.grid(True, linestyle='--')

# plotting R² values
ax2 = ax1.twinx()
ax2.set_ylabel('R² Value', color='darkorange', fontsize=14)
ax2.plot(filename_list, r2_values, marker='o', color='darkorange', label='R²')
ax2.set_ylim(-2, 1.2)
ax2.tick_params(axis='y', labelcolor='darkorange')

fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
fig.suptitle(f"Quench Validation: RSME and R² Data for Cryomodule {parts[2][:2]} Cavity {parts[2][2]}", fontsize=14)
fig.tight_layout()
# fig.savefig('validation_error_calculation.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

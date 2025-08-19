import h5py
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import re
import os
from test_rmse_r2 import cm_cavity_rmse_dict    # dictionary with average RMSE values per cavity
LOADED_Q_CHANGE_FOR_QUENCH = 0.6

def updated_validate_quench(fault_data, time_data, saved_loaded_q, frequency, avg_RMSE):
    time_0 = 0
    for time_0, timestamp in enumerate(time_data):
        if timestamp >= 0:
            break
    
    fault_data = fault_data[time_0:]
    time_data = time_data[time_0:]

    end_decay = len(fault_data) - 1
    for end_decay, amp in enumerate(fault_data):
        if amp < 0.002:
            break
    
    fault_data = fault_data[:end_decay]
    time_data = time_data[:end_decay]

    pre_quench_amp = fault_data[0]

    exponential_term = np.polyfit(time_data, np.log(pre_quench_amp / fault_data), 1)[0]
    loaded_q = (np.pi * frequency) / exponential_term

    thresh_for_quench = LOADED_Q_CHANGE_FOR_QUENCH * saved_loaded_q

    is_real = loaded_q < thresh_for_quench

    """
    CALCULATION OF FITTING ERROR
    Calculates how well the exponential fit A(t) = A0 * exp(-pi*f*t / Q) matches the waveform data.
    Returns error metrics like MSE, RMSE, and residuals.
    """

    # fitted_amplitude is created using the dacay rate (exponential term) that we got from np.polyfit
    fitted_amplitude = pre_quench_amp * np.exp(-exponential_term * time_data)

    # error metrics using numpy
    rmse = np.sqrt(np.mean((fault_data - fitted_amplitude)**2))
    r2 = 1 - (np.sum((fault_data - fitted_amplitude)**2) / np.sum((fault_data - np.mean(fault_data))**2))

    if rmse > avg_RMSE:
        if 0 < r2 < 0.5:
            new_classification = False 
        elif -20 < r2 < 0:
            new_classification = False
        elif r2 <= -20:     # range is -inf to -20
            new_classification = True
        else:               # if r2 > 0.5
            new_classification = True 
    else:
        if r2 < 0:
            new_classification = False
        elif r2 >= 0.5:
            new_classification = False
        else:   # if r2 > 0.5
            new_classification = False

    return new_classification, rmse, r2


folder_path = "C:/Users/leila/Documents/Visual Studio/slac_quenches_2025/quench_data_per_cryomodule"
h5_files = [f for f in os.listdir(folder_path) if f.endswith('.h5')]

# initializing dictionaries
rmse_per_cryomodule = {}
waveform_data_per_cryomodule = {}
real_quenches_per_cryo = {}   
fake_quenches_per_cryo = {}
real_quench_count_new = {}
fake_quench_count_new = {}
valid_real_per_cryo = {}
valid_fake_per_cryo = {}
valid_misclassification_per_cryo = {}
misclassified_real = {}
misclassified_fake = {}

# for file in h5_files:
for file in h5_files:
    file_path = os.path.join(folder_path, file)
    print(f"\nProcessing: {file_path} - {os.path.basename(file_path)}")
    cryo = os.path.basename(file_path).replace("quench_data_CM", "").split(".")[0]

    # initializing lists for waveform plots
    quench_names = []
    cavity_waveforms = []
    forward_waveforms = []
    reverse_waveforms = []
    decay_waveforms = []
    time_waveforms = []

    # initializing lists for classification validation
    rmse_values = []
    r2_values = []

    current_classifications = []
    real_quenches_per_cryo[cryo] = 0
    fake_quenches_per_cryo[cryo] = 0

    new_classifications = []
    real_quench_count_new[cryo] = 0
    fake_quench_count_new[cryo] = 0

    valid_real_per_cryo[cryo] = 0
    valid_fake_per_cryo[cryo] = 0
    
    valid_misclassification_per_cryo[cryo] = 0
    misclassified_real[cryo] = 0
    misclassified_fake[cryo] = 0

    with h5py.File(file_path, 'r') as f:

        for cavity in f.keys():
            cavity_group = f[cavity]
        
            for year in cavity_group.keys():
                year_group = cavity_group[year]
                
                for month in year_group.keys():
                    month_group = year_group[month]
                    
                    for day in month_group.keys():
                        day_group = month_group[day]

                        for quench_timestamp in day_group.keys():
                            quench_group = day_group[quench_timestamp]

                            try:
                                # getting waveform data for each quench
                                time_data = quench_group['time_seconds'][:]
                                cavity_data = quench_group['cavity_amplitude_MV'][:]
                                forward_data = quench_group['forward_power_W2'][:]
                                reverse_data = quench_group['reverse_power_W2'][:]
                                decay_data = quench_group['decay_reference_MV'][:]
                                q_value = quench_group.attrs.get("saved_q_value")

                                # getting metadata
                                filename = quench_group.attrs.get("filename")
                                current_classification = quench_group.attrs.get("quench_classification", None)
                                current_classifications.append(current_classification)

                                # calculating error metrics
                                avg_RMSE = cm_cavity_rmse_dict[cryo][cavity]
                                new_classification, rmse, r2 = updated_validate_quench(cavity_data, time_data, saved_loaded_q=q_value, frequency=1300000000.0, avg_RMSE=avg_RMSE)
                                new_classifications.append(new_classification)
                                rmse_values.append(rmse)
                                r2_values.append(r2)

                            except Exception as e:
                                # print(f"Error in file {file_path}, timestamp {quench_timestamp}: {e}")
                                continue

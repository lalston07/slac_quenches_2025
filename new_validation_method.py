import h5py
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import re
import os
LOADED_Q_CHANGE_FOR_QUENCH = 0.6

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

    :return: bool representing whether quench was real
    """
    
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

    return is_real

def updated_validate_quench(fault_data, time_data, saved_loaded_q, frequency, cavity_avg):
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

    if rmse > cavity_avg:
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

# using a number of sample files to make process shorter
sample_files = ["31"]
filtered_h5_files = [f for f in h5_files if any(cm in f for cm in sample_files)]

# for file in h5_files:
for file in filtered_h5_files:
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
    new_classifications = []
    real_quench_count = 0
    fake_quench_count = 0
    real_quench_count_new = 0
    fake_quench_count_new = 0
    valid_real = 0
    valid_fake = 0
    valid_misclassification = 0

    with h5py.File(file_path, 'r') as f:

        for cavity in f.keys():
            cavity_group = f[cavity]

            if cryo == "31" and cavity == "CAV8":   # changes for what subset of data is being evaluated
                for year in cavity_group.keys():
                    year_group = cavity_group[year]

                    # getting quench count from cavity as a whole
                    cavity_quench_count = cavity_group.attrs.get("quench_count", 0)
                    
                    for month in year_group.keys():
                        month_group = year_group[month]
                        
                        for day in month_group.keys():
                            day_group = month_group[day]

                            for quench_timestamp in day_group.keys():
                                quench_group = day_group[quench_timestamp]

                                try:
                                    # getting waveform data for each "real" quench
                                    time_data = quench_group['time_seconds'][:]
                                    cavity_data = quench_group['cavity_amplitude_MV'][:]
                                    forward_data = quench_group['forward_power_W2'][:]
                                    reverse_data = quench_group['reverse_power_W2'][:]
                                    decay_data = quench_group['decay_reference_MV'][:]
                                    q_value = quench_group.attrs.get("saved_q_value")

                                    # saving waveforms
                                    cavity_waveforms.append(cavity_data)
                                    forward_waveforms.append(forward_data)
                                    reverse_waveforms.append(reverse_data)
                                    decay_waveforms.append(decay_data)
                                    time_waveforms.append(time_data)

                                    # getting metadata
                                    filename = quench_group.attrs.get("filename")
                                    current_classification = quench_group.attrs.get("quench_classification", None)
                                    current_classifications.append(current_classification)

                                    # calculating error metrics
                                    is_real = validate_quench(cavity_data, time_data, saved_loaded_q=q_value, frequency=1300000000.0)
                                    new_classification, rmse, r2 = updated_validate_quench(cavity_data, time_data, saved_loaded_q=q_value, frequency=1300000000.0, cavity_avg= )
                                    new_classifications.append(new_classification)

                                    # determining statistics for new classification
                                    if new_classification == True:
                                        real_quench_count_new += 1
                                    else: 
                                        fake_quench_count_new += 1

                                    # determining statistics for current classification
                                    if current_classification == True:
                                        real_quench_count += 1
                                    else:
                                        fake_quench_count += 1

                                    # determining statistics for misclassifications
                                    if new_classification == current_classification:
                                        if new_classification == True:
                                            valid_real += 1
                                        else:
                                            valid_fake += 1
                                    else:
                                        valid_misclassification += 1
                                        if new_classification == True:
                                            valid_real += 1
                                        else: 
                                            valid_fake += 1

                                    full_timestamp = f"{cavity}-{year}-{month}-{day}-{quench_timestamp}"
                                    quench_names.append(full_timestamp)

                                except Exception as e:
                                    # print(f"Error in file {file_path}, timestamp {quench_timestamp}: {e}")
                                    continue

# storing rmse data per file/cryomodule in dictionary
rmse_per_cryomodule[file_path] = {
    'rmse': rmse_values,
    'classification': current_classifications,
    'quench': quench_names, 
    'r2' : r2_values
}

# storing waveform data per file/cryomodule in dictionary
waveform_data_per_cryomodule[file_path] = {
    'cavity': cavity_waveforms,
    'forward': forward_waveforms,
    'reverse': reverse_waveforms,
    'decay': decay_waveforms,
    'time': time_waveforms,
}

# plotting waveforms for each quench
for file_path, waveforms in waveform_data_per_cryomodule.items():
    quench_labels = rmse_per_cryomodule[file_path]['quench']
    rmse_value = rmse_per_cryomodule[file_path]['rmse']
    classification_value = rmse_per_cryomodule[file_path]['classification']

    cavity_list = waveforms['cavity']
    forward_list = waveforms['forward']
    reverse_list = waveforms['reverse']
    decay_list = waveforms['decay']
    time_list = waveforms['time']

    for i in range(len(cavity_list)):
        # print(f"Quench File Count: {i+1}")
        fig2, ax2 = plt.subplots(figsize=(8,5))
        ax3 = ax2.twinx()
        time = time_list[i]
        cav, = ax2.plot(time, cavity_list[i], label='Cavity Amplitude (MV)', color='#377eb8')
        fwd, = ax3.plot(time, forward_list[i], label='Forward Power (W²)', color='#4daf4a')
        rev, = ax3.plot(time, reverse_list[i], label='Reverse Power (W²)', color='#e41a1c')
        dec, = ax2.plot(time, decay_list[i], label='Decay Reference (MV)', color='#ff7f00', linestyle='--')
        ax2.set_xlim(-0.03, 0.03)
        ax2.set_ylim(cavity_list[i].min(), cavity_list[i].max() + 20)
        ax3.set_ylim(reverse_list[i].min(), reverse_list[i].max())
        ax2.set_title(f'Waveforms for {quench_labels[i]} in {os.path.basename(file_path)} \n(RMSE={rmse_value[i]}; R2={r2_values[i]}) and classification is {classification_value[i]}', fontsize=12)
        ax2.set_xlabel('Time in Seconds')
        ax2.set_ylabel('Amplitude in MV')
        ax3.set_ylabel('Power in W²')
        ax3.legend(
                handles=[cav, dec, fwd, rev],
                labels=["Cavity Amplitude (MV)", "Decay Reference (MV)", "Forward Power (W²)", "Reverse Power (W²)"],
                loc='upper right'
            )
        ax2.grid(True, alpha=0.5)
        plt.tight_layout()
        plt.show()
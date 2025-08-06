import h5py
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import re
import os
LOADED_Q_CHANGE_FOR_QUENCH = 0.6

def validate_quench(fault_data, time_data, saved_loaded_q, frequency):
    # starts the time closer to when the quench happens to make the fit more accurate
    time_0 = 0
    for time_0, timestamp in enumerate(time_data):
        if timestamp >= 0:
            break
     
    fault_data = fault_data[time_0:]
    time_data = time_data[time_0:]

    # ends the time closer to when the quench is over to eliminate when the amplitude=0
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

    # plotting the fit over the raw cavity amplitude data
    plt.figure(figsize=(8, 5))
    plt.plot(time_data, fault_data, label='Raw Amplitude Data', marker='o')
    plt.plot(time_data, fitted_amplitude, label='Linear Exponential Fit', linestyle='--')
    plt.xlabel("Time in Seconds")
    plt.ylabel("Amplitude")
    plt.ylim(0, 15)
    plt.title(f"Exponential Fit vs Raw Amplitude (RMSE = {rmse})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.close()

    return rmse, r2

folder_path = "C:/Users/leila/Documents/Visual Studio/slac_quenches_2025/quench_data_per_cryomodule"
h5_files = [f for f in os.listdir(folder_path) if f.endswith('.h5')]

"""
This section of code calculates the RMSE values for each "real" quench to access accuracy of the classification.
The goal is to determine a threshold RMSE value above which the classification of a quench as "real" can be considered reliable.

Questions answered with this plot:
(1) How many quenches were falsely classified?
(2) Above what RMSE value are we certain that the quenches are real?
"""

# initializing dictionaries
rmse_per_cryomodule = {}
waveform_data_per_cryomodule = {}

# sample_files = ["02", "03", "05", "05", "06", "10", "11", "14", "17"
#                 "18", "19", "21", "23", "28", "30"]
sample_files = ["31"]
filtered_h5_files = [f for f in h5_files if any(cm in f for cm in sample_files)]

rmse_values_real = []
rmse_values_fake = []

for file in filtered_h5_files:
    file_path = os.path.join(folder_path, file)
    print(f"\nProcessing: {file_path} - {os.path.basename(file_path)}")

    # initializign lists
    rmse_values = []
    r2_values = []
    quench_names = []
    cavity_waveforms = []
    forward_waveforms = []
    reverse_waveforms = []
    decay_waveforms = []
    time_waveforms = []

    # initializing variables
    real_quench_count = 0
    fake_quench_count = 0
    false_classification_count = 0

    with h5py.File(file_path, 'r') as f:

        for cavity_num in f.keys():
            cavity_group = f[cavity_num]

            if cavity_num == "CAV8":
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

                                classification = quench_group.attrs.get("quench_classification", None)
                                filename = quench_group.attrs.get("filename")

                                if classification == True: 
                                    # if quench_timestamp=="10:11:37" or quench_timestamp=="10:11:42":
                                    #     false_classification_count += 1
                                    # else:
                                    real_quench_count += 1

                                    try:
                                        # getting waveform data for each "real" quench
                                        time_data = quench_group['time_seconds'][:]
                                        cavity_data = quench_group['cavity_amplitude_MV'][:]
                                        forward_data = quench_group['forward_power_W2'][:]
                                        reverse_data = quench_group['reverse_power_W2'][:]
                                        decay_data = quench_group['decay_reference_MV'][:]
                                        q_value = quench_group.attrs.get("saved_q_value")
                                        
                                        # calculating RMSE
                                        print(f"Fit for {cavity_num}/{year}/{month}/{day}/{quench_timestamp}")
                                        rmse, r2 = validate_quench(cavity_data, time_data, saved_loaded_q=q_value, frequency=1300000000.0)
                                        rmse_values.append(rmse)
                                        r2_values.append(r2)

                                        # getting label with full timestamp
                                        full_timestamp = f"{cavity_num}-{year}-{month}-{day}-{quench_timestamp}"
                                        quench_names.append(full_timestamp)

                                        # saving waveforms
                                        cavity_waveforms.append(cavity_data)
                                        forward_waveforms.append(forward_data)
                                        reverse_waveforms.append(reverse_data)
                                        decay_waveforms.append(decay_data)
                                        time_waveforms.append(time_data)
                                    except Exception as e:
                                        print(f"Error in file {file_path}, timestamp {quench_timestamp}: {e}")
                                else:
                                    fake_quench_count += 1
                                    if quench_timestamp=="11:23:38":
                                        time_data = quench_group['time_seconds'][:]
                                        cavity_data = quench_group['cavity_amplitude_MV'][:]
                                        forward_data = quench_group['forward_power_W2'][:]
                                        reverse_data = quench_group['reverse_power_W2'][:]
                                        decay_data = quench_group['decay_reference_MV'][:]
                                        q_value = quench_group.attrs.get("saved_q_value")

                                        print(f"\nFake Quench Fit for {cavity_num}/{year}/{month}/{day}/{quench_timestamp}")
                                        rmse, r2 = validate_quench(cavity_data, time_data, saved_loaded_q=q_value, frequency=1300000000.0)
                                        rmse_values.append(rmse)
                                        r2_values.append(r2)

                                        full_timestamp = f"{cavity_num}-{year}-{month}-{day}-{quench_timestamp}"
                                        quench_names.append(full_timestamp)

                                        # saving waveforms
                                        cavity_waveforms.append(cavity_data)
                                        forward_waveforms.append(forward_data)
                                        reverse_waveforms.append(reverse_data)
                                        decay_waveforms.append(decay_data)
                                        time_waveforms.append(time_data)
    
    # storing rmse data per file/cryomodule in dictionary
    rmse_per_cryomodule[file_path] = {
        'rmse': rmse_values,
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

# for file_path, data in rmse_per_cryomodule.items():
#     if not data['rmse']:
#         continue

#     fig, ax = plt.subplots(figsize=(14, 6))
#     # bars = ax.bar(data['quench'], data['rmse'], color='darkorange')
#     ax.scatter(data['quench'], data['rmse'], color='blue', label="RMSE Value")
#     ax.scatter(data['quench'], data['r2'], color='green', label="R2 Score")

#     # for bar in bars:
#     #     height = bar.get_height()
#     #     ax.text(
#     #         bar.get_x() + bar.get_width() / 2,
#     #         height + 0.01,    # Adjust offset as needed
#     #         f"{height:.3f}",  # You can round the RMSE value
#     #         ha='center',
#     #         va='bottom',
#     #         fontsize=8
#     #     )

#     # plotting RMSE distribution 
#     ax.set_title(f'RMSE per Real Quench in "{os.path.basename(file_path)}"')
#     ax.set_xlabel('Quench Timestamp')
#     ax.set_ylabel('RMSE Value')
#     # ax.set_ylim(0, 40)
#     ax.set_xticks(range(len(data['quench'])))
#     ax.set_xticklabels(data['quench'], rotation=90, ha='right')
#     ax.grid(True, alpha=0.4)
#     plt.tight_layout()
#     plt.show()

print(f"Real Quenches in CM31 CAV8: {real_quench_count}")
print(f"Fake Quenches in CM31 CAV8: {fake_quench_count}")
print(f"Falsely Classifed Quenches in CM31 CAV8: {false_classification_count}")

# pie chart of real vs fake classified quenches in the whole machine
labels = ['Real Quenches', 'Fake Quenches']
sizes = [real_quench_count, fake_quench_count]
colors = ['#4daf4a', '#e41a1c']
fig4, ax4 = plt.subplots()
ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
ax4.set_title('Real vs Fake Classifications Quenches CM31 CAV8 (2022-2025)')
ax4.axis('equal')   # equal aspect ratio makes the pie chart a circle
plt.show()

# plotting waveforms for each "real" quench
for file_path, waveforms in waveform_data_per_cryomodule.items():
    quench_labels = rmse_per_cryomodule[file_path]['quench']
    
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
        ax2.set_title(f'Waveforms for {quench_labels[i]} in {os.path.basename(file_path)}', fontsize=12)
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
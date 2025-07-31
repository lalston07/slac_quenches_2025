import h5py
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import re
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

    return rmse




"""
This section of code plots the number of quenches per cryomodule (real or fake)

Questions answered with this plot:
(1) Which cryomodule quenched the most?
(2) Which cryomodule quenched the least?
"""
import os
import h5py
import matplotlib.pyplot as plt

folder_path = "C:/Users/leila/Documents/Visual Studio/slac_quenches_2025/quench_data_per_cryomodule"
h5_files = [f for f in os.listdir(folder_path) if f.endswith('.h5')]

quench_counts_per_cryo = {} # initializing dictionary 
cryo_names = []             # initializing list

for file in h5_files:
    file_path = os.path.join(folder_path, file)
    cryo_label = file.replace("quench_data_", "").replace(".h5", "")
    cryo_names.append(cryo_label)   # cryo_label is the key

    if cryo_label not in quench_counts_per_cryo:
        quench_counts_per_cryo[cryo_label] = 0

    with h5py.File(file_path, 'r') as f:
        for cavity_num in f.keys():
            cav_group = f[cavity_num]
            quench_counts_per_cryo[cryo_label] += cav_group.attrs.get('quench_count', 0)

for cryomodule, count in quench_counts_per_cryo.items():   
    print(f"{cryomodule}: {count} total quenches")

# plot for quenches per cryomodule
plt.figure(figsize=(12,6))
bars = plt.bar(cryo_names, quench_counts_per_cryo.values(), color='skyblue')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 30, str(height), ha='center', fontsize=8)
plt.xlabel('Cryomodule Number', fontsize=14)
plt.ylabel('Total Number of Quenches', fontsize=14)
plt.title('Number of Quenches Per Cryomodule (2022-2025)', fontsize=14)
plt.xticks(rotation=90)
plt.tight_layout()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()




"""
This section of code plots the number of real and fake quenches per cryomodule 

Questions answered with this plot:
(1) How many real quenches per cryomodule?
(2) How many fake quenches per cryomodule?
"""
real_quenches_per_cryo = {}
fake_quenches_per_cryo = {}

for file in h5_files:
    file_path = os.path.join(folder_path, file)
    cryo_label = file.replace("quench_data_", "").replace(".h5", "")

    # intializing counts for real and fake classifications
    real_quenches_per_cryo[cryo_label] = 0
    fake_quenches_per_cryo[cryo_label] = 0

    with h5py.File(file_path, 'r') as f:
        for cavity_num in f.keys():
            cavity_group = f[cavity_num]

            for year in cavity_group.keys():
                year_group = cavity_group[year]
                for month in year_group.keys():
                    month_group = year_group[month]
                    for day in month_group.keys():
                        day_group = month_group[day]

                        for quench_timestamp in day_group.keys():
                            quench_group = day_group[quench_timestamp]
                            
                            classification = quench_group.attrs.get("quench_classification", None)
                            if classification is None:
                                continue

                            if classification == True:
                                real_quenches_per_cryo[cryo_label] += 1
                            elif classification == False:
                                fake_quenches_per_cryo[cryo_label] += 1

print("Real Quenches Per Cryomodule (Classified by Validation Method):\n")
for cryomodule, count in real_quenches_per_cryo.items():   
    print(f"{cryomodule}: {count} real quenches")

print("Fake Quenches Per Cryomodule (Classified by Validation Method):\n")
for cryomodule, count in fake_quenches_per_cryo.items():   
    print(f"{cryomodule}: {count} fake quenches")

all_cryomodules = list(real_quenches_per_cryo.keys())
real_counts = [real_quenches_per_cryo[cm] for cm in all_cryomodules]
fake_counts = [fake_quenches_per_cryo[cm] for cm in all_cryomodules]

x = np.arange(len(all_cryomodules))
width = 0.4

# plotting both real and fake quench data on scatter plot
fig, ax = plt.subplots(figsize=(15, 9))
real_bars = ax.bar(x, real_counts, bottom=fake_counts, label='Real Quenches', color='green')
fake_bars = ax.bar(x, fake_counts, label='Fake Quenches', color='red')
ax.set_xlabel('Cryomodule', fontsize=14)
ax.set_ylabel('Number of Quenches', fontsize=14)
ax.set_ylim(0, 5100)
ax.set_title('Real vs Fake Quenches per Cryomodule (2022-2025)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(all_cryomodules, rotation=90)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# plotting only real quench data
fig2, ax2 = plt.subplots(figsize=(15, 7))
real_bars = ax2.bar(x, real_counts, label='Real Quenches', color='green')
for bar in real_bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, height + 30, str(height), ha='center', fontsize=8)
ax2.set_xlabel('Cryomodule', fontsize=14)
ax2.set_ylabel('Number of Quenches', fontsize=14)
ax2.set_title('Real Quenches per Cryomodule (2022-2025)', fontsize=14)
ax2.set_xticks(x)
ax2.set_xticklabels(all_cryomodules, rotation=90)
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# plotting only fake quench data
fig3, ax3 = plt.subplots(figsize=(15, 7))
fake_bars = ax3.bar(x, fake_counts, label='Fake Quenches', color='red')
for bar in fake_bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2, height + 30, str(height), ha='center', fontsize=8)
ax3.set_xlabel('Cryomodule', fontsize=14)
ax3.set_ylabel('Number of Quenches', fontsize=14)
ax3.set_title('Fake Quenches per Cryomodule (2022-2025)', fontsize=14)
ax3.set_xticks(x)
ax3.set_xticklabels(all_cryomodules, rotation=90)
ax3.legend()
ax3.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# pie chart of real vs fake classified quenches in the whole machine
labels = ['Real Quenches', 'Fake Quenches']
sizes = [sum(real_counts), sum(fake_counts)]
colors = ['green', 'red']
fig4, ax4 = plt.subplots()
ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
ax4.set_title('Overall Quench Classification CM01-CM35 (2022-2025)')
ax4.axis('equal')   # equal aspect ratio makes the pie chart a circle
plt.show()





"""
This section of code calculates the RMSE values for each "real" quench to access accuracy of the classification.
The goal is to determine a threshold RMSE value above which the classification of a quench as "real" can be considered reliable.

Questions answered with this plot:
(1) How many quenches were falsely classified?
(2) Above what RMSE value are we certain that the quenches are real?
"""
# initializing dictionary for RMSE values per cryomodule
rmse_per_cryomodule = {}
sample_files = ["02", "03", "05", "05", "06", "10", "11", "14", "17"
                "18", "19", "21", "23", "28", "30"]
filtered_h5_files = [f for f in h5_files if any(cm in f for cm in sample_files)]

for file in filtered_h5_files:
    file_path = os.path.join(folder_path, file)
    rmse_values = []
    quench_names = []

    with h5py.File(file_path, 'r') as f:

        for cavity_num in f.keys():
            cavity_group = f[cavity_num]

            for year in cavity_group.keys():
                year_group = cavity_group[year]
                
                for month in year_group.keys():
                    month_group = year_group[month]
                    
                    for day in month_group.keys():
                        day_group = month_group[day]

                        for quench_timestamp in day_group.keys():
                            quench_group = day_group[quench_timestamp]

                            classification = quench_group.attrs.get("quench_classification", None)
                            
                            if classification == True:
                                time_data = quench_group['time_seconds'][:]
                                cavity_data = quench_group['cavity_amplitude_MV'][:]
                                q_value = quench_group.attrs.get("saved_q_value")
                                
                                try:
                                    rmse = validate_quench(cavity_data, time_data, saved_loaded_q=q_value, frequency=1300000000.0)
                                    rmse_values.append(rmse)
                                    full_timestamp = f"{cavity_num}-{year}-{month}-{day}-{quench_timestamp}"
                                    quench_names.append(full_timestamp)
                                except Exception as e:
                                    print(f"Error in file {file_path}, timestamp {quench_timestamp}: {e}")
    
    # storing rmse data per file/cryomodule in dictionary
    rmse_per_cryomodule[file_path] = {
        'rmse': rmse_values,
        'quench': quench_names
    }

for file_path, data in rmse_per_cryomodule.items():
    if not data['rmse']:
        continue

    figure, axis = plt.subplots(figsize=(14, 6))
    bars = axis.bar(data['quench'], data['rmse'], color='darkorange')
    for bar in bars:
        height = bar.get_height()
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,  # Adjust offset as needed
            f"{height:.3f}",  # You can round the RMSE value
            ha='center',
            va='bottom',
            fontsize=8
        )
    axis.set_title(f'RMSE per Real Quench in "{os.path.basename(file_path)}"')
    axis.set_xlabel('Quench Timestamp')
    axis.set_ylabel('RMSE Value')
    axis.set_xticks(range(len(data['quench'])))
    axis.set_xticklabels(data['quench'], rotation=90, ha='right')
    axis.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()
import h5py
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
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
    
    print(f"Time Zero: {time_0}")
    print(f"End Decay: {end_decay}")

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

    # error metrics using numpy
    rmse = np.sqrt(np.mean((fault_data - fitted_amplitude)**2))
    r2 = 1 - (np.sum((fault_data - fitted_amplitude)**2) / np.sum((fault_data - np.mean(fault_data))**2))

    print("\nFit Error Metrics From Polyfit Method: ")
    print(f"RMSE = {rmse}")
    print(f"R^2 = {r2}")

    return is_real, rmse, r2, time_0, end_decay




"""
THIS WAS BEFORE PUTTING BACK THE CODE FROM LISA'S VALIDATION METHOD

Error Message:
    /sdf/home/l/lalston/save_data_h5.py:52: RuntimeWarning: divide by zero encountered in divide
    exponential_term = np.polyfit(time_data, np.log(pre_quench_amp / fault_data), 1)[0]

LOOK AT THIS FILE TO SEE IF NP.POLYFIT() METHOD WAS MESSED UP SINCE SOME VALUES OF cavity_data WERE ZERO
ACCL_L3B_3280_20241002_151905_QUENCH.txt
"""
# filename = "quench_data_CM32_v2.h5"
# day_path = "CAV1/2022/07/21"        # 20220721
# timestamp_list = ["11:01:57", "11:08:55", "13:52:15", "18:52:42", "12:31:16", "20:05:15", "17:18:06", "18:24:29"]
# with h5py.File(filename, 'r') as f:
#     if day_path in f:
#         day_group = f[day_path]
#         print(f"\n\nFound day group: {day_path}")
#         for timestamp in day_group:
#             if timestamp in timestamp_list:
#                 quench_group = day_group[timestamp]
#                 print(f"\n\n    Timestamp: {timestamp}")

#                 # reading waveform data
#                 time_data = quench_group['time_seconds'][:]
#                 cavity_data = quench_group['cavity_amplitude_MV'][:]
#                 forward_power = quench_group['forward_power_W2'][:]
#                 reverse_power = quench_group['reverse_power_W2'][:]
#                 decay_ref = quench_group['decay_reference_MV'][:]

#                 np.set_printoptions(threshold=np.inf)
#                 print("Cavity Data:\n", cavity_data)
#                 print("Decay Reference:\n", decay_ref)

#                 # for i, data_point in enumerate(cavity_data):
#                 #     if np.isclose(data_point, 0, atol=1e-5):  # adjust tolerance if needed
#                 #         print(f"⚠️ Near-zero cavity amplitude at index {i}, value: {data_point}")

#                 # metadata = {data: quench_group.attrs[data] for data in quench_group.attrs}
#                 for metadata_key, metadata_value in quench_group.attrs.items():
#                     print(f"{metadata_key}: {metadata_value}")
#                 """
#                     Metadata: {'calculated_q_value': nan, 'cavity_number': '8', 'cryomodule': '32', 'faultname': 'ACCL:L3B:3280:CAV:FLTAWF', 
#                     'filename': 'ACCL_L3B_3280_20241002_151905_QUENCH.txt', 'quench_classification': False, 'saved_q_value': 41681390.47526922, 
#                     'timestamp': '2024-10-02_15:19:05.864747'}

#                     Calculated Q-value is 'nan' because we divided by zero from the cavity_data in the np.polyfit() validation method
#                     This quench was classified as fake but how? Looking at the waveform plot, it was a fake quench.
#                         >> is_real = loaded_q < thresh_for_quench
#                         >> is_real = NaN < thresh_for_quench always returns boolean False
#                     How can we confirm that this quench was fake if the loaded_q value could not be calculated?
#                 """
            
#                 saved_loaded_q = quench_group.attrs['saved_q_value']
                
#                 try:
#                     classification, rmse, r2, time_0, end_decay = validate_quench(cavity_data, time_data, saved_loaded_q= saved_loaded_q, frequency=1300000000.0)
#                     print(f"Time Zero: {time_0}, End Decay: {end_decay}")
#                     print(f"RMSE Value is {rmse}, and R^2 Score is {r2}")
#                 except IndexError as e:
#                     print(f"Processing {filename} failed with {e}")

#                 # plotting waveform data for each group/quench file
#                 plt.figure(figsize=(14,6))
#                 plt.plot(time_data, cavity_data, label='Cavity Data (MV)', linewidth=3)
#                 plt.plot(time_data, forward_power, label='Forward Power (W^2)', linewidth=2)
#                 plt.plot(time_data, reverse_power, label='Reverse Power (W^2)', linewidth=2)
#                 plt.xlim(-0.02, 0.02)
#                 plt.plot(time_data, decay_ref, label='Decay Reference (MV)', linestyle='--', linewidth=1.5)
#                 plt.xlabel('Time in Seconds')
#                 plt.ylabel('Amplitude')
#                 plt.title(f'Cavity Quench Waveform\n{filename}')
#                 plt.legend()
#                 plt.grid(True)
#                 plt.tight_layout()
#                 plt.show()



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
plt.bar(cryo_names, quench_counts_per_cryo.values(), color='skyblue')
plt.xlabel('Cryomodule Number', fontsize=14)
plt.ylabel('Total Number of Quenches', fontsize=14)
plt.title('Number of Quenches Per Cryomodule (2022–2025)', fontsize=14)
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
invalid_quenches_per_cryo = {}

for file in h5_files:
    file_path = os.path.join(folder_path, file)
    cryo_label = file.replace("quench_data_", "").replace(".h5", "")

    # intializing counts for real and fake classifications
    real_quenches_per_cryo[cryo_label] = 0
    fake_quenches_per_cryo[cryo_label] = 0
    invalid_quenches_per_cryo[cryo_label] = 0

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

# plotting both real and fake quench data
fig, ax = plt.subplots(figsize=(15, 7))
real_bars = ax.bar(x - width/2, real_counts, width, label='Real Quenches', color='green')
fake_bars = ax.bar(x + width/2, fake_counts, width, label='Fake Quenches', color='red')
ax.set_xlabel('Cryomodule', fontsize=14)
ax.set_ylabel('Number of Quenches', fontsize=14)
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
ax2.set_xlabel('Cryomodule', fontsize=14)
ax2.set_ylabel('Number of Quenches', fontsize=14)
ax2.set_title('Real vs Fake Quenches per Cryomodule (2022-2025)', fontsize=14)
ax2.set_xticks(x)
ax2.set_xticklabels(all_cryomodules, rotation=90)
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# plotting only fake quench data
fig3, ax3 = plt.subplots(figsize=(15, 7))
fake_bars = ax3.bar(x, fake_counts, label='Fake Quenches', color='red')
ax3.set_xlabel('Cryomodule', fontsize=14)
ax3.set_ylabel('Number of Quenches', fontsize=14)
ax3.set_title('Real vs Fake Quenches per Cryomodule (2022-2025)', fontsize=14)
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
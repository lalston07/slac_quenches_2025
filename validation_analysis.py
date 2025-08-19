import h5py
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import re
import os
from avg_rmse_per_cav import cm_cavity_rmse_dict
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

# # using a number of sample files to make process shorter
# sample_files = ["06", "31"]
# filtered_h5_files = [f for f in h5_files if any(cm in f for cm in sample_files)]

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

                # getting quench count from cavity as a whole
                cavity_quench_count = cavity_group.attrs.get("quench_count", 0)
                
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
                                avg_RMSE = cm_cavity_rmse_dict[cryo][cavity]
                                new_classification, rmse, r2 = updated_validate_quench(cavity_data, time_data, saved_loaded_q=q_value, frequency=1300000000.0, avg_RMSE=avg_RMSE)
                                new_classifications.append(new_classification)
                                rmse_values.append(rmse)
                                r2_values.append(r2)

                                # determining statistics for new classification
                                if new_classification == True:
                                    real_quench_count_new[cryo] += 1
                                else: 
                                    fake_quench_count_new[cryo] += 1

                                # determining statistics for current classification
                                if current_classification == True:
                                    real_quenches_per_cryo[cryo] += 1
                                else:
                                    fake_quenches_per_cryo[cryo] += 1

                                # determining statistics for misclassifications
                                if new_classification == current_classification:
                                    if new_classification == True:
                                        valid_real_per_cryo[cryo] += 1
                                    else:
                                        valid_fake_per_cryo[cryo] += 1
                                else:
                                    valid_misclassification_per_cryo[cryo] += 1
                                    if new_classification == True:
                                        misclassified_real[cryo] += 1
                                    else: 
                                        misclassified_fake[cryo] += 1

                                full_timestamp = f"{cavity}-{year}-{month}-{day}-{quench_timestamp}"
                                quench_names.append(full_timestamp)

                            except Exception as e:
                                # print(f"Error in file {file_path}, timestamp {quench_timestamp}: {e}")
                                continue
                
# storing rmse data per file/cryomodule in dictionary
rmse_per_cryomodule[file_path] = {
    'rmse': rmse_values,
    'classification': new_classifications,
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

# lists for current method
all_cryomodules = list(real_quenches_per_cryo.keys())
real_counts_current = [real_quenches_per_cryo[cm] for cm in all_cryomodules]
fake_counts_current = [fake_quenches_per_cryo[cm] for cm in all_cryomodules]

# lists for new method
real_counts_new = [real_quench_count_new[cm] for cm in all_cryomodules]
fake_counts_new = [fake_quench_count_new[cm] for cm in all_cryomodules]

# lists for statistics including misclassifications
valid_real = [valid_real_per_cryo[cm] for cm in all_cryomodules]
valid_fake = [valid_fake_per_cryo[cm] for cm in all_cryomodules]
valid_misclassification = [valid_misclassification_per_cryo[cm] for cm in all_cryomodules]

# lists for misclassifications details
misclassification_real = [misclassified_real[cm] for cm in all_cryomodules]
misclassification_fake = [misclassified_fake[cm] for cm in all_cryomodules]

# plotting pie chart with real, fake, and misclassifications
labels = ['Real Quenches', 'Fake Quenches', 'Misclassified Quenches']
sizes = [sum(valid_real), sum(valid_fake), sum(valid_misclassification)]
colors = ['#4daf4a', '#e41a1c', '#999999']
fig4, ax4 = plt.subplots()
ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
ax4.set_title('Overall Quench Classification CM01-CM35 (2022-2025)')
ax4.axis('equal')   # equal aspect ratio makes the pie chart a circle
plt.show()

# plotting pie chart with real and fake classifications from current method
labels = ['Real Quenches', 'Fake Quenches']
sizes = [sum(real_counts_current), sum(fake_counts_current)]
colors = ['#4daf4a', '#e41a1c']
fig5, ax5 = plt.subplots()
ax5.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
ax5.set_title('Quench Classifications with Current Method CM01-CM35 (2022-2025)')
ax5.axis('equal')   # equal aspect ratio makes the pie chart a circle
plt.show()

# plotting pie chart with real and fake classifications from new method
labels = ['Real Quenches', 'Fake Quenches']
sizes = [sum(real_counts_new), sum(fake_counts_new)]
colors = ['#4daf4a', '#e41a1c']
fig6, ax6 = plt.subplots()
ax6.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
ax6.set_title('Quench Classifications with New Method CM01-CM35 (2022-2025)')
ax6.axis('equal')   # equal aspect ratio makes the pie chart a circle
plt.show()

# plotting real, fake, misclassified real, and misclassified fake from comparisons
labels = ['Real Quenches', 'Fake Quenches', 'Misclassified as Fake (Real)', 'Misclassified as Real (Fake)']
sizes = [sum(valid_real), sum(valid_fake), sum(misclassification_real), sum(misclassification_fake)]
colors = ['#4daf4a', '#e41a1c', '#377eb8','#ff7f00']
fig6, ax6 = plt.subplots()
ax6.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
ax6.set_title('Overall Quench Classification CM01-CM35 (2022-2025)')
ax6.axis('equal')   # equal aspect ratio makes the pie chart a circle
plt.show()

print(f"Total valid real: {sum(valid_real)}")
print(f"Total valid fake: {sum(valid_fake)}")
print(f"Total misclassifications: {sum(valid_misclassification)}")

# # plotting both real and fake quench data on bar chart (LOG SCALE)
# fig8, ax8 = plt.subplots(figsize=(30, 10))
# x = np.arange(len(all_cryomodules))
# real_bars = ax8.bar(x, valid_real, label='Real Quenches', color='#4daf4a')
# fake_bars = ax8.bar(x, valid_fake, bottom=valid_real, label='Fake Quenches', color='#e41a1c')
# ax8.set_xlabel('Cryomodule', fontsize=14)
# ax8.set_ylabel('Number of Quenches', fontsize=14)
# ax8.set_yscale('log')
# ax8.set_title('Real vs Fake Quenches per Cryomodule on Log Scale (2022-2025)', fontsize=14)
# ax8.set_xticks(x)
# ax8.set_xticklabels(all_cryomodules, rotation=90)
# ax8.legend()
# ax8.grid(True, alpha=0.5)
# ax8.set_axisbelow(True)
# plt.tight_layout()
# plt.show()

# # plotting waveforms for each quench
# for file_path, waveforms in waveform_data_per_cryomodule.items():
#     quench_labels = rmse_per_cryomodule[file_path]['quench']
#     rmse_value = rmse_per_cryomodule[file_path]['rmse']
#     classification_value = rmse_per_cryomodule[file_path]['classification']

#     cavity_list = waveforms['cavity']
#     forward_list = waveforms['forward']
#     reverse_list = waveforms['reverse']
#     decay_list = waveforms['decay']
#     time_list = waveforms['time']

#     for i in range(len(cavity_list)):
#         fig2, ax2 = plt.subplots(figsize=(8,5))
#         ax3 = ax2.twinx()
#         time = time_list[i]
#         cav, = ax2.plot(time, cavity_list[i], label='Cavity Amplitude (MV)', color='#377eb8')
#         fwd, = ax3.plot(time, forward_list[i], label='Forward Power (W²)', color='#4daf4a')
#         rev, = ax3.plot(time, reverse_list[i], label='Reverse Power (W²)', color='#e41a1c')
#         dec, = ax2.plot(time, decay_list[i], label='Decay Reference (MV)', color='#ff7f00', linestyle='--')
#         ax2.set_xlim(-0.03, 0.03)
#         ax2.set_ylim(cavity_list[i].min(), cavity_list[i].max() + 20)
#         ax3.set_ylim(reverse_list[i].min(), reverse_list[i].max())
#         ax2.set_title(f'Waveforms for {quench_labels[i]} in {os.path.basename(file_path)} \n(RMSE={rmse_value[i]}; R2={r2_values[i]}) and classification is {classification_value[i]}', fontsize=12)
#         ax2.set_xlabel('Time in Seconds')
#         ax2.set_ylabel('Amplitude in MV')
#         ax3.set_ylabel('Power in W²')
#         ax3.legend(
#                 handles=[cav, dec, fwd, rev],
#                 labels=["Cavity Amplitude (MV)", "Decay Reference (MV)", "Forward Power (W²)", "Reverse Power (W²)"],
#                 loc='upper right'
#             )
#         ax2.grid(True, alpha=0.5)
#         plt.tight_layout()
#         plt.show()


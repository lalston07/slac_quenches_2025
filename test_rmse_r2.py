import h5py
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import re
import os
LOADED_Q_CHANGE_FOR_QUENCH = 0.6

"""
This section of code calculates the RMSE values for each "real" quench to access accuracy of the classification.
The goal is to determine a threshold RMSE value above which the classification of a quench as "real" can be considered reliable.

Questions answered with this plot:
(1) How many quenches were falsely classified?
(2) How can we combine RMSE Value and R-Squared Score be used to improve classification?
(3) What is the RMSE Average Value per cavity in the accelerator?
"""

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

    # # plotting the fit over the raw cavity amplitude data
    # plt.figure(figsize=(8, 5))
    # plt.plot(time_data, fault_data, label='Raw Amplitude Data', marker='o')
    # plt.plot(time_data, fitted_amplitude, label='Linear Exponential Fit', linestyle='--')
    # plt.xlabel("Time in Seconds")
    # plt.ylabel("Amplitude")
    # plt.ylim(0, 15)
    # plt.title(f"Exponential Fit vs Raw Amplitude (RMSE = {rmse})")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    # plt.close()

    return rmse, r2

folder_path = "C:/Users/leila/Documents/Visual Studio/slac_quenches_2025/quench_data_per_cryomodule"
h5_files = [f for f in os.listdir(folder_path) if f.endswith('.h5')]

# Dictionary to store results
# Structure: { "CMxx": { "CAVx": avg_rmse, ... }, ... }
cm_cavity_rmse_dict = {}

for file in h5_files:
    file_path = os.path.join(folder_path, file)
    print(f"\nProcessing: {file_path} - {os.path.basename(file_path)}")
    cryo = os.path.basename(file_path).replace("quench_data_CM", "").split(".")[0]
    
    with h5py.File(file_path, 'r') as f:

        for cavity in f.keys():
            cavity_group = f[cavity]
            
            # initializing lists
            rmse_values = []
            r2_values = []
            cavity_list = []
            cls_list = []
            ts_cls_list = []

            for year in cavity_group.keys():
                year_group = cavity_group[year]

                for month in year_group.keys():
                    month_group = year_group[month]

                    for day in month_group.keys():
                        day_group = month_group[day]

                        for quench_timestamp in day_group.keys():
                            quench_group = day_group[quench_timestamp]

                            try:
                                # getting waveform data
                                time_data = quench_group['time_seconds'][:]
                                cavity_data = quench_group['cavity_amplitude_MV'][:]
                                q_value = quench_group.attrs.get("saved_q_value")

                                # saving metadata
                                classification = quench_group.attrs.get("quench_classification", None)

                                # calculating error metrics
                                rmse, r2 = validate_quench(cavity_data, time_data, saved_loaded_q=q_value, frequency=1300000000.0)
                                rmse_values.append(rmse)
                                r2_values.append(r2)
                                ts_cls_list.append(f"{year}-{month}-{day}-{quench_timestamp} {classification}")
                                cavity_list.append(cavity)
                            except Exception as e:
                                continue
            
            if rmse_values:
                avg_rmse = float(np.mean(rmse_values))

                if cryo not in cm_cavity_rmse_dict:
                    cm_cavity_rmse_dict[cryo] = {}
                cm_cavity_rmse_dict[cryo][cavity] = avg_rmse

            # # plotting per cavity after all quenches processed
            # if rmse_values and r2_values:
            #     fig, ax1 = plt.subplots(figsize=(16, 6))

            #     # RMSE line (left y-axis)
            #     ax1.set_xlabel("Quench Timestamp")
            #     ax1.set_ylabel("RMSE", color="tab:blue")
            #     ax1.plot(ts_cls_list, rmse_values, marker='o', color="tab:blue", label="RMSE")
            #     ax1.tick_params(axis='y', labelcolor="tab:blue")

            #     # Average RMSE line
            #     avg_rmse = sum(rmse_values) / len(rmse_values)
            #     ax1.axhline(avg_rmse, color="tab:blue", linestyle="--", alpha=0.6, label=f"Avg RMSE: {avg_rmse:.3f}")

            #     # R² line (right y-axis)
            #     ax2 = ax1.twinx()
            #     ax2.set_ylabel("R² Score", color="tab:orange")
            #     ax2.set_ylim(-2.5, 2.5)
            #     ax2.plot(ts_cls_list, r2_values, marker='x', color="tab:orange", label="R²")
            #     ax2.tick_params(axis='y', labelcolor="tab:orange")

            #     plt.title(f"RMSE Value and R-Squared Score for CM{cryo} {cavity} Quenches")
            #     fig.autofmt_xdate(rotation=90)

            #     # Combine legends
            #     lines1, labels1 = ax1.get_legend_handles_labels()
            #     lines2, labels2 = ax2.get_legend_handles_labels()
            #     ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

            #     plt.tight_layout()
            #     plt.show()

# # Print final dictionary
# print("\nAverage RMSE per cavity per cryomodule:")
# for cm, cavities in cm_cavity_rmse_dict.items():
#     for cav, avg_rmse in cavities.items():
#         print(f"CM{cm} - {cav}: {avg_rmse:.4f}")
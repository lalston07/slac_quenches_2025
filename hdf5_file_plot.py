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
filename = "quench_data_CM32_v2.h5"
day_path = "CAV1/2022/07/21"        # 20220721
timestamp_list = ["11:01:57", "11:08:55", "13:52:15", "18:52:42", "12:31:16", "20:05:15", "17:18:06", "18:24:29"]
with h5py.File(filename, 'r') as f:
    if day_path in f:
        day_group = f[day_path]
        print(f"\n\nFound day group: {day_path}")
        for timestamp in day_group:
            if timestamp in timestamp_list:
                quench_group = day_group[timestamp]
                print(f"\n\n    Timestamp: {timestamp}")

                # reading waveform data
                time_data = quench_group['time_seconds'][:]
                cavity_data = quench_group['cavity_amplitude_MV'][:]
                forward_power = quench_group['forward_power_W2'][:]
                reverse_power = quench_group['reverse_power_W2'][:]
                decay_ref = quench_group['decay_reference_MV'][:]

                np.set_printoptions(threshold=np.inf)
                print("Cavity Data:\n", cavity_data)
                print("Decay Reference:\n", decay_ref)

                # for i, data_point in enumerate(cavity_data):
                #     if np.isclose(data_point, 0, atol=1e-5):  # adjust tolerance if needed
                #         print(f"⚠️ Near-zero cavity amplitude at index {i}, value: {data_point}")

                # metadata = {data: quench_group.attrs[data] for data in quench_group.attrs}
                for metadata_key, metadata_value in quench_group.attrs.items():
                    print(f"{metadata_key}: {metadata_value}")
                """
                    Metadata: {'calculated_q_value': nan, 'cavity_number': '8', 'cryomodule': '32', 'faultname': 'ACCL:L3B:3280:CAV:FLTAWF', 
                    'filename': 'ACCL_L3B_3280_20241002_151905_QUENCH.txt', 'quench_classification': False, 'saved_q_value': 41681390.47526922, 
                    'timestamp': '2024-10-02_15:19:05.864747'}

                    Calculated Q-value is 'nan' because we divided by zero from the cavity_data in the np.polyfit() validation method
                    This quench was classified as fake but how? Looking at the waveform plot, it was a fake quench.
                        >> is_real = loaded_q < thresh_for_quench
                        >> is_real = NaN < thresh_for_quench always returns boolean False
                    How can we confirm that this quench was fake if the loaded_q value could not be calculated?
                """
            
                saved_loaded_q = quench_group.attrs['saved_q_value']
                
                try:
                    classification, rmse, r2, time_0, end_decay = validate_quench(cavity_data, time_data, saved_loaded_q= saved_loaded_q, frequency=1300000000.0)
                    print(f"Time Zero: {time_0}, End Decay: {end_decay}")
                    print(f"RMSE Value is {rmse}, and R^2 Score is {r2}")
                except IndexError as e:
                    print(f"Processing {filename} failed with {e}")

                # plotting waveform data for each group/quench file
                plt.figure(figsize=(14,6))
                plt.plot(time_data, cavity_data, label='Cavity Data (MV)', linewidth=3)
                plt.plot(time_data, forward_power, label='Forward Power (W^2)', linewidth=2)
                plt.plot(time_data, reverse_power, label='Reverse Power (W^2)', linewidth=2)
                plt.xlim(-0.02, 0.02)
                plt.plot(time_data, decay_ref, label='Decay Reference (MV)', linestyle='--', linewidth=1.5)
                plt.xlabel('Time in Seconds')
                plt.ylabel('Amplitude')
                plt.title(f'Cavity Quench Waveform\n{filename}')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()




# # extracting data and calculating RMSE values for each quench in a cryomodule
# filename = "quench_data_CM30.h5"
# with h5py.File(filename, 'r') as f:



# # extracting data from the H5 file
# filename = "test_data_CM31.h5"
# month_path = "CAV1/2025/07"

# with h5py.File(filename, 'r') as f:
#     if month_path in f:
#         month_group = f[month_path]
#         print(f"Found month group: {month_path}")
#         days = list(month_group.keys())
#         print(f"Days in {month_path}: {days}")
#         for day in days:
#             day_group = month_group[day]
#             for timestamp in day_group: 
#                 quench_group = day_group[timestamp]
#                 print(f"    Timestamp: {timestamp}")

#                 # reading waveform data
#                 time_data = quench_group['time_seconds'][:]
#                 cavity_data = quench_group['cavity_amplitude_MV'][:]
#                 forward_power = quench_group['forward_power_W2'][:]
#                 reverse_power = quench_group['reverse_power_W2'][:]
#                 decay_ref = quench_group['decay_reference_MV'][:]
                
#                 # retrieving the metadata
#                 metadata = {data: quench_group.attrs[data] for data in quench_group.attrs}
                
#                 # printing the information
#                 print(f"    Time Data: {time_data}")
#                 print(f"    Cavity Data: {cavity_data}")
#                 print(f"    Forward Power Data: {forward_power}")
#                 print(f"    Reverse Power Data: {reverse_power}")
#                 print(f"    Decay Reference Data: {decay_ref}")
#                 print(f"    Metadata: {metadata}")

#                 # plotting the waveform data
#                 fig, ax = plt.plot(figsize=(12,6))
#                 ax.set_title(f"Plot of Quench Wavform Data - {month_group}_{day_group}_{timestamp}")
#                 ax.set_xlabel("Time in Seconds")
#                 ax.set_ylabel("Amplitude")
#                 ax.plot(time_data, cavity_data, label='Cavity (MV)', color='blue')
#                 ax.plot(time_data, forward_power, label='Forward Power (W2)')
#     else: 
#         print(f"Path '{month_path}' not found.")

# with h5py.File('cavity_quench_data_google_drive_v2.h5', 'r') as f:
#     for group_name in f.keys():     # looping through all groups
#         group = f[group_name]       # this is the current file/group
        
#         time_data = group['time_seconds'][()]  # reads entire dataset into a numpy array
#         cavity_data = group['cavity_amplitude_MV'][()]
#         forward_data = group['forward_power_W2'][()]
#         reverse_data = group['reverse_power_W2'][()]
#         decay_data = group['decay_reference_MV'][()]
        
#         filename = group.attrs['filename'] # getting filename from group attribute
#         timestamp = group.attrs['timestamp']

#         # plotting waveform data for each group/quench file
#         plt.figure(figsize=(14,6))
#         plt.plot(time_data, cavity_data, label='Cavity Data (MV)', linewidth=3)
#         plt.plot(time_data, forward_data, label='Forward Power (W^2)', linewidth=2)
#         plt.plot(time_data, reverse_data, label='Reverse Power (W^2)', linewidth=2)
#         plt.xlim(-0.02, 0.02)
#         plt.plot(time_data, decay_data, label='Decay Reference (MV)', linestyle='--', linewidth=1.5)
#         plt.xlabel('Time in Seconds')
#         plt.ylabel('Amplitude')
#         plt.title(f'Cavity Quench Waveform\n{filename}')
#         plt.legend()
#         plt.grid(True)
#         plt.tight_layout()
#         plt.show()
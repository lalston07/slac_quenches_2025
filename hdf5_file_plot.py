import h5py
import numpy as np
import matplotlib.pyplot as plt

# extracting data from the H5 file
filename = "test_data_CM31.h5"
month_path = "CAV1/2025/07"

with h5py.File(filename, 'r') as f:
    if month_path in f:
        month_group = f[month_path]
        print(f"Found month group: {month_path}")
        days = list(month_group.keys())
        print(f"Days in {month_path}: {days}")
        for day in days:
            day_group = month_group[day]
            for timestamp in day_group: 
                quench_group = day_group[timestamp]
                print(f"    Timestamp: {timestamp}")

                # reading waveform data
                time_data = quench_group['time_seconds'][:]
                cavity_data = quench_group['cavity_amplitude_MV'][:]
                forward_power = quench_group['forward_power_W2'][:]
                reverse_power = quench_group['reverse_power_W2'][:]
                decay_ref = quench_group['decay_reference_MV'][:]
                
                # retrieving the metadata
                metadata = {data: quench_group.attrs[data] for data in quench_group.attrs}
                
                # printing the information
                print(f"    Time Data: {time_data}")
                print(f"    Cavity Data: {cavity_data}")
                print(f"    Forward Power Data: {forward_power}")
                print(f"    Reverse Power Data: {reverse_power}")
                print(f"    Decay Reference Data: {decay_ref}")
                print(f"    Metadata: {metadata}")

                # plotting the waveform data
                fig, ax = plt.plot(figsize=(12,6))
                ax.set_title(f"Plot of Quench Wavform Data - {month_group}_{day_group}_{timestamp}")
                ax.set_xlabel("Time in Seconds")
                ax.set_ylabel("Amplitude")
                ax.plot(time_data, cavity_data, label='Cavity (MV)', color='blue')
                ax.plot(time_data, forward_power, label='Forward Power (W2)')
    else: 
        print(f"Path '{month_path}' not found.")

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
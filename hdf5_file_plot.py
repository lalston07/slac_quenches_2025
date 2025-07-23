import h5py
import numpy as np
import matplotlib.pyplot as plt

with h5py.File('cavity_quench_data_google_drive.h5', 'r') as f:
    for group_name in f.keys():            # looping through all groups
        group = f[group_name]              # this is the current file/group
        
        time_data = group['time_seconds'][()]  # reads entire dataset into a numpy array
        cavity_data = group['cavity_amplitude_MV'][()]
        forward_data = group['forward_power_W2'][()]
        reverse_data = group['reverse_power_W2'][()]
        decay_data = group['decay_reference_MV'][()]
        
        filename = group.attrs['filename'] # getting filename from group attribute
        timestamp = group.attrs['timestamp']

        # plotting waveform data for each group/quench file
        plt.figure(figsize=(14,6))
        plt.plot(time_data, cavity_data, label='Cavity Data (MV)', linewidth=3)
        plt.plot(time_data, forward_data, label='Forward Power (W^2)', linewidth=2)
        plt.plot(time_data, reverse_data, label='Reverse Power (W^2)', linewidth=2)
        plt.plot(time_data, decay_data, label='Decay Reference (MV)', linestyle='--', linewidth=1.5)
        plt.xlabel('Time in Seconds')
        plt.ylabel('Amplitude')
        plt.title(f'Cavity Quench Waveform\n{filename}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

filename = 'ACCL_L3B_3180_20240509_144613_QUENCH.txt'
timestamp = '2024-05-09_14:46:13.516921'

# PV labels
cavity_faultname = 'ACCL:L3B:3180:CAV:FLTAWF'
forward_pow = 'ACCL:L3B:3180:FWD:FLTAWF'
reverse_pow = 'ACCL:L3B:3180:REV:FLTAWF'
decay_ref = 'ACCL:L3B:3180:DECAYREFWF'
time_range = 'ACCL:L3B:3180:CAV:FLTTWF'

def extracting_data(path_name, faultname):
    with open(path_name, 'r') as file:
        for line in file:
            if faultname in line and f"{faultname}." not in line:
                data = pd.Series(line.split())
                target_timestamp = line.split()[1]
                values = data[2:].astype(float).values  # extracts everything after timestamp
                
                print(f"{faultname} Information:")
                print(f"Length of data: {len(values)}")
                print(f"First value: {values[0]}, Last value: {values[-1]}")
                print(f"Min value: {np.min(values)}, Max value: {np.max(values)}\n")

    return values, target_timestamp

# extracting all data for quench waveform
cavity_data, cavity_time = extracting_data(filename, cavity_faultname)
forward_data, forward_time = extracting_data(filename, forward_pow)
reverse_data, reverse_time = extracting_data(filename, reverse_pow)
decay_data, decay_time = extracting_data(filename, decay_ref)
time_data, time_timestamp = extracting_data(filename, time_range)

# making them all the same length for ACCL_L3B_3180_20240509_144613_QUENCH.txt file
forward_data = forward_data[:len(cavity_data)]

# plotting data together
plt.figure(figsize=(14,6))
plt.plot(time_data, cavity_data, label="Cavity", color='blue', linewidth=3)
plt.plot(time_data, forward_data, label="Forward Power", color='green', linewidth=3)
plt.plot(time_data, reverse_data, label="Reverse Power", color='red', linewidth=3)
plt.plot(time_data, decay_data, label="Decay Reference", color='cyan', linewidth=1)
plt.xlabel("Sample Index")
plt.ylabel("Amplitude (MV)")
plt.title(f"Cavity Quench Waveform for {cavity_faultname} {cavity_time}")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

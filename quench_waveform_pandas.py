import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

filename = 'ACCL_L3B_3180_20220630_154720_QUENCH.txt'
timestamp = '2022-06-30_15:47:20.802694'

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

# making them all the same length in case the length varies
forward_data = forward_data[:len(cavity_data)]

# plotting data together on one y=-axis
# plt.figure(figsize=(14,6))
# plt.plot(time_data, cavity_data, label="Cavity", color='blue', linewidth=3)
# plt.plot(time_data, forward_data, label="Forward Power", color='green', linewidth=3)
# plt.plot(time_data, reverse_data, label="Reverse Power", color='red', linewidth=3)
# plt.plot(time_data, decay_data, label="Decay Reference", color='cyan', linewidth=1)
# plt.xlabel("Time in Seconds")
# plt.ylabel("Amplitude in MV")
# plt.title(f"Cavity Quench Waveform for {cavity_faultname} {cavity_time}")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

# plotting the data on two separate y-axis for MV and sqrt(W)
fig, ax1 = plt.subplots()
ax1.set_xlabel('Time in Seconds')   # First y-axis (MV)
ax1.set_ylabel('Amplitude in MV', color='black')
ax1.plot(time_data, cavity_data, label="Cavity (MV)", color='blue', linewidth=3)
ax1.plot(time_data, decay_data, label="Decay Reference", color='cyan', linewidth=1, linestyle='--')
ax1.tick_params(axis='y', labelcolor='black')
ax1.set_ylim(-1, 23)
# ax1.grid(True, linewidth=0.5)
ax2 = ax1.twinx()   # Second y-axis (W^2)
ax2.set_ylabel('Amplitude in W^2', color='black')
ax2.plot(time_data, forward_data, label="Forward Power (W^2)", color='red', linewidth=2)
ax2.plot(time_data, reverse_data, label="Reverse Power(W^2)", color="green", linewidth=2)
ax2.tick_params(axis='y', labelcolor='black')
ax2.set_ylim(-1, 65)
fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))    # Combining the legends for both y-axes
fig.suptitle(f"Cavity Quench Waveform for {cavity_faultname} {cavity_time}", fontsize=14)
fig.tight_layout()
plt.show()



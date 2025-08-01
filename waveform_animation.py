import h5py
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import re
import os
from matplotlib.animation import FuncAnimation

# filename: "quench_data_CM31.h5"
# real quench file: CAV8/2022/07/01/12:16:28
# fake quench file: CAV8/2022/06/30/16:49:05

folder_path = "C:/Users/leila/Documents/Visual Studio/slac_quenches_2025/quench_data_per_cryomodule"
h5_files = ['quench_data_CM31.h5']

real_quench_path = "CAV8/2022/07/01/12:16:28"
fake_quench_path = "CAV8/2022/06/30/16:49:05"

filename = "quench_data_CM31.h5"
file_path = os.path.join(folder_path, filename)

with h5py.File(file_path, 'r') as f:
    quench_paths = ["CAV8/2022/07/01/12:16:28", "CAV8/2022/06/30/16:49:05"]
    # real_quench_path = "CAV8/2022/07/01/12:16:28"
    # fake_quench_path = "CAV8/2022/06/30/16:49:05"

    for path in quench_paths:
        if path in f:
            quench_group = f[real_quench_path]
            # print(f"Datasets in {path}: {list(quench_group.keys())}")

            # getting the waveform data
            time_data = quench_group['time_seconds'][:]             # for the x-axis
            cavity_data = quench_group['cavity_amplitude_MV'][:]    # y-axis on left (amp)
            forward_data = quench_group['forward_power_W2'][:]      # y-axis on right (power)
            reverse_data = quench_group['reverse_power_W2'][:]      # y-axis on right (power)
            decay_data = quench_group['decay_reference_MV'][:]      # y-axis on left (amp)
            
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()

            ax1.set_xlim(-0.05, 0.05)
            ax1.set_ylim(cavity_data.min() * 1.1, (cavity_data.max() * 1.1) + 23)
            ax2.set_ylim(forward_data.min() * 1.1, (forward_data.max() * 1.1) + 15)
            ax1.set_xlabel("Time in Seconds")
            ax1.set_ylabel("Amplitude in MV")
            ax2.set_ylabel("Power in W²")
            ax1.set_title(f"{filename}: {path}")

            line_cav, = ax1.plot([], [], label="Cacvity (MV)", color='blue', linewidth=3)
            lines = [line_cav]

            line_dec, = ax1.plot([], [], label = "Decay Reference (MV)", color='orange', linestyle="--", linewidth=1)
            lines.append(line_dec)

            line_fwd, = ax2.plot([], [], label="Forward Power (W²)", color='green', linewidth=3)
            lines.append(line_fwd)

            line_rev, = ax2.plot([], [], label="Reverse Power (W²)", color='red', linewidth=3)
            lines.append(line_rev)

            ax1.legend(
                handles=[line_cav, line_dec, line_fwd, line_rev],
                labels=["Cavity (MV)", "Decay Reference (MV)", "Forward Power (W²)", "Reverse Power (W²)"],
                loc='upper right'
            )

            def init():
                for line in lines:
                    line.set_data([], [])
                return lines
            
            def update(frame):
                line_cav.set_data(time_data[:frame], cavity_data[:frame])
                line_dec.set_data(time_data[:frame], decay_data[:frame])
                line_fwd.set_data(time_data[:frame], forward_data[:frame])
                line_rev.set_data(time_data[:frame], reverse_data[:frame])
                return lines
            
            visible_start = np.argmax(time_data >= -0.05)

            ani = FuncAnimation(
                fig, update, frames=range(visible_start, len(time_data)), init_func=init,
                blit=True, interval=10
            )

            plt.tight_layout()
            plt.show()
        else:
            print(f"Path not found: {quench_paths}")

    

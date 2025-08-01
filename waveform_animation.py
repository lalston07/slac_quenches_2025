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
            quench_group = f[path]
            # print(f"Datasets in {path}: {list(quench_group.keys())}")

            # getting the waveform data
            time_data = quench_group['time_seconds'][:]             # for the x-axis
            cavity_data = quench_group['cavity_amplitude_MV'][:]    # y-axis on left (amp)
            forward_data = quench_group['forward_power_W2'][:]      # y-axis on right (power)
            reverse_data = quench_group['reverse_power_W2'][:]      # y-axis on right (power)
            decay_data = quench_group['decay_reference_MV'][:]      # y-axis on left (amp)
            
            # setting up dual y-axes
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()

            # axis formatting 
            ax1.set_xlim(-0.05, 0.05)
            ax1.set_ylim(cavity_data.min(), cavity_data.max() + 20)
            ax2.set_ylim(reverse_data.min(), reverse_data.max() + 10)
            ax1.set_xlabel("Time in Seconds")
            ax1.set_ylabel("Amplitude in MV")
            ax2.set_ylabel("Power in W²")
            ax1.set_title(f"{filename}: {path}")

            # creating line objects for each waveform
            line_cav, = ax1.plot([], [], label="Cacvity (MV)", color='blue', linewidth=4)   # plot([], []) created an empty line object which is updated frame by frame
            lines = [line_cav]

            line_dec, = ax1.plot([], [], label = "Decay Reference (MV)", color='orange', linestyle="--", linewidth=1)
            lines.append(line_dec)

            line_fwd, = ax2.plot([], [], label="Forward Power (W²)", color='green', linewidth=3)
            lines.append(line_fwd)

            line_rev, = ax2.plot([], [], label="Reverse Power (W²)", color='red', linewidth=3)
            lines.append(line_rev)

            # combining the legend
            ax1.legend(
                handles=[line_cav, line_dec, line_fwd, line_rev],
                labels=["Cavity (MV)", "Decay Reference (MV)", "Forward Power (W²)", "Reverse Power (W²)"],
                loc='upper right'
            )

            # init is used to clear all lines
            # ensures that the plot starts with no lines before the animation begins
            def init():
                for line in lines:
                    line.set_data([], [])
                return lines
            
            # update is used to show more data as time goes on (each frame draws a longer section of the time series)
            def update(frame):
                line_cav.set_data(time_data[:frame], cavity_data[:frame])
                line_dec.set_data(time_data[:frame], decay_data[:frame])
                line_fwd.set_data(time_data[:frame], forward_data[:frame])
                line_rev.set_data(time_data[:frame], reverse_data[:frame])
                return lines    # lines = [line_cav, line_dec, line_fwd, line_rev]
            
            # making it so that time starts at the xlimit (no delay in animation)
            # np.argmax() returns the index of when time_data >= -0.05 so we can find xlimit starting point
            visible_start = np.argmax(time_data >= -0.05)
            visible_end = np.argmax(time_data >= 0.05)

            # this function runs update(frame) for each frame
            # matplotlib.animation.FunctionAnimation(fig, func, frames=None, init_func=None, blit=True)
            # frames=range(...) makes it so that we plot the time points after time=-0.05 until the waveform is finished
            # blit=True ensures that only the parts that change are redrawn and not the whole figure
            ani = FuncAnimation(
                fig, update, frames=range(visible_start, visible_end), init_func=init,
                blit=True, interval=10
            )

            plt.tight_layout()
            plt.show()
        else:
            print(f"Path not found: {quench_paths}")

    

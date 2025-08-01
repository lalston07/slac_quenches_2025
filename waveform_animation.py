import h5py
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import re
import os

# filename: "quench_data_CM31.h5"
# real quench file: CAV8/2022/07/01/12:16:28
# fake quench file: CAV8/2022/06/30/16:49:05

folder_path = "C:/Users/leila/Documents/Visual Studio/slac_quenches_2025/quench_data_per_cryomodule"
h5_files = ['quench_data_CM31.h5']

real_quench_path = "CAV8/2022/07/01/12:16:28"
fake_quench_path = "CAV8/2022/06/30/16:49:05"

for file in h5_files:
    file_path = os.path.join(folder_path, file)
    cryo_label = file.replace("quench_data_", "").replace(".h5", "")

    with h5py.File(file_path, 'r') as f:
        for label, path in [("real", real_quench_path), ("fake", fake_quench_path)]:
            if path in f:
                cavity_group = f[path]
                print[cavity_group]

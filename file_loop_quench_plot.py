import matplotlib.pyplot as plt
import numpy as np
import glob
import re
from datetime import datetime
import pandas as pd
import h5py
from matplotlib.backends.backend_pdf import PdfPages

# to extract data using a directory:
directory_path = r"G:\My Drive\ACCL_L3B_3180"

# print all of the file names with "_QUENCH" in the folder using a loop (using glob module)
results = glob.glob(directory_path + '/**/*QUENCH.txt', recursive=True) # force this to have a number before _QUENCH
print(f"Found {len(results)} matching QUENCH text files:")
# print(results) # results is a list
quench_files = [f for f in results if re.search(r"\d+_QUENCH", f)]
print(f"Found {len(quench_files)} matching '##_QUENCH' text files:")
print(quench_files)

# creating a function to extract the waveform data and timestamps from each file
def extracting_data(path_name, faultname): 
    with open(path_name, 'r') as file:
        for line in file: 
            if f"{faultname}" in line and f"{faultname}." not in line: 
                data = pd.Series(line.split())
                target_timestamp = line.split()[1]  # searching for timestamp in case it varies
                values = data[2:].astype(float).values

                print(f"{faultname} Information:")
                print(f"Length of data: {len(values)}")
                print(f"First value: {values[0]}, Last value: {values[-1]}")
                print(f"Min value: {np.min(values)}, Max value: {np.max(values)}\n")

                return values, target_timestamp
    return None, None   # added in case the PV line is not found

# saving data to an HDF5 file - cavity amplitude, forward power, reverse power, decay reference, faultname, timestamp, filename, cryomodule, and cavity number
output_filename = 'cavity_quench_data_google_drive.h5'
data_long=[]

# this block of code is for saving the quench plots to a single pdf
pdf_filename = 'all_quench_plots_google_drive.pdf'
pdf = PdfPages(pdf_filename)
# plotting the data together
for i, file in enumerate(quench_files):
    print("\nProcessing file: " + file)
        
    # getting PV and timestamp information from {file}
    # line below splits the file in to 4 parts (after the '\') and gets the last part (filename)
    filename = file.split("\\", 4)[-1].replace('.txt','') 
    parts = filename.split('_') # splits the filename into parts at each '_'
    pv_base = parts[0] + ":" + parts[1] + ":" + parts[2]    # ex: pt(0): ACCL, pt(1): L3B, pt(2):3180
    timestamp_raw = parts[3] + "_" + parts[4]               # ex: pt(3): 20221028, pt(4): 235218
    # line below formats the timestamp to match the file layout
    timestamp = datetime.strptime(timestamp_raw, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d_%H:%M:%S.")
    print("PV label: " + pv_base)
    print("Timestamp: " + timestamp + "\n")

    # constructing PV label strings
    cavity_faultname = pv_base + ':CAV:FLTAWF'  # ex: ACCL:L3B:3180:CAV:FLTAWF
    forward_pow = pv_base + ':FWD:FLTAWF'       # ex: ACCL:L3B:3180:FWD:FLTAWF
    reverse_pow = pv_base + ':REV:FLTAWF'       # ex: ACCL:L3B:3180:REV:FLTAWF
    decay_ref = pv_base + ':DECAYREFWF'         # ex: ACCL:L3B:3180:DECAYREFWF    
    time_range = pv_base + ':CAV:FLTTWF'         # ex: ACCL:L3B:3180:CAV:FLTTWF

    # extracting all data for quench waveform using defined function
    cavity_data, cavity_time = extracting_data(file, cavity_faultname)
    forward_data, forward_time = extracting_data(file, forward_pow)
    reverse_data, reverse_time = extracting_data(file, reverse_pow)
    decay_data, decay_time = extracting_data(file, decay_ref)
    time_data, time_timestamp = extracting_data(file, time_range)

    # making them all the same length in case the length varies
    if len(forward_data) > len(cavity_data):
        data_long.append(f"{filename}.txt")
    forward_data = forward_data[:len(cavity_data)]

    # plotting data on the same axis
    fig = plt.figure(figsize=(14,6))
    plt.plot(time_data, cavity_data, label="Cavity (MV)", color='blue', linewidth=4)
    plt.plot(time_data, forward_data, label="Forward Power (W^2)", color='green', linewidth=3)
    plt.plot(time_data, reverse_data, label="Reverse Power (W^2)", color='red', linewidth=3)
    plt.plot(time_data, decay_data, label="Decay Reference (MV)", color='cyan', linewidth=1, linestyle='--')
    plt.xlabel("Time in Seconds")
    plt.ylabel("Amplitude")
    plt.title(f"Cavity Quench Waveform\n{filename}.txt")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
pdf.close()
print(f"All quench plots saved to {pdf_filename}.")


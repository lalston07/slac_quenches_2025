import matplotlib.pyplot as plt
import numpy as np
import io

filename = 'ACCL_L3B_3180_20220630_164905_QUENCH.txt'   # imput data file
timestamp = '2022-06-30_16:49:05.440831'                # waveform timestamp
time_timestamp = '2022-06-30_15:46:04.712966'            # time data has its own timestamp

# PV or fault string to search for and precise timestamp of the waveform
cavity_faultname = 'ACCL:L3B:3180:CAV:FLTAWF'    # cavity details for amplitude
forward_pow = 'ACCL:L3B:3180:FWD:FLTAWF'         # forward power details for amplitude
reverse_pow = 'ACCL:L3B:3180:REV:FLTAWF'         # reverse power details for amplitude
decay_ref = 'ACCL:L3B:3180:DECAYREFWF'           # decay reference details for amplitude
time_range = 'ACCL:L3B:3180:CAV:FLTTWF'          # time data in seconds for amplitude

# to extract data using a directory:
# directory = r"G:\My Drive\ACCL_L3B_3180\ACCL_L3B_3180_20220630_164905_QUENCH"
# filename = 'ACCL_L3B_3180_20220630_164905_QUENCH.txt'
# full_path = directory + filename
# if directory.exists():
#   print("Google Drive directory found.")
# else: 
#   print("Directory not found.")

# extracting data directly from the file
def extracting_data(path_name, faultname, timestamp):
    section_lines = []
    with open(path_name, 'r') as file:
        for line in file:
            if f"{faultname} {timestamp}" in line:
                section_lines.append(line) # if the target string matches then that entire line is added to section_lines

    if not section_lines:
        print("No matching lines found.")
    else:
        # convert list of lines to file-like object
        # joins section_lines into a single string 
        # wraps it with io.StringIO so it behaves like a file to be used with np.loadtxt
        section_data = io.StringIO("".join(section_lines))

        # figuring out how many columns there are
        # looks at the first matching line in section_lines and strips leading or trailing whitespace
        # splits the line into columns by whitespace
        # line.strip().split() >> ['ACCL:L3B:3180:CAV:FLTAWF', '2022-06-30_16:49:05.440831', '15.021'...]
        num_cols = len(section_lines[0].strip().split())
        
        # builds a list of column indexes starting from index 2 up to but not including num_cols
        usecolumns = list(range(2, num_cols))

        # loading the data
        data_array = np.loadtxt(section_data, delimiter=None, usecols=usecolumns) # col(0) is PV and col(1) is timestamp
        print(f"{faultname} Information:")
        print(f"Data points are: {data_array}") # prints range of data
        print(f"Length of data is: {len(data_array)}") # prints number of data points
        print(f"First value is {data_array[0]}, and last value is {data_array[len(data_array)-1]}")
        print(f"Min value is {np.min(data_array)}, and max value is {np.max(data_array)}\n")

        return data_array # returns the extracted data

# extract each waveform using defined function
cavity_data = extracting_data(filename, cavity_faultname, timestamp)
forward_data = extracting_data(filename, forward_pow, timestamp)
reverse_data = extracting_data(filename, reverse_pow, timestamp)
decay_data = extracting_data(filename, decay_ref, timestamp)
time_data = extracting_data(filename, time_range, time_timestamp)

# plot setup
plt.figure(figsize=(14,6))
plt.plot(time_data, cavity_data, label = "Cavity", color = 'blue', linewidth=3)
plt.plot(time_data, forward_data, label = "Forward Power", color = 'green', linewidth=3)
plt.plot(time_data, reverse_data, label = "Reverse Power", color = 'red', linewidth=3)
plt.scatter(time_data, decay_data, label = "Normal Cavity Decay Reference", color = 'cyan', s=1, marker='o', linewidth=5)

# plot formatting
plt.xlabel('Time in Seconds', fontsize='large') # or fontsize=14 for large font
plt.ylabel('MV', fontsize='large')
plt.title(f'Quench Waveforms - {cavity_faultname} {timestamp}')
plt.legend(fontsize='large') 
plt.grid(True)
plt.tight_layout()

# save the plot to file
# plot_filename = f"test_v3_{filename.replace('.txt',"")}.png"
# plt.savefig(plot_filename)
# print(f"Plot saved as: {plot_filename}")

# show plot 
plt.show()


# NUMPY LOAD TXT NOTES
# numpy.loadtxt(fname, dtype=<class 'float', comments='#', delimeter=None, converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0, encoding=None, max_rows=None, *, quotechar=None, like=None)
# fname - filename, list, or generator to read
# dtype (optional) - data type of resulting array
# comments (optional) - characters or list of characters used to indicate start of a comment
# delimiter (optional) - character used to separate values
# skiprows (optional) - skip the first 'skiprows' lines including comments
# usecols (optional) - which columns to read with 0 being first
# max_rows (optional) - reads 'max_rows' of content after 'skiprows' lines

# NUMPY FROMSTRING NOTES
# numpy.fromstring(string, dtype=float, count=-1, *, sep, like=None)
# str - string containing data
# dtype (optional) - data type of the array
# count (optional) - reads this number of dtype elements from the data
# sep (optional) - string separating numbers in the data
# like (optional) - reference object to allow creation of arrays which aren't numpy arrays


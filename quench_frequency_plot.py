import matplotlib.pyplot as plt
import numpy as np
import glob
import re
import pandas as pd
from datetime import datetime

# step one - loop through files and extract timestamp data for the x-axis
directory_path = r"G:\My Drive\ACCL_L3B_3180"
results = glob.glob(directory_path + '/**/*QUENCH.txt', recursive=True) # force this to have a number before _QUENCH
# print(f"Found {len(results)} matching QUENCH text files:")
# print(results) 
quench_files = [f for f in results if re.search(r"\d+_QUENCH", f)]
print(f"Found {len(quench_files)} matching '##_QUENCH' text files:")
# print(quench_files)
timestamps = []
for file in quench_files:
    # line below splits the file in to 4 parts (after the '\') and gets the last part (filename)
    filename = file.split("\\", 4)[-1].replace('.txt','') 
    parts = filename.split('_') # splits the filename into parts at each '_'
    CM_name = parts[2][:2]
    CAV_name = parts[2][-2]
    timestamp_raw = parts[3]   # ex: pt(3): 20221028, pt(4): 235218
    # line below formats the timestamp to match the file layout
    timestamp = datetime.strptime(timestamp_raw, "%Y%m%d")
    timestamps.append(timestamp)

# step two - converting list to datetime or series object
# timestamp_index = pd.to_datetime(timestamps)  # pandas datetime object
timestamp_index = pd.Series(timestamps)         # pandas series object

# step three - count the number of quenches per timestamp for the y-axis
# .dt - accesses the datetime properties
# .to_period("M") - converts timestamp to year and month and groups timestamps from same month together
# .value_counts() - counts number of times each value appears 
# .sort_index() - sorts results in chronological order
quench_frequency = timestamp_index.dt.to_period("M").value_counts().sort_index()        

# step four - create full date range including skipped months
full_range = pd.period_range(start=timestamp_index.min(), end=timestamp_index.max(), freq='M')
quench_frequency = quench_frequency.reindex(full_range, fill_value=0)
print(quench_frequency)

# step five - plot the quench frequency data as bar chart
# x-axis is months and y-axis is number of quench files per month
# plt.figure(figsize=(12,6))
# ax = quench_frequency.plot(kind='bar', color='blue', label = f"CM: {CM_name}, CAV: {CAV_name}") 
# ax.set_xlabel("Month", fontsize=14)
# ax.set_ylabel("Number of Quenches", fontsize=14)
# ax.set_title("Quench Frequency Over Time", fontsize=14)
# ax.grid(True)  
# ax.set_axisbelow(True) # makes grid go to back
# ax.legend() # adds legend for cryomodle and cavity number
# plt.tight_layout()
# plt.show()

# step six - plot the quench frequency data as scatter plot
plt.figure(figsize=(12,6))
x = quench_frequency.index.astype(str)
    # quench_frequency.index gives the index of the Series (monthly)
    # .astype(str) converts the period object into a string
y = quench_frequency.values
    # quench_frequency_values extracts the data vaules from the series (counts of quenches per month)
plt.scatter(x, y, color = 'blue', label = f"CM: {CM_name}, CAV: {CAV_name}")

# for file in []:
#     if real_quench in file:
#         plt.scatter(x, y, color = 'blue', label = f"Real Quenches")
#     else:
#         plt.scatter(x, y, color = 'red', label = f"Fake Quenches")

plt.xlabel("Month", fontsize=14)
plt.ylabel("Number of Quenches", fontsize=14)
plt.title("Quench Frequency Over Time", fontsize=14)

# plt.title(f"Quench Frequency Over Time in CM: {CM_name}, CAV: {CAV_name}", fontsize=14)

plt.xticks(rotation=90)
plt.grid(True, linewidth=0.5)
plt.gca().set_axisbelow(True)
plt.legend()
plt.tight_layout()
plt.show()

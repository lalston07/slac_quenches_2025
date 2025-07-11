import numpy as np
from multiprocessing import Process
import sim_nrd_stel
import importlib
importlib.reload(sim_nrd_stel)
# from sim_nrd_stel import poincare_section_fieldline
from sim_nrd_stel_multiprocessing import poincare_section_fieldline
import time

# # Define the parameters you want to run with
# # radii_res1 = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.80, 0.85, 0.86, 0.87]
# radii_res1 = np.linspace(0.80, 0.82, 3)
# # radii_res1 = []
# # parameters = [0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.00, 1.01]
# radii_res2 = np.linspace(0.83, 0.85, 3)
# # radii_res2 = [0.8831]
# #radii_res2 = []
# radii_res3 = np.linspace(0.86, 0.88, 21)
# radii_res4 = []
# radii = np.append(np.append(radii_res1, radii_res2), np.append(radii_res3, radii_res4))

radii_res1 = []
radii_res2 = np.linspace(0.86, 0.90, 3)
radii_res3 = []
radii_res4 = []
radii = np.append(np.append(radii_res1, radii_res2), np.append(radii_res3, radii_res4))

#next compute the res1, radius = 0.05 to 0.85

map_iterations = 1000
# Function to run tasks in parallel
def run_in_parallel():
    processes = []

    for radius in radii:
        print(f"Starting process for radius: {radius}")
        # Create a process for each set of parameters
        p = Process(target=poincare_section_fieldline, args=(radius,map_iterations))
        processes.append(p)
        p.start()

    # Optionally, wait for all processes to complete
    for p in processes:
        p.join()


# Run the tasks
if __name__ == "__main__":
    t0 = time.time()
    
    run_in_parallel()

    t1 = time.time()
    elapsed = t1-t0
    print(f'Time taken: {elapsed:.6f} seconds')
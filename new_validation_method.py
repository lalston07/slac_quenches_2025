import h5py
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import re
import os
from avg_rmse_per_cav import cm_cavity_rmse_dict    # dictionary with average RMSE values per cavity
LOADED_Q_CHANGE_FOR_QUENCH = 0.6

def updated_validate_quench(fault_data, time_data, saved_loaded_q, frequency, avg_RMSE):
    time_0 = 0
    for time_0, timestamp in enumerate(time_data):
        if timestamp >= 0:
            break
    
    fault_data = fault_data[time_0:]
    time_data = time_data[time_0:]

    end_decay = len(fault_data) - 1
    for end_decay, amp in enumerate(fault_data):
        if amp < 0.002:
            break
    
    fault_data = fault_data[:end_decay]
    time_data = time_data[:end_decay]

    pre_quench_amp = fault_data[0]

    exponential_term = np.polyfit(time_data, np.log(pre_quench_amp / fault_data), 1)[0]
    loaded_q = (np.pi * frequency) / exponential_term

    thresh_for_quench = LOADED_Q_CHANGE_FOR_QUENCH * saved_loaded_q

    is_real = loaded_q < thresh_for_quench

    """
    CALCULATION OF FITTING ERROR
    Calculates how well the exponential fit A(t) = A0 * exp(-pi*f*t / Q) matches the waveform data.
    Returns error metrics like MSE, RMSE, and residuals.
    """

    # fitted_amplitude is created using the dacay rate (exponential term) that we got from np.polyfit
    fitted_amplitude = pre_quench_amp * np.exp(-exponential_term * time_data)

    # error metrics using numpy
    rmse = np.sqrt(np.mean((fault_data - fitted_amplitude)**2))
    r2 = 1 - (np.sum((fault_data - fitted_amplitude)**2) / np.sum((fault_data - np.mean(fault_data))**2))

    # correcting the classification only in this case
    if rmse > avg_RMSE and -20 < r2 < 0:
            new_classification = False
    elif rmse > avg_RMSE and r2 <= -20:
        new_classification = True
    else:
        new_classification = is_real

    return new_classification, rmse, r2
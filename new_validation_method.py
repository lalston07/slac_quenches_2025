import h5py
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import re
import os
LOADED_Q_CHANGE_FOR_QUENCH = 0.6

def validate_quench(fault_data, time_data, saved_loaded_q, frequency):
    """
    Parsing the fault waveforms to calculate the loaded Q to try to determine
    if a quench was real.

    DERIVATION NOTES
    A(t) = A0 * e^((-2 * pi * cav_freq * t)/(2 * loaded_Q)) = A0 * e ^ ((-pi * cav_freq * t)/loaded_Q)

    ln(A(t)) = ln(A0) + ln(e ^ ((-pi * cav_freq * t)/loaded_Q)) = ln(A0) - ((pi * cav_freq * t)/loaded_Q)
    polyfit(t, ln(A(t)), 1) = [-((pi * cav_freq)/loaded_Q), ln(A0)]
    polyfit(t, ln(A0/A(t)), 1) = [(pi * f * t)/Ql]

    https://education.molssi.org/python-data-analysis/03-data-fitting/index.html

    :return: bool representing whether quench was real
    """
    
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
    print(f"Saved Q: {saved_loaded_q:.2e}")
    print(f"Last recorded amplitude: {fault_data[0]}")
    print(f"Threshold: {thresh_for_quench:.2e}")
    print(f"Calculated Loaded Q: {loaded_q:.2e}")

    is_real = loaded_q < thresh_for_quench

    return is_real

def updated_validate_quench(fault_data, time_data, saved_loaded_q, frequency):
    """
    """

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

    return is_real
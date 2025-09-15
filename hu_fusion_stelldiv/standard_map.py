import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, root_scalar
import time

# initializing constants
TWOPI = 2 * np.pi

# parameters 
K = 0   # start with "k/TWOPI = 0" for initial conditions, then increase to around 5.19

"""
    This Python script is used to plot the poincare section for the STANDARD MAP
    which is given by y_{n+1} = y_n - k/TWOPI * np.sin(TWOPI*x) and x_{n+1} = x_n + y_{n+1}
    where 'y' is a momentum-like variable and 'x' is a coordinate-like variable that is assumed to be periodic with period 1

    Notes: 
        (i) the variables 'TWOPI*x' and 'y_n' respectively determine the angular position of the stick and its angular momentum after the n-th kick
        (ii) the constanst represented by 'k/TWOPI' measures the intensity of the kicks on the kicked rotator (mechanical system)
        (iii) its important to note that in order to calculate x_next, you first need y_next since x_next depends on it
        (iv) this standard map is a simple model of a conservative system that displays Hamiltonian chaos
    
    Examples:
        (i) For 'TWOPI*x = 0' the map is linear and only periodic and quasiperiodic orbits are possible
        (ii) When this map is plotted in phase space, periodic orbits appear as closed curves
        (iii) Nonlinearity of the map increases with 'TWOPI*x', and with it the possibility to observe chaotic dynamics for appropriate initial conditions
"""

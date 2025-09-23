import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, root_scalar

# initializing constants
TWOPI = 2 * np.pi

# parameters 
# K = 0   # start with "k/TWOPI = 0" for initial conditions, then increase to reproduce images

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

def standard_map(theta, p, K):
    """
    One iteration of the standard map.
    Equations: 
        p_{n+1} = p_n - K * sin(theta_n)
        theta_{n+1} = theta_n + p_{n+1}
    """
    p_new = p - K * np.sin(theta)
    theta_new = (theta + p_new) % (2*np.pi)
    return theta_new, p_new

def generate_points(K, n_ic=50, n_iter=2000, n_transient=500, seed=0):
    """
    Generate Poincare section points for the standard map.
    Parameters: 
        K: float
            standard map parameter
        n_ic: int
            number of random initial conditions
        n_transient: int
            total number of iterations to discard
        seed: int
            random seed for reproducibility
    Returns: 
        thetas, ps: np.ndarray
    """
    rng = np.random.default_rng(seed)
    theta0 = rng.uniform(0, 2*np.pi, n_ic)
    p0 = rng.uniform(-np.pi, np.pi, n_ic)   # start momenta around [-π, π]

    thetas, ps = [], []

    for i in range(n_ic):
        theta, p = theta0[i], p0[i]
        for _ in range (n_transient):
            theta, p = standard_map(theta, p, K)
        
        # collecting points
        for _ in range(n_iter - n_transient):
            theta, p = standard_map(theta, p, K)
            thetas.append(theta)
            ps.append(p)
    
    return np.array(thetas), np.array(ps)

if __name__ == "__main__":
    # parameters
    K_values = [0.5, 1.0, 2.5]
    n_ic = 80
    n_iter = 2000
    n_transient = 400

    # creating the figure
    fig, axes = plt.subplots(1, len(K_values), figsize=(15,4), constrained_layout=True)

    for ax, K in zip(axes, K_values):
        thetas, ps = generate_points(K, n_ic=n_ic, n_iter=n_iter, n_transient=n_transient, seed=42)
        ax.scatter(thetas, ps, s=0.4, alpha=0.6)
        ax.set_title(f"Standard map (K = {K})")
        ax.set_xlim(0, 2*np.pi)
        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel(r"$p$")
    
    plt.show()
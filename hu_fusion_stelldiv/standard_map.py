import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, root_scalar

# initializing constants
# TWOPI = 2 * np.pi

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

# function used to iterate the standard map
def standard_map(theta, p, K):
    """
    One iteration of the standard (Chirikov-Taylor) map.
    Equations: 
        p_{n+1} = p_n - K * sin(theta_n) where 'K' is the nonlinearity parameter controlling chaos
        theta_{n+1} = theta_n + p_{n+1}
    Important to note that theta is wrapped into [0,2π) because it's an angle
    and p is not wrapped so it can grow positive/negative freely
    """
    p_new = p - K * np.sin(theta)
    theta_new = (theta + p_new) % (2*np.pi)
    return theta_new, p_new

# function to generate Poincare section points for the standard map
def generate_points(K, n_ic=50, n_iter=2000, n_transient=500, seed=0):
    """
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
    theta0 = rng.uniform(0, 2*np.pi, n_ic)  # random theta in the range [0, 2π]
    p0 = rng.uniform(-np.pi, np.pi, n_ic)   # start momenta around [-π, π]

    thetas, ps = [], []

    for i in range(n_ic):
        theta, p = theta0[i], p0[i]

        # throwing away the transients
        for _ in range (n_transient):
            theta, p = standard_map(theta, p, K)
        
        # collecting points and recording trajectory
        for _ in range(n_iter - n_transient):
            theta, p = standard_map(theta, p, K)
            thetas.append(theta)
            ps.append(p)
    
    return np.array(thetas), np.array(ps)

def generate_points_grid(K, n_theta=20, n_p=20, n_iter=2000, n_transient=500):
    """
    Parameters: 
        K: float
            Standard map parameter
        n_theta: int 
            Number of grid points in theta direction
        n_p: int
            Number of grid points in p direction
        n_iter: int
            Total number of iterations per trajectory
        n_transient: int
            Steps to discard as transient
    
    Returns: 
        thetas, ps: np.ndarray
            Arrays of (theta, p) points for plotting
    """

    # regular grid in (theta, p) space
    theta_vals = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
    p_vals = np.linspace(-np.pi, np.pi, n_p, endpoint=False)
    theta0, p0 = np.meshgrid(theta_vals, p_vals)
    theta0 = theta0.ravel()
    p0 = p0.ravel()

    thetas, ps = [], []

    for theta, p in zip(theta0, p0):
        # Burn-in
        for _ in range(n_transient):
            theta, p = standard_map(theta, p, K)
        # Collect trajectory
        for _ in range(n_iter - n_transient):
            theta, p = standard_map(theta, p, K)
            thetas.append(theta)
            ps.append(p)

    return np.array(thetas), np.array(ps)


if __name__ == "__main__":
    # parameters for 'generate_points'
    K_values = [0, 0.5, 1.0]  # trying different values of K
    n_ic = 80                   # number of trajectories
    n_iter = 2000               # steps per trajectory
    n_transient = 400           # discarded transients

    # parameters for 'generate_points_grid'
    K_values = [0, 0.5, 1.0]  # values of K for different dynamical regimes
    n_theta = 20                # grid resolution in theta
    n_p = 20                    # grid resolution in p
    n_iter = 2000               # iterations per trajectory
    n_transient = 400           # transient steps to discard

    # creating the figure (one subplot per K value)
    # results in multiple panels showing how dynamics change as K increases
    fig, axes = plt.subplots(1, len(K_values), figsize=(15,4), constrained_layout=True)

    for ax, K in zip(axes, K_values):
        thetas, ps = generate_points(K, n_ic=n_ic, n_iter=n_iter, n_transient=n_transient, seed=42)
        # thetas, ps = generate_points_grid(K, n_theta=n_theta, n_p=n_p, n_iter=n_iter, n_transient=n_transient )
        ax.scatter(thetas, ps, s=0.4, alpha=0.6)
        ax.set_title(f"Standard map (K = {K})")
        ax.set_xlim(0, 2*np.pi)
        ax.set_ylim(-4.5, 4.5)
        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel(r"$p$")
    
    plt.show()
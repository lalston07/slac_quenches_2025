import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, root_scalar, newton
import time

# FOR USING NUMBA LIBRARY
from numba import jit, njit, vectorize

# Constants
TWOPI = 2 * np.pi
np_points = 3600  # Number of points for the map
dzeta = TWOPI / np_points  # Step size in toroidal angle

# Parameters for the stellarator
nperiods = 1
xnp = float(nperiods)
e0 = 0.5 # \epsilon_0
et = 0.5 # \epsilon_t
ex = -0.31 # \epsilon_x
iota0 = 0.15 
Bc = 1.0/np.pi  
wall_radius_sq = 16 # wall_radius = 4

# Number of iterations 
n_iterations = 1

# Initial conditions
initial_guess = 0.3        # radius
psi_t = initial_guess**2   # radius squared
theta_initial = 0
zeta_initial = 0
radius = psi_t
radii = [radius]


# USING NUMBA NEWTON'S METHOD 
# solve_psi_t_next as original function (f)
@njit
def f(x, psi_t, theta, zeta, e0, et, ex, iota0): 
    f_nonlinear = (
            (e0 / 4) * (-2*(2 * iota0 - 1) * np.sin(2 * theta - zeta) - 4 * iota0 * np.sin(2 * theta)) * x
            + (ex / 8) * (-4*(4 * iota0 - 1) * np.sin(4 * theta - zeta) - 16 * iota0 * np.sin(4 * theta)) * x**2.0
            + (et / 6) * (-3*(3 * iota0 - 1) * np.sin(3 * theta - zeta) + 9 * iota0 * np.sin(3 * theta)) * x**(1.5)
        )
    return x - (psi_t - dzeta * f_nonlinear)

# jacobian_psi_t_next as fprime (df)
@njit
def df(x, psi_t, theta, zeta, e0, et, ex, iota0):
    df_nonlinear = (
            (e0 / 4) * (-2*(2 * iota0 - 1) * np.sin(2 * theta - zeta) - 4 * iota0 * np.sin(2 * theta)) * 1
            + (ex / 8) * (-4*(4 * iota0 - 1) * np.sin(4 * theta - zeta) - 16 * iota0 * np.sin(4 * theta)) * (2.0*x)
            + (et / 6) * (-3*(3 * iota0 - 1) * np.sin(3 * theta - zeta) + 9 * iota0 * np.sin(3 * theta)) * (1.5*x**(0.5))
        )
    return 1 - ( - (dzeta * df_nonlinear))

@njit
def newton_solver(psi_t_initial, theta, zeta, e0, et, ex, iota0, tol=1e-14, max_iter=100):
    psi = psi_t_initial  # psi_t (x) = psi_t_initial (x0)
    iter = 0
    converge_flag = False
    while iter < max_iter and not converge_flag:
        f_val = f(psi, psi_t_initial, theta, zeta, e0, et, ex, iota0)
        df_val = df(psi, psi_t_initial, theta, zeta, e0, et, ex, iota0)
        if df_val == 0:
            break
        else:
            psi_new = psi - f_val / df_val
            if np.abs(psi_new - psi) < tol:
                converge_flag = True
                # return psi, theta_next, zeta_next
                break
        psi = psi_new
        iter = iter+1

    theta_next = theta + (      
            (iota0 + (e0 / 4) * ((2 * iota0 - 1) * np.cos(2 * theta - zeta) + 2 * iota0 * np.cos(2 * theta)))
            + (ex / 8) * ((4 * iota0 - 1) * np.cos(4 * theta - zeta) + 4 * iota0 * np.cos(4 * theta)) * 2 * psi
            + (et / 6) * ((3 * iota0 - 1) * np.cos(3 * theta - zeta) - 3 * iota0 * np.cos(3 * theta)) * (3/2) * psi**(0.5)
        ) * dzeta 
    
    zeta_next = zeta + dzeta

    return psi, theta_next, zeta_next

for r in radii:
    initial_guess = 0.3       # radius
    psi_t = initial_guess**2  # radius squared
    theta = theta_initial
    zeta = zeta_initial
    trajectory = [(psi_t, theta, zeta)]

    for i in range(n_iterations): 
        zeta = 0
        while zeta < TWOPI:
            psi_t_next, theta_next, zeta_next = newton_solver(trajectory[-1][0], trajectory[-1][1], zeta, e0, et, ex, iota0)
            zeta = zeta_next # zeta_next = zeta + dzeta
            trajectory.append((psi_t_next, theta_next, zeta_next))

# last_psi, last_theta, last_zeta = trajectory[-1]
# root, theta_i, zeta_i = newton_solver(last_psi, last_theta, last_zeta, e0, et, ex, iota0)
last_psi = trajectory[-1][0]
root, theta_i, zeta_i = newton_solver(last_psi, theta, zeta, e0, et, ex, iota0)
# root, theta_i, zeta_i = newton_solver(initial_guess**2, theta, zeta, e0, et, ex, iota0)
print(f"NUMBA NEWTONS METHOD - Root: {root}")






# USING SCIPY FSOLVE METHOD (scipy.optimize.fsolve)
def jacobian_psi_t_next(x, *args):
        psi_t, theta, zeta, e0, et, ex, iota0 = args 
        jac = 1 - ( - (
            (e0 / 4) * (-2*(2 * iota0 - 1) * np.sin(2 * theta - zeta) - 4 * iota0 * np.sin(2 * theta)) * 1
            + (ex / 8) * (-4*(4 * iota0 - 1) * np.sin(4 * theta - zeta) - 16 * iota0 * np.sin(4 * theta)) * (2.0*x)
            + (et / 6) * (-3*(3 * iota0 - 1) * np.sin(3 * theta - zeta) + 9 * iota0 * np.sin(3 * theta)) * (1.5*x**(0.5))
        ) * dzeta)
        return jac

def solve_psi_t_next(x, *args):
    psi_t, theta, zeta, e0, et, ex, iota0 = args 
    # print(x)
    f = x - (psi_t - (
        (e0 / 4) * (-2*(2 * iota0 - 1) * np.sin(2 * theta - zeta) - 4 * iota0 * np.sin(2 * theta)) * x
        + (ex / 8) * (-4*(4 * iota0 - 1) * np.sin(4 * theta - zeta) - 16 * iota0 * np.sin(4 * theta)) * x**2.0
        + (et / 6) * (-3*(3 * iota0 - 1) * np.sin(3 * theta - zeta) + 9 * iota0 * np.sin(3 * theta)) * x**(1.5)
    ) * dzeta)
    return f

def hamiltonian_map(psi_t, theta, zeta, e0, et, ex, iota0):
    psi_t_next = fsolve(solve_psi_t_next, psi_t, fprime=jacobian_psi_t_next, \
        args=(psi_t, theta, zeta, e0, et, ex, iota0), xtol=1.0e-15)
    
    theta_next = theta + (      
            (iota0 + (e0 / 4) * ((2 * iota0 - 1) * np.cos(2 * theta - zeta) + 2 * iota0 * np.cos(2 * theta)))
            + (ex / 8) * ((4 * iota0 - 1) * np.cos(4 * theta - zeta) + 4 * iota0 * np.cos(4 * theta)) * 2 * psi_t_next
            + (et / 6) * ((3 * iota0 - 1) * np.cos(3 * theta - zeta) - 3 * iota0 * np.cos(3 * theta)) * (3/2) * psi_t_next**(0.5)
        ) * dzeta 
    
    zeta_next = zeta + dzeta

    # print(f"NUMPY FSOLVE METHOD - Root: {psi_t_next[0]}")
    return psi_t_next, theta_next, zeta_next

for r in radii:
    initial_guess = 0.3       # radius
    psi_t = initial_guess**2  # radius squared
    theta = theta_initial
    zeta = zeta_initial
    trajectory = [(psi_t, theta, zeta)]

    for i in range(n_iterations): 
        zeta = 0
        while zeta < TWOPI:
            psi_t_next, theta_next, zeta_next = hamiltonian_map(trajectory[-1][0], trajectory[-1][1], zeta, e0, et, ex, iota0)
            zeta = zeta_next # zeta_next = zeta + dzeta
            trajectory.append((psi_t_next[0], theta_next[0], zeta_next))

# calling the function so it will print
# hamiltonian_map(psi_t, theta, zeta, e0, et, ex, iota0)
print(f"SCIPY FSOLVE METHOD - Root: {psi_t_next[0]}")





# USING SCIPY NEWTON'S METHOD (scipy.optimize.newton)
def f(x, psi_t, theta, zeta, e0, et, ex, iota0): 
    return x - (psi_t - (
            (e0 / 4) * (-2*(2 * iota0 - 1) * np.sin(2 * theta - zeta) - 4 * iota0 * np.sin(2 * theta)) * x
            + (ex / 8) * (-4*(4 * iota0 - 1) * np.sin(4 * theta - zeta) - 16 * iota0 * np.sin(4 * theta)) * x**2.0
            + (et / 6) * (-3*(3 * iota0 - 1) * np.sin(3 * theta - zeta) + 9 * iota0 * np.sin(3 * theta)) * x**(1.5)
        ) * dzeta)

# jacobian_psi_t_next as fprime (df)
def df(x, psi_t, theta, zeta, e0, et, ex, iota0):
    return 1 - ( - (
            (e0 / 4) * (-2*(2 * iota0 - 1) * np.sin(2 * theta - zeta) - 4 * iota0 * np.sin(2 * theta)) * 1
            + (ex / 8) * (-4*(4 * iota0 - 1) * np.sin(4 * theta - zeta) - 16 * iota0 * np.sin(4 * theta)) * (2.0*x)
            + (et / 6) * (-3*(3 * iota0 - 1) * np.sin(3 * theta - zeta) + 9 * iota0 * np.sin(3 * theta)) * (1.5*x**(0.5))
        ) * dzeta)

x0 = psi_t
last_psi = trajectory[-1][0]

# print("Initial guess passed to scipy.newton", x0)
root_scipy = newton(func=f, x0=last_psi, fprime=df, args=(last_psi, theta, zeta, e0, et, ex, iota0))

print(f"SCIPY NEWTONS METHOD - Root:", root_scipy)
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, root_scalar, newton
import time
from numba import jit, njit, vectorize

# NUMBA NEWTONS METHOD
@njit
def f(x):
    return x**3 - 27

@njit
def df(x):
    return 3*x**2

@njit
def newton_solver(x0, tol=1e-8, max_iter=100):
    x = x0
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        if dfx == 0:
            break
        else:
            x_new = x - fx / dfx
            if np.abs(x_new - x) < tol:
                return x_new, i+1
            x = x_new

initial_guess = 1.0
root, iterations = newton_solver(initial_guess)
print(f"NUMBA NEWTONS METHOD - Root: {root}, Iterations: {iterations}")


# SCIPY FSOLVE METHOD
def f(x):   
    return x**3 - 27
x0 = 1.0 
solution = fsolve(f, x0) 
print(f"NUMPY FSOLVE - Root:", solution[0])


# SCIPY NEWTONS METHOD
def f(x):
    return x**3 - 27
def df(x):
    return 3*x**2
x0 = 1.0
root = newton(func=f, x0=x0, fprime=df)
print(f"NUMPY NEWTONS METHOD - Root:", root)


# SCIPY HALLEY'S METHOD
def f(x):
    return x**3 - 27
def df(x):
    return 3*x**2
def d2f(x):
    return 6*x
x0 = 1.0
root_halley = newton(func=f, x0=x0, fprime=df, fprime2=d2f)
print(f"NUMPY HALLEY's METHOD - Root (Halley):", root_halley)



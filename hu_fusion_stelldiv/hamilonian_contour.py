import sys
import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, root_scalar

# GRID RESOLUTION
r = np.linspace(-0.8, 0.8, 200) # we want the range -1 <= r <= 1
theta = np.linspace(0, 2*np.pi, 200) # we want the range 0 <= theta <= 2*np.pi
# R, Theta = np.meshgrid(r, theta)
x_grid = r * np.cos(theta)
y_grid = r * np.sin(theta)
x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)


# PARAMETER LIST
Bc = 1/np.pi
e0 = 0.5 # \epsilon_0
et = 0.5 # \epsilon_t
ex = -0.31 # \epsilon_x
iota0 = 0.15 
zeta = 0 # value is zero to define psi_p as depending on two variables (r, theta)

# COMPUTE psi_p ON THE GRID
def model_hamiltonian(x_mesh, y_mesh, *args):
    
    zeta, e0, et, ex, iota0 = args

    r_sqrd = x_mesh**2 + y_mesh**2
    theta = np.arctan2(y_mesh,x_mesh) #theta=np.arctan(y_mesh/x_mesh)
    psi_t = np.pi*r_sqrd * Bc # toroidal flux
    psi_p = (iota0 + (e0/4)*((2*iota0 - 1)*np.cos(2*theta - zeta) + 2*iota0*np.cos(2*theta)))*psi_t \
        + (ex/8)*((4*iota0 - 1)*np.cos(4*theta - zeta) + 4*iota0*np.cos(4*theta))*(psi_t**2) \
        + (et/6)*((3*iota0 - 1)*np.cos(3*theta - zeta) - 3*iota0*np.cos(3*theta))*(psi_t**(3/2))
    
    return psi_p

# COMPUTE psi_p values
# psi_p = model_hamiltonian(R, Theta, zeta, e0, et, ex, iota0)
psi_p = model_hamiltonian(x_mesh, y_mesh, zeta, e0, et, ex, iota0)

# CONVERT TO CARTESIAN COORDINATES
# X = R * np.cos(Theta)
# Y = R * np.sin(Theta)

# PLOTTING
plt.figure(figsize=(8,6))
# CS2 = plt.contourf(x_mesh, y_mesh, psi_p, levels = 20, cmap = 'plasma')
CS = plt.contour(x_mesh, y_mesh, psi_p, levels = 100, colors='black', linestyles='dashed', linewidths=0.8)
plt.colorbar(CS, label=r'$\psi_p$')
plt.xlabel('x')
plt.ylabel('y')
plt.title(r'$\psi_p(r, \theta)$ in Cartesian Coordinates')
plt.axis('equal')
# plt.xlim(-2.5, 2.5)
# plt.ylim(-2.5, 2.5)
plt.grid(True)
plt.tight_layout()
plt.show()
    
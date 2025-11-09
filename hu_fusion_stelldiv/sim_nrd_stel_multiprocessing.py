import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, root_scalar
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
# wall_radius = 4
wall_radius_sq = 16

theta_initial = 0.0
zeta_initial = 0.0

# @njit decorators are added to all functions to speed them up
# Numba does better when function are not nested so these functions are moved outside of poincare_section_fieldline 

@njit
def solve_psi_t_next(psi_guess, psi_t, theta, zeta, e0, et, ex, iota0): 
    f_nonlinear = (
            (e0 / 4) * (-2*(2 * iota0 - 1) * np.sin(2 * theta - zeta) - 4 * iota0 * np.sin(2 * theta)) * psi_guess
            + (ex / 8) * (-4*(4 * iota0 - 1) * np.sin(4 * theta - zeta) - 16 * iota0 * np.sin(4 * theta)) * psi_guess**2.0
            + (et / 6) * (-3*(3 * iota0 - 1) * np.sin(3 * theta - zeta) + 9 * iota0 * np.sin(3 * theta)) * psi_guess**(1.5)
        )
    return psi_guess - (psi_t - dzeta * f_nonlinear)

@njit
def jacobian_psi_t_next(psi_guess, psi_t, theta, zeta, e0, et, ex, iota0):
    df_nonlinear = (
            (e0 / 4) * (-2*(2 * iota0 - 1) * np.sin(2 * theta - zeta) - 4 * iota0 * np.sin(2 * theta)) * 1
            + (ex / 8) * (-4*(4 * iota0 - 1) * np.sin(4 * theta - zeta) - 16 * iota0 * np.sin(4 * theta)) * (2.0*psi_guess)
            + (et / 6) * (-3*(3 * iota0 - 1) * np.sin(3 * theta - zeta) + 9 * iota0 * np.sin(3 * theta)) * (1.5*psi_guess**(0.5))
        )
    return 1 - ( - (dzeta * df_nonlinear))
        
# FSOLVE IS NOT COMPATIBLE WITH NUMBA SO WE USE NEWTONS METHOD INSTEAD
# THIS BLOCK OF CODE REPLACES solve_psi_t_next from sim_nrd_stel.py code
@njit   
def newton_solver(psi_t_init, theta, zeta, e0, et, ex, iota0, tol=1e-14, max_iter=100):
    """
    Function to call for solving the implicit equation for toroidal flux update
    """
    psi = psi_t_init

    # warning statement for if it does not converge
    
    iter = 0
    converge_flag = False
    while iter < max_iter and not converge_flag:
        f = solve_psi_t_next(psi, psi_t_init, theta, zeta, e0, et, ex, iota0)
        df = jacobian_psi_t_next(psi, psi_t_init, theta, zeta, e0, et, ex, iota0)
        if df == 0:
            break
        else:
            psi_new = psi - f / df
            if np.abs(psi_new - psi) < tol:
                converge_flag = True
                # return psi_new
                break
        psi = psi_new
        iter = iter+1
    return psi  # fallback 

@njit        
def hamiltonian_map(psi_t, theta, zeta, e0, et, ex, iota0):
    """
    Hamiltonian map equations for the stellarator.
    Computes the next step in toroidal flux and poloidal angle using a custom Newton-Raphson solver.
    """
    psi_t_next = newton_solver(psi_t, theta, zeta, e0, et, ex, iota0)

    theta_next = theta + (      
        (iota0 + (e0 / 4) * ((2 * iota0 - 1) * np.cos(2 * theta - zeta) + 2 * iota0 * np.cos(2 * theta))) +
        (ex / 8) * ((4 * iota0 - 1) * np.cos(4 * theta - zeta) + 4 * iota0 * np.cos(4 * theta)) * 2 * psi_t_next +
        (et / 6) * ((3 * iota0 - 1) * np.cos(3 * theta - zeta) - 3 * iota0 * np.cos(3 * theta)) * 1.5 * psi_t_next**0.5
    ) * dzeta

    zeta_next = zeta + dzeta

    return psi_t_next, theta_next, zeta_next
       
def poincare_section_fieldline(radius = 0.6, n_iterations = 1000):
    # Initial conditions
    psi_t = np.pi * radius**2 * Bc      # Initial psi_t or toroidal flux
    theta = theta_initial               # Initial theta = 0.0
    zeta = zeta_initial                 # Initial zeta = 0.0

    data = np.empty((n_iterations, 3)) # Used to store psi_t, theta, zeta
    # wall_hit_flag = False
    # wall_hit_iter = -1

    # for i in range(n_iterations):
    #     psi_t, theta, zeta = hamiltonian_map(psi_t, theta, zeta, e0, et, ex, iota0)
    #     data[i, 0] = psi_t
    #     data[i, 1] = theta
    #     data[i, 2] = zeta

    #     # break if radius is exceeded
    #     if psi_t / (np.pi * Bc) > wall_radius_sq:
    #         wall_hit_flag = True
    #         print(f"For radius = {radius:.6f}, field line hits the vessel wall after {i:6d} iterations")
    #         break

    #     # if wall_hit_flag:
    #     #     return data, wall_hit_iter
    #     # else: 
    #     #     return data, -1
    
    # radii = [radius]
    # data, hit_iter = poincare_section_fieldline(radius, n_iterations)
    
    # if hit_iter >= 0:
    #     print(f"For radius = {radius:.6f}, field line hits the vessel wall after {hit_iter:6d} iterations")
    # else:
    #     print(f"Field line did not hit the wall within {n_iterations} iterations")
    
    # DEFINING ARRAY TO STORE POINCARE SECTION CALCULATIONS AT EVERY 45 DEGREES
    phase_data = []
    phase_data_zeta_pi = []
    phase_data_zeta_pi_over4 = []
    phase_data_zeta_pi_over2 = []
    phase_data_zeta_3pi_over4 = []

    start_time = time.time()

    # for r in radii:
    wall_hit_flag = False # Resets flag for each new radius
    # psi_t = np.pi * r**2 * Bc # Toroidal flux
    # theta = theta_initial
    # zeta = zeta_initial
    trajectory = [(psi_t, theta, zeta)] # THIS IS THE LINE THAT CREATES THE TRAJECTORY AROUND IN THE DEVICE
    # COORDINATE TRANSFORMATION FROM THIS (FLUX SURFACE COORDINATES) TO R and Z (CYCLINDRICAL COORDINATES) 
    # TRAJECTORY IS AN ARRAY (dimensions number of points x three)
    # AFTER COORDINATE TRANSFORMATION THERE WILL BE (number of points x two for R and Z)
    # START WITH 1000 or 2000 points at a smaller radius r=0.5, 0.6, 0.7 close to axis (well-defined frequencies) to use in fft.py file

    # reset data for each radius
    # phase_data = []
    # phase_data_zeta_pi = []
    # phase_data_zeta_pi_over4 = []
    # phase_data_zeta_pi_over2 = []
    # phase_data_zeta_3pi_over4 = []

    phase_data.append((psi_t, theta, zeta))
    print(f"Computing phase portrait for radius = {radius:.4f} with iterations = {n_iterations:d}")
    
    for i in range(n_iterations): 
        zeta = zeta_initial        
        while zeta < TWOPI:
            # start_time_iter = time.time()
            psi_t_next, theta_next, zeta_next = hamiltonian_map(trajectory[-1][0], trajectory[-1][1], zeta, e0, et, ex, iota0)
            # end_time_iter = time.time()
            # print("time to iterate once: ", (end_time_iter - start_time_iter))

            zeta = zeta_next # zeta_next = zeta + dzeta
            trajectory.append((psi_t_next, theta_next, zeta_next))
            # print(i, np.mod(zeta, np.pi))

            # check if the next iterate falls outside the vessel wall
            # x_next = np.sqrt(psi_t_next / (np.pi * Bc))*np.cos(theta_next)
            # y_next = np.sqrt(psi_t_next / (np.pi * Bc))*np.sin(theta_next)
            # if x_next**2 + y_next**2 > wall_radius_sq:
            if psi_t_next / (np.pi * Bc) > wall_radius_sq:
                print(f"For radius = {radius:.6f}, field line hits the vessel wall after {i:6d} iterations")
                wall_hit_flag = True
                break
                
            if np.abs(zeta - 2*np.pi) < 1.0e-8:
                # print(zeta)
                phase_data.append((psi_t_next, theta_next, zeta_next))

            # PLOT WHEN ZETA = PI
            if np.abs(zeta - np.pi) < 1.0e-8:
                # print(zeta)
                phase_data_zeta_pi.append((psi_t_next, theta_next, zeta_next))

            # MORE SECTIONS FOR EVERY 45 DEGREES BETWEEN 0 AND PI
            # PLOT WHEN ZETA = PI/4
            if np.abs(zeta-(np.pi/4)) < 1.0e-8:
                phase_data_zeta_pi_over4.append((psi_t_next, theta_next, zeta_next))
            
            # PLOT WHEN ZETA = PI/2
            if np.abs(zeta-(np.pi/2)) < 1.0e-8:
                phase_data_zeta_pi_over2.append((psi_t_next, theta_next, zeta_next))

            # PLOT WHEN ZETA = 3PI/4
            if np.abs(zeta-3*(np.pi/4)) < 1.0e-8:
                phase_data_zeta_3pi_over4.append((psi_t_next, theta_next, zeta_next))

        # print(psi_t_next, theta_next, zeta)
        
        if wall_hit_flag:
            break
    
    end_time = time.time()
    print(end_time-start_time)

    # filename to save needs to account for higher resolution runs!!!
    # Saving the trajectory and Poincare section data
    # SAVING PLOT AT ZETA = 0
    np.array(phase_data).dump(open(f"numba_phase_portrait_r={radius:.4f}_i={n_iterations:06d}.npy", 'wb'))
    np.savetxt(f"numba_phase_portrait_r={radius:.4f}_i={n_iterations:06d}.txt", np.array(phase_data).squeeze(), \
    fmt='%.18e', delimiter=' ', newline='\n')
    
    # SAVING PLOT AT ZETA = PI
    np.array(phase_data_zeta_pi).dump(open(f"numba_phase_portrait_zeta_pi_r={radius:.4f}_i={n_iterations:06d}.npy", 'wb'))
    np.savetxt(f"numba_phase_portrait_zeta_pi_r={radius:.4f}_i={n_iterations:06d}.txt", np.array(phase_data_zeta_pi).squeeze(), \
        fmt='%.18e', delimiter=' ', newline='\n')

    # SAVING PLOT AT ZETA = PI/4
    np.array(phase_data_zeta_pi_over4).dump(open(f"numba_phase_portrait_zeta_pi_over4_r={radius:.4f}_i={n_iterations:06d}.npy", 'wb'))
    np.savetxt(f"numba_phase_portrait_zeta_pi_over4_r={radius:.4f}_i={n_iterations:06d}.txt", np.array(phase_data_zeta_pi_over4).squeeze(), \
        fmt='%.18e', delimiter=' ', newline='\n')
    
    # SAVING PLOT AT ZETA = PI/2
    np.array(phase_data_zeta_pi_over2).dump(open(f"numba_phase_portrait_zeta_pi_over2_r={radius:.4f}_i={n_iterations:06d}.npy", 'wb'))
    np.savetxt(f"numba_phase_portrait_zeta_pi_over2_r={radius:.4f}_i={n_iterations:06d}.txt", np.array(phase_data_zeta_pi_over2).squeeze(), \
        fmt='%.18e', delimiter=' ', newline='\n')
    
    # SAVING PLOT AT ZETA = 3PI/4
    np.array(phase_data_zeta_3pi_over4).dump(open(f"numba_phase_portrait_zeta_3pi_over4_r={radius:.4f}_i={n_iterations:06d}.npy", 'wb'))
    np.savetxt(f"numba_phase_portrait_zeta_3pi_over4_r={radius:.4f}_i={n_iterations:06d}.txt", np.array(phase_data_zeta_3pi_over4).squeeze(), \
        fmt='%.18e', delimiter=' ', newline='\n')

    # THESE LINES OF CODE SAVE THE TRAJECTORY DATA
    np.array(trajectory).dump(open(f"trajectory_r={radius:.2f}.npy", 'wb'))
    np.savetxt(f"trajectory_r={radius:.2f}.txt", np.array(trajectory).squeeze(), \
        fmt='%.18e', delimiter=' ', newline='\n')
    print(f"Saved file for radius = {radius:.4f}")

    # Plot Phase Portraits
    # SAVING FIGURE FOR ZETA = 0
    plt.figure(figsize=(10, 6))
    # for i, data in enumerate(phase_data):
    #     psi_t_vals, theta_vals = zip(*data)
    #     psi_t_vals = np.array(psi_t_vals)
    #     theta_vals = np.array(theta_vals)
    #     x = np.sqrt(psi_t_vals/(np.pi * Bc)) * np.cos(theta_vals)
    #     y = np.sqrt(psi_t_vals/(np.pi * Bc)) * np.sin(theta_vals)
    #     plt.plot(theta_vals, np.sqrt(iota0_vals), label=f"Radius r={radii[i]:.1f}")
    #     plt.plot(x, y, label=f"Radius r={radii[i]:.1f}")
    
    # Unzipping the entire phase_data list
    psi_t_vals, theta_vals, zeta_vals = zip(*phase_data)
    psi_t_vals = np.array(psi_t_vals)
    theta_vals = np.array(theta_vals)
    x = np.sqrt(psi_t_vals/(np.pi * Bc)) * np.cos(theta_vals)
    y = np.sqrt(psi_t_vals/(np.pi * Bc)) * np.sin(theta_vals)
    # plt.plot(theta_vals, np.sqrt(iota0_vals), label=f"Radius r={radii[i]:.1f}")
    plt.plot(x, y, '.r', markersize = 1, label=f"r={radius:.4f}")
    # plt.xlabel("Poloidal Angle \u03b8 (radians)")
    # plt.ylabel("Toroidal Flux \u03c8")
    plt.title("Phase Portrait in Poloidal Plane")
    # plt.set_aspect('equal')
    # plt.xlim([-2,2])
    # plt.ylim([-2,2])
    plt.gca().set_aspect('equal')
    plt.legend(loc='upper right') 
    plt.grid()  
    plt_filename = f"numba_phase_portrait_fig1a_r={radius:.4f}_i={n_iterations:06d}.png" # i = ___ CHANGES FOR NUMBER OF ITERATIONS
    plt.savefig(plt_filename, bbox_inches='tight', dpi=200)
    plt.close()

    # SAVING FIGURE FOR ZETA = PI
    plt.figure(figsize=(10, 6))
    # Unzipping the entire phase_data list
    psi_t_vals, theta_vals, zeta_vals = zip(*phase_data_zeta_pi)
    psi_t_vals = np.array(psi_t_vals)
    theta_vals = np.array(theta_vals)
    x = np.sqrt(psi_t_vals/(np.pi * Bc)) * np.cos(theta_vals)
    y = np.sqrt(psi_t_vals/(np.pi * Bc)) * np.sin(theta_vals)
    # plt.plot(theta_vals, np.sqrt(iota0_vals), label=f"Radius r={radii[i]:.1f}")
    plt.plot(x, y, '.r', markersize = 1, label=f"r={radius:.4f}")

    # plt.xlabel("Poloidal Angle \u03b8 (radians)")
    # plt.ylabel("Toroidal Flux \u03c8")
    plt.title("Phase Portrait in Poloidal Plane")
    # plt.set_aspect('equal')
    # plt.xlim([-2,2])
    # plt.ylim([-2,2])
    plt.gca().set_aspect('equal')
    plt.legend(loc='upper right') 
    plt.grid()  
    plt_filename = f"numba_phase_portrait_fig1b_r={radius:.4f}_i={n_iterations:06d}.png" # i = ___ CHANGES FOR NUMBER OF ITERATIONS
    plt.savefig(plt_filename, bbox_inches='tight', dpi=200)
    plt.close()


    # SAVING THE FIGURES
    # SAVING FIGURE FOR ZETA = PI/4
    # flux_to_cartesian =  input: psi_t, theta output: x, y parameters: Bc
    plt.figure(figsize=(10, 6))
    # Unzipping the entire phase_data list
    psi_t_vals, theta_vals, zeta_vals = zip(*phase_data_zeta_pi_over4) ## CHANGES FOR ZETA VALUES
    psi_t_vals = np.array(psi_t_vals)
    theta_vals = np.array(theta_vals)
    x = np.sqrt(psi_t_vals/(np.pi * Bc)) * np.cos(theta_vals)
    y = np.sqrt(psi_t_vals/(np.pi * Bc)) * np.sin(theta_vals)
    # plt.plot(theta_vals, np.sqrt(iota0_vals), label=f"Radius r={radii[i]:.1f}")
    plt.plot(x, y, '.r', markersize = 1, label=f"r={radius:.4f}")

    # plt.xlabel("Poloidal Angle \u03b8 (radians)")
    # plt.ylabel("Toroidal Flux \u03c8")
    plt.title("Phase Portrait in Poloidal Plane")
    # plt.set_aspect('equal')
    # plt.xlim([-2,2])
    # plt.ylim([-2,2])
    plt.gca().set_aspect('equal')
    plt.legend(loc='upper right') 
    plt.grid()  
    plt_filename = f"numba_phase_portrait_piby4_r={radius:.4f}_i={n_iterations:06d}.png" # LINE CHANGES FOR ZETA VALUES # i = ___ CHANGES FOR NUMBER OF ITERATIONS
    plt.savefig(plt_filename, bbox_inches='tight', dpi=200)
    plt.close()

    #SAVING FIGURE FOR ZETA = PI/2
    plt.figure(figsize=(10, 6))
    # Unzipping the entire phase_data list
    psi_t_vals, theta_vals, zeta_vals = zip(*phase_data_zeta_pi_over2) # LINE CHANGES FOR ZETA VALUES
    psi_t_vals = np.array(psi_t_vals)
    theta_vals = np.array(theta_vals)
    x = np.sqrt(psi_t_vals/(np.pi * Bc)) * np.cos(theta_vals)
    y = np.sqrt(psi_t_vals/(np.pi * Bc)) * np.sin(theta_vals)
    # plt.plot(theta_vals, np.sqrt(iota0_vals), label=f"Radius r={radii[i]:.1f}")
    plt.plot(x, y, '.r', markersize = 1, label=f"r={radius:.4f}")

    # plt.xlabel("Poloidal Angle \u03b8 (radians)")
    # plt.ylabel("Toroidal Flux \u03c8")
    plt.title("Phase Portrait in Poloidal Plane")
    # plt.set_aspect('equal')
    # plt.xlim([-2,2])
    # plt.ylim([-2,2])
    plt.gca().set_aspect('equal')
    plt.legend(loc='upper right') 
    plt.grid()  
    plt_filename = f"numba_phase_portrait_piby2_r={radius:.4f}_i={n_iterations:06d}.png" # LINE CHANGES FOR ZETA VALUES # i = ___ CHANGES FOR NUMBER OF ITERATIONS
    plt.savefig(plt_filename, bbox_inches='tight', dpi=200)
    plt.close()

    # SAVING FIGURE FOR ZETA = 3PI/4
    plt.figure(figsize=(10, 6))
    # Unzipping the entire phase_data list
    psi_t_vals, theta_vals, zeta_vals = zip(*phase_data_zeta_3pi_over4) # CHANGES FOR ZETA VALUES 
    psi_t_vals = np.array(psi_t_vals)
    theta_vals = np.array(theta_vals)
    x = np.sqrt(psi_t_vals/(np.pi * Bc)) * np.cos(theta_vals)
    y = np.sqrt(psi_t_vals/(np.pi * Bc)) * np.sin(theta_vals)
    # plt.plot(theta_vals, np.sqrt(iota0_vals), label=f"Radius r={radii[i]:.1f}")
    plt.plot(x, y, '.r', markersize = 1, label=f"r={radius:.4f}")

    # plt.xlabel("Poloidal Angle \u03b8 (radians)")
    # plt.ylabel("Toroidal Flux \u03c8")
    plt.title("Phase Portrait in Poloidal Plane")
    # plt.set_aspect('equal')
    # plt.xlim([-2,2])
    # plt.ylim([-2,2])
    plt.gca().set_aspect('equal')
    plt.legend(loc='upper right') 
    plt.grid()  
    plt_filename = f"numba_phase_portrait_3piby4_r={radius:.4f}_i={n_iterations:06d}.png" # LINE CHANGES FOR ZETA VALUES # i = ___ CHANGES FOR NUMBER OF ITERATIONS
    plt.savefig(plt_filename, bbox_inches='tight', dpi=200)
    plt.close()
    

    
    
    

    
    
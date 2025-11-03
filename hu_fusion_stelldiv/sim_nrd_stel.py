import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, root_scalar
import time

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

def poincare_section_fieldline(radius=0.2, n_iterations=100):
    
    def jacobian_psi_t_next(x, *args):
        
        psi_t, theta, zeta, e0, et, ex, iota0 = args 
        
        jac = 1 - ( - (
            (e0 / 4) * (-2*(2 * iota0 - 1) * np.sin(2 * theta - zeta) - 4 * iota0 * np.sin(2 * theta)) * 1
            + (ex / 8) * (-4*(4 * iota0 - 1) * np.sin(4 * theta - zeta) - 16 * iota0 * np.sin(4 * theta)) * (2.0*x)
            + (et / 6) * (-3*(3 * iota0 - 1) * np.sin(3 * theta - zeta) + 9 * iota0 * np.sin(3 * theta)) * (1.5*x**(0.5))
        ) * dzeta)
        
        return jac
    
    def solve_psi_t_next(x, *args):
        """
        Function to call for solving the implicit equation for toroidal flux update
        """
        psi_t, theta, zeta, e0, et, ex, iota0 = args 
        # print(x)
        f = x - (psi_t - (
            (e0 / 4) * (-2*(2 * iota0 - 1) * np.sin(2 * theta - zeta) - 4 * iota0 * np.sin(2 * theta)) * x
            + (ex / 8) * (-4*(4 * iota0 - 1) * np.sin(4 * theta - zeta) - 16 * iota0 * np.sin(4 * theta)) * x**2.0
            + (et / 6) * (-3*(3 * iota0 - 1) * np.sin(3 * theta - zeta) + 9 * iota0 * np.sin(3 * theta)) * x**(1.5)
        ) * dzeta)
        
        return f

    def hamiltonian_map(psi_t, theta, zeta, e0, et, ex, iota0):
        """
        Hamiltonian map equations for the stellarator.
        Computes the next step in toroidal flux and poloidal angle using the Symplectic Euler method
        """
        # psi_t_next = psi_t - (
        #     (e0 / 2) * (-(2 * iota0 - 1) * np.sin(2 * theta - zeta) - 4 * iota0 * np.sin(2 * theta)) * psi_t
        #     + (ex / 2) * (-(4 * iota0 - 1) * np.sin(4 * theta - zeta) - 16 * iota0 * np.sin(4 * theta)) * psi_t**2.0
        #     + (et / 2) * (-(3 * iota0 - 1) * np.sin(3 * theta - zeta) + 9 * iota0 * np.sin(3 * theta)) * psi_t**(1.5)
        # ) * dzeta
        
        
        # using root solver
        psi_t_next = fsolve(solve_psi_t_next, psi_t, fprime=jacobian_psi_t_next, \
            args=(psi_t, theta, zeta, e0, et, ex, iota0), xtol=1.0e-15)

        theta_next = theta + (      
            (iota0 + (e0 / 4) * ((2 * iota0 - 1) * np.cos(2 * theta - zeta) + 2 * iota0 * np.cos(2 * theta)))
            + (ex / 8) * ((4 * iota0 - 1) * np.cos(4 * theta - zeta) + 4 * iota0 * np.cos(4 * theta)) * 2 * psi_t_next
            + (et / 6) * ((3 * iota0 - 1) * np.cos(3 * theta - zeta) - 3 * iota0 * np.cos(3 * theta)) * (3/2) * psi_t_next**(0.5)
        ) * dzeta 

        zeta_next = zeta + dzeta
        
        # print(psi_t_next)
        return psi_t_next, theta_next, zeta_next

    # Initial conditions
    # radii = [0.7, 1.0, 1.3]  # Example radii for phase portraits
    radii = [0.86, 0.88]
    # radii = [radius]
    zeta_initial = 0.0
    theta_initial = 0.0
    
    # Phase Portrait calculation
    wall_hit_flag = False
    phase_data = []
    phase_data_zeta_pi = []

    # DEFINING ARRAY TO STORE POINCARE SECTION CALCULATIONS AT EVERY 45 DEGREES
    phase_data_zeta_pi_over4 = []
    phase_data_zeta_pi_over2 = []
    phase_data_zeta_3pi_over4 = []

    start_time = time.time()

    for r in radii:
        psi_t = np.pi * r**2 * Bc # Toroidal flux
        theta = theta_initial
        zeta = zeta_initial
        trajectory = [(psi_t, theta, zeta)]

        phase_data.append((psi_t, theta, zeta))
        print(f"Computing phase portrait for radius = {r:.4f} with iterations = {n_iterations:d}")
        
        for i in range(n_iterations): 
            zeta = zeta_initial        
            while zeta < TWOPI:
                # start_time_iter = time.time()
                psi_t_next, theta_next, zeta_next = hamiltonian_map(trajectory[-1][0], trajectory[-1][1], zeta, e0, et, ex, iota0)
                # end_time_iter = time.time()
                # print("time to iterate once: ", (end_time_iter - start_time_iter))

                zeta = zeta_next # zeta_next = zeta + dzeta
                trajectory.append((psi_t_next[0], theta_next[0], zeta_next))
                # print(i, np.mod(zeta, np.pi))

                # check if the next iterate falls outside the vessel wall
                # x_next = np.sqrt(psi_t_next[0] / (np.pi * Bc))*np.cos(theta_next[0])
                # y_next = np.sqrt(psi_t_next[0] / (np.pi * Bc))*np.sin(theta_next[0])
                # if x_next**2 + y_next**2 > wall_radius_sq:
                if psi_t_next[0] / (np.pi * Bc) > wall_radius_sq:
                    print(f"For radius = {r:.4f}, field line hits the vessel wall after {i:4d} iterations")
                    wall_hit_flag = True
                    break
                
                # PLOT WHEN ZETA = PI
                if np.abs(zeta - np.pi) < 1.0e-8:
                    # print(zeta)
                    phase_data_zeta_pi.append((psi_t_next[0], theta_next[0], zeta_next))

                # MORE SECTIONS FOR EVERY 45 DEGREES BETWEEN 0 AND 2PI
                # PLOT WHEN ZETA = PI/4
                if np.abs(zeta-(np.pi/4)) < 1.0e-8:
                    phase_data_zeta_pi_over4.append((psi_t_next[0], theta_next[0], zeta_next))
                
                # PLOT WHEN ZETA = PI/2
                if np.abs(zeta-(np.pi/2)) < 1.0e-8:
                    phase_data_zeta_pi_over2.append((psi_t_next[0], theta_next[0], zeta_next))
                
                # PLOT WHEN ZETA = 3PI/4
                if np.abs(zeta-3*(np.pi/4)) < 1.0e-8:
                    phase_data_zeta_3pi_over4.append((psi_t_next[0], theta_next[0], zeta_next))

            # print(psi_t_next, theta_next, zeta)
            phase_data.append((psi_t_next[0], theta_next[0], zeta_next))
            
            if wall_hit_flag:
                break
        
        end_time = time.time()
        print(end_time-start_time)

        # filename to save needs to account for higher resolution runs!!!
        # Saving the trajectory and Poincare section data
        # SAVING PLOT AT ZETA = 0
        np.array(phase_data).dump(open(f"phase_portrait_r={r:.4f}.npy", 'wb'))
        np.savetxt(f"phase_portrait_r={r:.4f}.txt", np.array(phase_data).squeeze(), \
            fmt='%.18e', delimiter=' ', newline='\n')
        
        # SAVING PLOT AT ZETA = PI
        np.array(phase_data_zeta_pi).dump(open(f"phase_portrait_zeta_pi_r={r:.4f}.npy", 'wb'))
        np.savetxt(f"phase_portrait_zeta_pi_r={r:.4f}.txt", np.array(phase_data_zeta_pi).squeeze(), \
            fmt='%.18e', delimiter=' ', newline='\n')

        # SAVING PLOT AT ZETA = PI/4
        np.array(phase_data_zeta_pi_over4).dump(open(f"phase_portrait_zeta_pi_over4_r={r:.4f}.npy", 'wb'))
        np.savetxt(f"phase_portrait_zeta_pi_over4_r={r:.4f}.txt", np.array(phase_data_zeta_pi_over4).squeeze(), \
            fmt='%.18e', delimiter=' ', newline='\n')
        
        # SAVING PLOT AT ZETA = PI/2
        np.array(phase_data_zeta_pi_over2).dump(open(f"phase_portrait_zeta_pi_over2_r={r:.4f}.npy", 'wb'))
        np.savetxt(f"phase_portrait_zeta_pi_over2_r={r:.4f}.txt", np.array(phase_data_zeta_pi_over2).squeeze(), \
            fmt='%.18e', delimiter=' ', newline='\n')
        
        # SAVING PLOT AT ZETA = 3PI/4
        np.array(phase_data_zeta_3pi_over4).dump(open(f"phase_portrait_zeta_3pi_over4_r={r:.4f}.npy", 'wb'))
        np.savetxt(f"phase_portrait_zeta_3pi_over4_r={r:.4f}.txt", np.array(phase_data_zeta_3pi_over4).squeeze(), \
            fmt='%.18e', delimiter=' ', newline='\n')

        # np.array(trajectory).dump(open(f"trajectory_r={r:.2f}.npy", 'wb'))
        # np.savetxt(f"trajectory_r={r:.2f}.txt", np.array(trajectory).squeeze(), \
        #     fmt='%.18e', delimiter=' ', newline='\n')
        print(f"Saved file for radius = {r:.4f}")

    # Plot Phase Portraits
    plt.figure(figsize=(10, 6))
    # for i, data in enumerate(phase_data):
    #     psi_t_vals, theta_vals = zip(*data)
    #     psi_t_vals = np.array(psi_t_vals)
    #     theta_vals = np.array(theta_vals)
    #     x = np.sqrt(psi_t_vals/(np.pi * Bc)) * np.cos(theta_vals)
    #     y = np.sqrt(psi_t_vals/(np.pi * Bc)) * np.sin(theta_vals)
    #     # plt.plot(theta_vals, np.sqrt(iota0_vals), label=f"Radius r={radii[i]:.1f}")
    #     plt.plot(x, y, label=f"Radius r={radii[i]:.1f}")

    # Unzipping the entire phase_data list
    psi_t_vals, theta_vals, zeta_vals = zip(*phase_data)
    psi_t_vals = np.array(psi_t_vals)
    theta_vals = np.array(theta_vals)
    x = np.sqrt(psi_t_vals/(np.pi * Bc)) * np.cos(theta_vals)
    y = np.sqrt(psi_t_vals/(np.pi * Bc)) * np.sin(theta_vals)
    # plt.plot(theta_vals, np.sqrt(iota0_vals), label=f"Radius r={radii[i]:.1f}")
    plt.plot(x, y, '.r', markersize = 1, label=f"r={r:.4f}")

    # plt.xlabel("Poloidal Angle \u03b8 (radians)")
    # plt.ylabel("Toroidal Flux \u03c8")
    plt.title("Phase Portrait in Poloidal Plane")
    # plt.set_aspect('equal')
    # plt.xlim([-2,2])
    # plt.ylim([-2,2])
    plt.gca().set_aspect('equal')
    plt.legend(loc='upper right') 
    plt.grid()  
    plt_filename = f"phase_portrait_fig1a_r={r:.4f}_i={n_iterations:06d}.png" # i___ CHANGES FOR NUMBER OF ITERATIONS
    plt.savefig(plt_filename, bbox_inches='tight', dpi=200)
    plt.close()


    plt.figure(figsize=(10, 6))
    # Unzipping the entire phase_data list
    psi_t_vals, theta_vals, zeta_vals = zip(*phase_data_zeta_pi)
    psi_t_vals = np.array(psi_t_vals)
    theta_vals = np.array(theta_vals)
    x = np.sqrt(psi_t_vals/(np.pi * Bc)) * np.cos(theta_vals)
    y = np.sqrt(psi_t_vals/(np.pi * Bc)) * np.sin(theta_vals)
    # plt.plot(theta_vals, np.sqrt(iota0_vals), label=f"Radius r={radii[i]:.1f}")
    plt.plot(x, y, '.r', markersize = 1, label=f"r={r:.4f}")

    # plt.xlabel("Poloidal Angle \u03b8 (radians)")
    # plt.ylabel("Toroidal Flux \u03c8")
    plt.title("Phase Portrait in Poloidal Plane")
    # plt.set_aspect('equal')
    # plt.xlim([-2,2])
    # plt.ylim([-2,2])
    plt.gca().set_aspect('equal')
    plt.legend(loc='upper right') 
    plt.grid()  
    plt_filename = f"phase_portrait_fig1b_r={r:.4f}_i={n_iterations:06d}.png" # i___ CHANGES FOR NUMBER OF ITERATIONS
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
    plt.plot(x, y, '.r', markersize = 1, label=f"r={r:.4f}")

    # plt.xlabel("Poloidal Angle \u03b8 (radians)")
    # plt.ylabel("Toroidal Flux \u03c8")
    plt.title("Phase Portrait in Poloidal Plane")
    # plt.set_aspect('equal')
    # plt.xlim([-2,2])
    # plt.ylim([-2,2])
    plt.gca().set_aspect('equal')
    plt.legend(loc='upper right') 
    plt.grid()  
    plt_filename = f"phase_portrait_piby4_r={r:.4f}_i={n_iterations:06d}.png" # LINE CHANGES FOR ZETA VALUES # i___ CHANGES FOR NUMBER OF ITERATIONS
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
    plt.plot(x, y, '.r', markersize = 1, label=f"r={r:.4f}")

    # plt.xlabel("Poloidal Angle \u03b8 (radians)")
    # plt.ylabel("Toroidal Flux \u03c8")
    plt.title("Phase Portrait in Poloidal Plane")
    # plt.set_aspect('equal')
    # plt.xlim([-2,2])
    # plt.ylim([-2,2])
    plt.gca().set_aspect('equal')
    plt.legend(loc='upper right') 
    plt.grid()  
    plt_filename = f"phase_portrait_piby2_r={r:.4f}_i={n_iterations:06d}.png" # LINE CHANGES FOR ZETA VALUES # i___ CHANGES FOR NUMBER OF ITERATIONS
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
    plt.plot(x, y, '.r', markersize = 1, label=f"r={r:.4f}")

    # plt.xlabel("Poloidal Angle \u03b8 (radians)")
    # plt.ylabel("Toroidal Flux \u03c8")
    plt.title("Phase Portrait in Poloidal Plane")
    # plt.set_aspect('equal')
    # plt.xlim([-2,2])
    # plt.ylim([-2,2])
    plt.gca().set_aspect('equal')
    plt.legend(loc='upper right') 
    plt.grid()  
    plt_filename = f"phase_portrait_3piby4_r={r:.4f}_i={n_iterations:06d}.png" # LINE CHANGES FOR ZETA VALUES # i___ CHANGES FOR NUMBER OF ITERATIONS
    plt.savefig(plt_filename, bbox_inches='tight', dpi=200)
    plt.close()


# if __name__ == "__main__":
#     if len(sys.argv) == 2:
#         poincare_section_fieldline(radius=float(sys.argv[1]))
#     elif len(sys.argv) == 1:
#         poincare_section_fieldline(radius=0.5)
#     else:
#         print("Incorrect number of arguments!")
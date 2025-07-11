import matplotlib.pyplot as plt
import numpy as np

# path_to_data = './data/zetapi-iterates10000/run3/'
path_to_data = './'
# Radii to plot
# nonuni_radii = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
# nonuni_radii = []
# uni_radii = np.arange(0.87, 0.90, 0.01)
# radii = np.append(nonuni_radii, uni_radii)
# radii_res1 = np.arange(0.05, 0.86, 0.05)
# radii_res2 = np.arange(0.86, 0.901, 0.01)
radii_res1 = []
radii_res2 = []
radii_res3 = np.linspace(0.86, 0.88, 5)
radii = np.append(np.append(radii_res1, radii_res2), radii_res3)

Bc = 1.0/np.pi

plt.figure(figsize=(10, 6))
# Loop through each file, load the data and plot    
for i, r in enumerate(radii):
    # filename = f"phase_portrait_r={r:.4f}.txt"
    filename = f"phase_portrait_zeta_pi_r={r:.4f}.txt"
    array = np.loadtxt(path_to_data + filename)
    psi_t_vals = array[:,0]
    theta_vals = array[:,1]
    x = np.sqrt(psi_t_vals/(np.pi * Bc)) * np.cos(theta_vals)
    y = np.sqrt(psi_t_vals/(np.pi * Bc)) * np.sin(theta_vals)
    plt.figure()
    plt.plot(x, y, '.r', markersize=1, label=f"r={r:.4f}")

    # plt.xlim([-2,2])
    # plt.ylim([-2,2])
    plt.gca().set_aspect('equal')
    plt.legend(loc='upper right')
    plt.grid() 
    # plt_filename = f"phase_portrait_fig1a_r={r:.4f}.png"
    plt_filename = f"phase_portrait_fig1b_r={r:.4f}.png"
    # plt_filename = f"phase_portrait_fig1b_lcfs_v2.png" 
    # plt_filename = f"phase_portrait_fig1b_lcfs_v2.png"
    plt.savefig(plt_filename, bbox_inches='tight', dpi=200)
    plt.close()
    
print("Plots saved successfully.")



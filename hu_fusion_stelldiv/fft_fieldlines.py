import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

import seaborn as sns
sns.set_style("whitegrid")

# Paste the MagneticFieldLineAnalyser class definition here from the previous response
class MagneticFieldLineAnalyser:
    """
    Implements the method from Section II of Boozer, A. H. (2025),
    "Efficient analysis of magnetic field line behavior in toroidal plasmas"
    to analyze magnetic field line trajectories using a
    Gaussian-windowed Fourier transform.

    This class takes a field line trajectory defined by R and Z coordinates
    as a function of the toroidal angle ζ and applies a Gaussian-windowed
    Fourier transform to identify the underlying magnetic surface structure.
    """

    def __init__(self, zeta, R_zeta, Z_zeta):
        """
        Initializes the analyser with the field line trajectory data.

        Args:
            zeta (np.ndarray): 1D array of toroidal angle values, ζ.
            R_zeta (np.ndarray): 1D array of R-coordinate values for each ζ.
            Z_zeta (np.ndarray): 1D array of Z-coordinate values for each ζ.
        """
        if not (zeta.shape == R_zeta.shape == Z_zeta.shape):
            raise ValueError("Input arrays zeta, R_zeta, and Z_zeta must have the same shape.")
        if len(zeta) < 2:
            raise ValueError("Input arrays must contain at least two data points.")

        self.zeta = np.asarray(zeta, dtype=float)
        self.R_zeta = np.asarray(R_zeta, dtype=float)
        self.Z_zeta = np.asarray(Z_zeta, dtype=float)

    def _gaussian_window(self, zeta_vals, lambda_width):
        """
        Calculates the core of the Gaussian window function G(ζ).
        This is the exponential term from Eq. (10). The full
        normalization is handled in the transform calculation.

        Args:
            zeta_vals (np.ndarray): The toroidal angle values to evaluate the window at.
            lambda_width (float): The characteristic width (λ) of the Gaussian window.

        Returns:
            np.ndarray: The values of the Gaussian window function.
        """
        return np.exp(-0.5 * (zeta_vals / lambda_width)**2)

    def calculate_fourier_transform(self, coordinate_vals, omega_range, lambda_width):
        """
        Calculates the Gaussian-windowed Fourier transform Z_f(ω) or R_f(ω).
        This method implements the integral from Eq. (12) using
        a discrete numerical integration (trapezoidal rule), which is
        equivalent to the discrete sum formulation discussed in Appendix A.

        Args:
            coordinate_vals (np.ndarray): The Z(ζ) or R(ζ) data to be transformed.
            omega_range (np.ndarray): A 1D array of frequencies (ω) to analyze.
            lambda_width (float): The width (λ) of the Gaussian window.

        Returns:
            np.ndarray: The Fourier-transformed amplitudes for each frequency in omega_range.
        """
        if lambda_width <= 0:
            raise ValueError("Lambda width must be positive.")

        # Center the zeta array at 0 for the Gaussian window, assuming the
        # integration is centered around the start of the trajectory.
        centered_zeta = self.zeta - self.zeta[0]
        G_zeta = self._gaussian_window(centered_zeta, lambda_width)

        fourier_amplitudes = []
        for omega in omega_range:
            # This is the term inside the integral of Eq. (12)
            integrand = G_zeta * np.cos(omega * centered_zeta) * coordinate_vals
            # The integral is performed numerically.
            integral = np.trapz(integrand, self.zeta)
            fourier_amplitudes.append(2 * integral)

        return np.array(fourier_amplitudes)

    def find_modes(self, coordinate='Z', omega_min=-1.0, omega_max=1.0, num_omega=4000, lambda_width=50.0):
        """
        Performs the full analysis for a given coordinate to find spectral modes.
        It calculates the transform and identifies significant peaks.

        Args:
            coordinate (str): 'Z' or 'R' to specify which coordinate to analyze.
            omega_min (float): Minimum omega to scan.
            omega_max (float): Maximum omega to scan.
            num_omega (int): Number of omega points to compute.
            lambda_width (float): Gaussian window width (λ).

        Returns:
            dict: A dictionary containing the analysis results, including the
                  scanned frequencies, the transformed signal, and the properties
                  of detected peaks.
        """
        if coordinate.upper() == 'Z':
            coord_vals = self.Z_zeta
        elif coordinate.upper() == 'R':
            coord_vals = self.R_zeta
        else:
            raise ValueError("Coordinate must be 'R' or 'Z'")

        omega_range = np.linspace(omega_min, omega_max, num_omega)
        transformed_signal = self.calculate_fourier_transform(coord_vals, omega_range, lambda_width)

        # Find peaks in the absolute value of the transformed signal
        # The height threshold helps to filter out low-amplitude noise.
        peaks, properties = find_peaks(np.abs(transformed_signal), height=np.abs(transformed_signal).max() * 0.05)

        results = {
            'coordinate': coordinate,
            'omega_range': omega_range,
            'transformed_signal': transformed_signal,
            'lambda': lambda_width,
            'peak_indices': peaks,
            'peak_omegas': omega_range[peaks],
            'peak_amplitudes': transformed_signal[peaks]
        }
        return results

    def plot_analysis(self, analysis_results, ax):
        """
        Creates a plot to visualize the results of the Fourier analysis.

        Args:
            analysis_results (dict): The dictionary returned by the find_modes method.
        """
        res = analysis_results
        coord = res['coordinate']
        
        # plt.style.use('seaborn-whitegrid') # THIS IS WHAT THE LINE USED TO BE AND IT DIDNT WORK
        plt.style.use('seaborn-v0_8-whitegrid')

        # fig, ax = plt.subplots(figsize=(14, 7))

        ax.plot(res['omega_range'], res['transformed_signal'], label=f'$\\mathcal{{F}}[{{{coord}}}(\\zeta)]$')
        ax.plot(res['peak_omegas'], res['peak_amplitudes'], 'x', markersize=10, mew=2, label='Detected Peaks')

        ax.set_title(f"Gaussian-Windowed Fourier Transform of the ${coord}$ coordinate ($\\lambda={res['lambda']}$)")
        ax.set_xlabel("Frequency $\\omega = m\\iota_p - n$")
        ax.set_ylabel(f"Fourier Amplitude ${coord}_f(\\omega)$")
        ax.legend()
        plt.show()

        return None

# --- Main Execution ---

# 1. DEFINE TRAJECTORY PARAMETERS
# For this example, assume a tokamak (Np=1 period)
rotational_transform = 0.42  # Example value for ι_p
major_radius = 10.0          # Example R0
minor_radius = 1.0           # Example a

# Define the toroidal angle ζ over a long path to ensure modes can be resolved.
# The method is efficient, but a longer path gives better frequency resolution.
num_turns = 80
num_points = 20000
zeta = np.linspace(0, num_turns * 2 * np.pi, num_points)

# 2. GENERATE SYNTHETIC R(ζ) AND Z(ζ) DATA
# Create a trajectory by summing known Fourier modes (m, n)
# The frequency is given by ω_mn = m * ι_p - n 
#
# Z(ζ) modes:
# Mode 1: m=2, n=1 -> ω = 2 * 0.42 - 1 = -0.16. Amplitude Z_21 = 0.6
# Mode 2: m=3, n=1 -> ω = 3 * 0.42 - 1 = 0.26. Amplitude Z_31 = 0.3
# Mode 3: m=4, n=2 -> ω = 4 * 0.42 - 2 = -0.32. Amplitude Z_42 = 0.2
"""
# THIS IS THE WAY DR NAIK MANUALLY CALCULATED THE TRAJCTORIES
w21 = -0.16
w31 = 0.26
w42 = -0.32
z21 = 0.6
z31 = 0.3
z42 = 0.2
Z_zeta = (z21 * np.cos(w21 * zeta) +
          z31 * np.cos(w31 * zeta) + 
          z42 * np.cos(w42 * zeta))
"""

# R(ζ) modes:
# A large constant term for the major radius plus some oscillations.
# Mode 1: m=1, n=0 -> ω = 1 * 0.42 - 0 = 0.42. Amplitude R_10 = 0.8 (ellipticity)
# Mode 2: m=2, n=1 -> ω = 2 * 0.42 - 1 = -0.16. Amplitude R_21 = 0.2 (same ω as a Z mode)
# Mode 3: m=3, n=2 -> ω = 3 * 0.42 - 2 = -0.74. Amplitude R_32 = 0.1
"""
# THIS IS THE WAY DR NAIK MANUALLY CALCULATED THE TRAJECTORIES
w10 = 0.42
w21 = -0.16
w32 = -0.74
r10 = 0.8
r21 = 0.2
r32 = 0.1
R_zeta = (major_radius +
          r10 * np.cos(w10 * zeta) +
          r21 * np.cos(w21 * zeta) + 
          r32 * np.cos(w32 * zeta))
"""

def flux_to_cylindrical(file_path):
    """
    Loads data from sim_nrd_stel_multiprocessing.py, assumes that the first three columns are (psi_t, theta, zeta),
    and convertes them to cartesian coordinates (x, y).

    Args: file_path (str): The path to the input file.

    Returns: 
        numpy.darray: A NumPy array containing the original data with the 
                      first three columns relaced by x and y.
                      Returns None if the file cannot be loaded or processed.
    """
    try: 
        # Load the data from the file, skipping the header line (ip applicable)
        # Assuming the data is space-separated
        # data = np.loadtxt(file_path, skiprows=1) # my text file does not have any headers to skip
        data = np.loadtxt(file_path)

        # STEP ONE - Extract the first three columns (psi_t, theta, zeta)
        psi_t = data[:, 0]
        theta = data [:, 1]
        zeta = data[:, 2]

        # STEP TWO - Converting to cylinderical coordinates
        R = major_radius + minor_radius*np.sqrt(psi_t)*np.cos(theta)
        Z = np.sqrt(psi_t)*np.sin(theta)
        zeta = zeta

        # STEP THREE - Create a new array with the converted coordinates and the rest of the original data
        # We replace the first three columns with R, Z, theta
        converted_data = np.hstack((R[:, np.newaxis], Z[:, np.newaxis], zeta[:, np.newaxis], data[:, 3:]))

        return converted_data

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def convert_to_cylindrical(file_path):
    """
    Loads data from a file, assumes the first three columns are Flux Coordinates (psi_t, theta, zeta),
    and converts them to cylindrical coordinates (R, Z, zeta).
    # psi_t is a toroidal flux
    # theta is poloidal angle
    # zeta is toroidal angle

    Args:
        file_path (str): The path to the input file.

    Returns:
        numpy.ndarray: A NumPy array containing the original data with the
                       first three columns replaced by R, Z, and zeta.
                       Returns None if the file cannot be loaded or processed.
    """
    try:
        # Load the data from the file, skipping the header line
        # Assuming the data is space-separated
        data = np.loadtxt(file_path, skiprows=1)

        # Extract the first three columns (psi_t, theta, zeta)
        psi_t = data[:, 0]
        theta = data[:, 1]
        zeta = data[:, 2]

        # Convert to cylindrical coordinates
        R = major_radius + np.sqrt(psi_t)*np.cos(theta)
        Z = np.sqrt(psi_t)*np.sin(theta)
        # theta = np.arctan2(y, x)  # Angle in radians
        # Z = z  # Z-coordinate remains the same in cylindrical coordinates

        # Create a new array with the converted coordinates and the rest of the original data
        # We replace the first three columns with R, Z, theta
        converted_data = np.hstack((R[:, np.newaxis], Z[:, np.newaxis], zeta[:, np.newaxis], data[:, 3:]))

        return converted_data

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Example usage with your file:
file_path = "trajectory_r=0.60.txt"

# cylindrical_coordinates_data = convert_to_cylindrical(file_path)
cylindrical_coordinates_data = flux_to_cylindrical(file_path)

# if cylindrical_coordinates_data is not None:
#     print("Conversion successful. Here's a sample of the converted data (R, Z, theta, ...):")
#     print(cylindrical_coordinates_data[:5]) # Print first 5 rows
# else:
#     print("Failed to convert data.")


# --- Step 2: Convert cylindrical coordinates (R, ζ, Z) to Cartesian (X, Y, Z) ---
R_zeta = cylindrical_coordinates_data[:,0]
Z_zeta = cylindrical_coordinates_data[:,1]
zeta = cylindrical_coordinates_data[:,2]


# 3. INSTANTIATE THE ANALYSER
# Create an object with our synthetic trajectory data
analyser = MagneticFieldLineAnalyser(zeta, R_zeta - major_radius, Z_zeta)

# 4. RUN ANALYSIS FOR THE 'Z' COORDINATE
print("--- Analyzing Z(ζ) Coordinate ---")
# Use a sufficiently large lambda to get narrow peaks for this non-chaotic line
z_analysis_results = analyser.find_modes(coordinate='Z', lambda_width=100.0)

# Print the detected peak locations and their amplitudes
# print(f"Expected Z Frequencies (ω): [{w21:02f}, {w31:02f}, {w42:02f}]")
print(f"Detected Z Frequencies (ω): {np.round(z_analysis_results['peak_omegas'], 3)}")
# print(f"\nExpected Z Amplitudes (Z_mn): [{z21:02f}, {z31:02f}, {z42:02f}]")
# Note: The raw amplitude from the transform is related to Z_mn but also depends on lambda.
# From Eq. (14), Z_f(ω) approaches Z_mn at the peak.
print(f"Detected Z Amplitudes: {np.round(z_analysis_results['peak_amplitudes'], 3)}")


# 5. VISUALIZE THE Z ANALYSIS
# The plot should show sharp peaks at ω
fig, ax = plt.subplots(2, 1, figsize=(16, 6))
analyser.plot_analysis(z_analysis_results, ax[0])

# 6. RUN ANALYSIS FOR THE 'R' COORDINATE
print("\n--- Analyzing R(ζ) Coordinate ---")
r_analysis_results = analyser.find_modes(coordinate='R', lambda_width=100.0)

# Print the detected peak locations and their amplitudes
# print(f"Expected R Frequencies (ω): [{w10:02f}, {w21:02f}, {w32:02f}]")
print(f"Detected R Frequencies (ω): {np.round(r_analysis_results['peak_omegas'], 3)}")
# print(f"\nExpected R Amplitudes (R_mn): [{r10:02f}, {r21:02f}, {r32:02f}] (plus large R0 component at ω=0)")
print(f"Detected R Amplitudes: {np.round(r_analysis_results['peak_amplitudes'], 3)}")

# 7. VISUALIZE THE R ANALYSIS
# The plot should show sharp peaks at ω
# The large constant term R0 creates a very large peak at ω=0, which we can ignore for mode analysis.
analyser.plot_analysis(r_analysis_results, ax[1])

fig.subplots_adjust(hspace=0.4) 
fig.savefig('temp.png', dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()
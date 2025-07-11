import numpy as np
import matplotlib.pyplot as plt
import time

# 1. Create Data:
# Define a grid of x and y values
x = np.linspace(-3, 3, 1600)
y = np.linspace(-3, 3, 1600)

# Create a 2D grid from x and y for evaluating the function
start = time.time()
X, Y = np.meshgrid(x, y)
end = time.time()
print(end-start)

# Define a 2D function (e.g., a "sombrero" function or a simple quadratic)
start2 = time.time()
Z = np.sin(X**2 + Y**2) / (X**2 + Y**2)  # A bit like a sombrero function with a singularity at (0,0)
end2 = time.time()
print(end2-start2)
# Handle potential division by zero for the exact center if necessary (or use a small epsilon)
# For this function, as X,Y -> 0, Z -> 1, so let's adjust for numerical stability
Z[np.isnan(Z)] = 1.0 # Set NaN values (from division by zero) to 1.0

# 2. Create the Plot:
plt.figure(figsize=(8, 6))
# Plot contour lines
CS2 = plt.contourf(X, Y, Z, cmap='bwr', levels = 20)
#CS = plt.contour(X, Y, Z, colors='black', linestyles='dashed', linewidths=0.8)

# Add labels to the contour lines
plt.clabel(CS2, inline=True, fontsize=8)

# Add a title and labels
plt.title('Basic Contour Plot of sin(r^2)/r^2')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Show the plot
plt.grid(True, linestyle=':', alpha=0.6)
plt.colorbar(label='Z-value') # Add a color bar to indicate the Z values for contour lines
plt.show()
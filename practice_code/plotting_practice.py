import numpy as np 
import matplotlib.pyplot as plt 

## GRAPH ONE USING X AND Y COORDINATES
x = np.arange(0,100.5,0.5) # generate x-axis values
y = 2.0*np.sqrt(x) # calculate y-values
plt.plot(x,y) # plot data onto axes
    # the plt.plot() function performs three separate steps: 
    # (1) fig = plt.figure()
    # (2) ax = fig.add_axes()
    # (3) ax.plot(x,y)
# same as doing the following: 
# (1) fig = plt.figure()
# (2) ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
# (3) ax.plot(x,y)
plt.show() # display plot to the screen

## GRAPH TWO USING X AND Y COORDINATES
x = np.linspace(0, 2*np.pi)
y= np.sin(x)
plt.plot(x,y)
plt.ylim(-1.1, 1.1)
plt.xlim(0, 2*np.pi)
plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
           [r'$0$', r'$\frac{\pi}{2}$', r'$\pi$', 
            r'$\frac{3\pi}{2}$', r'$2\pi$'], 
           color = 'b', size = 'x-large')
plt.show()

## GRAPH THREE USING POLAR COORDINATES
theta = np.linspace(0, 8*np.pi, 100)
r = np.linspace(0, 20, 100)
plt.polar(theta, r, 's--') # using angle 'theta' and radial distance 'r'
plt.yticks(range(5, 25, 5))
plt.show()

## GRAPH - SHARING AXES AMONG SUBPLOTS
x = np.linspace(0, 2*np.pi, 100) # x-axis values
fig, ax = plt.subplots(2, 2, sharex = True, sharey = True)
ax[0,0].plot(x, np.cos(x))
ax[0,1].plot(x, np.sin(x))
ax[1,0].plot(x, np.cos(2*x))
ax[1,1].plot(x, np.sin(2*x))
ax[0,0].set_xlim(0, 2*np.pi) # set x-axis limits
plt.show()
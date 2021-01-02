import math
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

## define a function to output the probability densities 
def bi_gaussian_pdf(x, miu, sigma):
    
    p_intial = (np.exp(((x - miu).transpose()) @ (np.linalg.inv(sigma)) @ 
        (x - miu) / (-2))) / (2 * math.pi * math.sqrt(np.linalg.det(sigma)))

    p = np.diagonal(p_intial)

    return p

x = np.random.rand(2, 10000) # set a random x
miu = np.mean(x, axis=1).reshape(2, 1) # compute the average of x in each row
sigma = np.cov(x) # compute the covariance matrix

p = bi_gaussian_pdf(x, miu, sigma)

## draw a 3D triangular surface of p
fig = plt.figure()

ax = Axes3D(fig) 
surf = ax.plot_trisurf(x[0], x[1], p, cmap = cm.coolwarm, alpha = 0.5)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.title("Bivariate Gaussian Probability Density Surface")

## draw the three ellipsoid surface 
fig = plt.figure()

# redefine x, y, z grids by using interpolation
grid_x,grid_y = np.mgrid[np.min(x[0]):np.max(x[0]):0.001, np.min(x[1]):np.max(x[1]):0.001]
grid_z = griddata(x.transpose(), p.transpose(), (grid_x, grid_y), method='nearest')
a = np.max(p) # compute the maxmium of p to compute pencentiles
cs = plt.contour(grid_x, grid_y ,grid_z, [0.1 * a, 0.5 * a, 0.9 * a], alpha = 0.75, cmap = cm.jet)
plt.colorbar()
plt.title("10th, 50th, 90th percentiles of the probability densities")

plt.show()
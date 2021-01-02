import math
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

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

fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_trisurf(x[0], x[1], p, cmap = cm.coolwarm, alpha = 0.5)
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
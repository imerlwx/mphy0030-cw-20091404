import math
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

## define a function to output the probability densities 
def gaussian_pdf(x, miu, sigma):
    
    p_intial = (np.exp(((x - miu).transpose()) @ (np.linalg.inv(sigma)) @ 
        (x - miu) / (-2))) / (((2 * math.pi) ** 1.5) * math.sqrt(np.linalg.det(sigma)))

    p = np.diagonal(p_intial)

    return p

x = np.random.rand(3, 10000) # set a random x
miu = np.mean(x, axis=1).reshape(3, 1) # compute the average of x in each row
sigma = np.cov(x) # compute the covariance matrix

p = gaussian_pdf(x, miu, sigma)

def plot_ellipoid(x, p, percentiles, fig, ax):

    min_p = percentiles * np.max(p) - 0.01
    max_p = percentiles * np.max(p) + 0.01

    m = np.where((p >= min_p) & (p < max_p))
    n = np.array(m)
    xi = np.zeros((n.shape[1], 3))

    i = 0
    while i < n.shape[1]:
        xi[i] = x[:, n[:, i]].T
        i += 1 

    a = (np.max(xi[:, 0]) - np.min(xi[:, 0])) / 2
    b = (np.max(xi[:, 1]) - np.min(xi[:, 1])) / 2
    c = (np.max(xi[:, 2]) - np.min(xi[:, 2])) / 2

    a1 = (np.max(xi[:, 0]) + np.min(xi[:, 0])) / 2
    b1 = (np.max(xi[:, 1]) + np.min(xi[:, 1])) / 2
    c1 = (np.max(xi[:, 2]) + np.min(xi[:, 2])) / 2

    
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = a * np.outer(np.cos(u), np.sin(v)) + a1
    y = b * np.outer(np.sin(u), np.sin(v)) + b1
    z = c * np.outer(np.ones(np.size(u)), np.cos(v)) + c1

    ax.plot_surface(x, y, z, color='b',cmap=cm.coolwarm)


## draw the three ellipsoid surface 
fig = plt.figure()
ax = fig.subplots(111, projection='3d')
plot_ellipoid(x, p, 0.5, fig, ax)
plt.title("10th, 50th, 90th percentiles of the probability densities")

plt.show()
import numpy as np
from scipy.spatial import cKDTree as kdtree
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

## define a function to implement lowpass smoothing algorithm
def low_pass_smoothing(vertices, iterations, lambda1, mu):

    n = 0
    
    vertices_smoothed = np.zeros(np.shape(vertices))
    
    while n < iterations:
        for i in range(vertices.shape[0]):
            
            _, idx = kdtree.query(vertices[i], vertices, k = 1)
            
            q = np.zeros((1, 3)) # set the intial sum of nearest points
            m = len(idx) # number of nearest points
        
            for j in range(m):
                q += vertices[idx[j]]
            
            if n % 2 == 0:
                vertices_smoothed[i] = vertices[i] + lambda1 * (q / m - vertices[i])
            else:
                vertices_smoothed[i] = vertices[i] + mu * (q / m - vertices[i])

        vertices = vertices_smoothed

    return vertices

vertices = np.genfromtxt('../data/example_vertices.csv',delimiter=',')
triangles = np.genfromtxt('../data/example_triangles.csv',delimiter=',')

# plot the first figure before smoothing
fig = plt.figure()
ax = fig.gca(projection='3d')
plt.subplot(1, 2, 1)
ax.plot_trisurf(vertices[:,0], vertices[:,1], vertices[:,2], triangles = triangles, color='b')

# set up parameters
iterations = 10
lambda1 = 0.9
mu = -1.02

vertices = low_pass_smoothing(vertices, iterations, lambda1, mu)

# plot the second figure after smoothing
fig = plt.figure()
ax = fig.gca(projection='3d')
plt.subplot(1, 2, 2)
ax.plot_trisurf(vertices[:,0], vertices[:,1], vertices[:,2], triangles = triangles, color='r')

plt.show()
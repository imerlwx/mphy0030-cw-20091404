import numpy as np
from scipy.spatial import KDTree as kdtree
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

## define a function to implement lowpass smoothing algorithm
def low_pass_smoothing(vertices, iterations, lambda1, mu):
    
    vertices_smoothed = np.zeros(np.shape(vertices))
    
    for n in range(iterations):
        for i in range(vertices.shape[0]):
            
            vertices_tree = kdtree(vertices)
            _, idx = vertices_tree.query(vertices[i], k = 7)
            
            q = np.zeros((1, 3)) # set the initial sum of nearest points
            m = len(idx) # number of nearest points (actually minus 1 because this including the point itself)
        
            for j in range(m):
                q += vertices[idx[j]]
            
            if n % 2 == 0:
                vertices_smoothed[i] = vertices[i] + (q - m * vertices[i]) * lambda1 / (m - 1)
            else:
                vertices_smoothed[i] = vertices[i] + (q - m * vertices[i]) * mu / (m - 1)

        vertices = vertices_smoothed

    return vertices

vertices_intial = np.genfromtxt('../data/example_vertices.csv',delimiter=',')
triangles = np.genfromtxt('../data/example_triangles.csv',delimiter=',') - 1

# set up parameters
lambda1 = 0.9
mu = -1.02 * lambda1

# plot the first figure before smoothing
fig = plt.figure()
ax = fig.add_subplot(221, projection='3d')
ax.plot_trisurf(vertices_intial[:,0], vertices_intial[:,1], vertices_intial[:,2], triangles = triangles, color='b')
plt.title('before filtering')

# plot the second figure after 5 iterations' smoothing
ax = fig.add_subplot(222, projection='3d')
vertices = low_pass_smoothing(vertices_intial, 5, lambda1, mu)
ax.plot_trisurf(vertices[:,0], vertices[:,1], vertices[:,2], triangles = triangles, color='r')
plt.title('after 5 iterations filtering')

# plot the second figure after 10 iterations' smoothing 
ax = fig.add_subplot(223, projection='3d')
vertices = low_pass_smoothing(vertices_intial, 10, lambda1, mu)
ax.plot_trisurf(vertices[:,0], vertices[:,1], vertices[:,2], triangles = triangles, color='r')
plt.title('after 10 iterations filtering')

# plot the second figure after 25 iterations' smoothing 
ax = fig.add_subplot(224, projection='3d')
vertices = low_pass_smoothing(vertices_intial, 25, lambda1, mu)
ax.plot_trisurf(vertices[:,0], vertices[:,1], vertices[:,2], triangles = triangles, color='r')
plt.title('after 25 iterations filtering')

#plt.show()
plt.savefig('task4.png')
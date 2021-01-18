import numpy as np
from gs import RBFSpline

## define a class to store different characteristics of 3D images
class Image3D():
    
    def __init__(self, data):
        
        self.vox_dimension = data['voxdims']
        self.image_size = np.shape(data['vol'])
        self.data_type = data['voxdims'].dtype


## define a class to implement the free form deformation
class FreeFormDeformation():

    # precompute the coordinates of all the control points
    def __init__(self, Nx, Ny, Nz, min_x, max_x, min_y, max_y, min_z, max_z):
        
        # compute the x, y, z coordinates of control points
        self.points_num = Nx * Ny * Nz
        self.dx = (max_x - min_x) / Nx
        self.dy = (max_y - min_y) / Ny
        self.dz = (max_z - min_z) / Nz
       
        self.x_lattice, self.y_lattice, self.z_lattice = np.mgrid[min_x:max_x:Nx, min_y:max_y:Ny, min_z:max_z:Nz]
        self.x = self.x_lattice.reshape(self.points_num, 1)
        self.y = self.y_lattice.reshape(self.points_num, 1)
        self.z = self.z_lattice.reshape(self.points_num, 1)

        self.control_points = np.hstack((self.x, self.y, self.z))

    

    # define a function to compute the 
    def random_transform_generator(self, randomness):

        # a random displacement with constraint 
        displacement_x = randomness * (2 * np.random.rand(self.points_num, 1) - 1) * self.dx / 2
        displacement_y = randomness * (2 * np.random.rand(self.points_num, 1) - 1) * self.dy / 2
        displacement_z = randomness * (2 * np.random.rand(self.points_num, 1) - 1) * self.dz / 2
        displacement = np.hstack((displacement_x, displacement_y, displacement_z))
        
        transformed_control_points = self.control_points + displacement

        return transformed_control_points

    # define a function to compute a warped 3D image
    def warp_image(self, Image3D, RBFSpline):

        voxdims = Image3D.vox_dimension
        shape = Image3D.image_size
        
        # compute the coordinates of query points
        x_max = voxdims[0] * shape[0]
        y_max = voxdims[1] * shape[1]
        z_max = voxdims[2] * shape[2]

        query_x_lattice, query_y_lattice, query_z_lattice = np.mgrid[0:x_max:voxdims[0], 0:y_max:voxdims[1], 0:z_max:voxdims[2]]
        query_x = query_x_lattice.reshape(shape[0] * shape[1] * shape[2], 1)
        query_y = query_y_lattice.reshape(shape[0] * shape[1] * shape[2], 1)
        query_z = query_z_lattice.reshape(shape[0] * shape[1] * shape[2], 1)

        query_points = np.hstack((query_x, query_y, query_z))

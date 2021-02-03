import numpy as np
from scipy.interpolate import interpn
from matplotlib import pyplot as plt
from gs import RBFSpline

## define a class to store different characteristics of 3D images
class Image3D():
    
    def __init__(self, data):
        
        self.intensity = data['vol']
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
       
        # construct the control points lattice
        self.x_lattice, self.y_lattice, self.z_lattice = np.mgrid[min_x:max_x:self.dx, min_y:max_y:self.dy, min_z:max_z:self.dz]
        self.x = self.x_lattice.reshape(self.points_num, 1) # reshape the lattice to get the coordinate of each control point
        self.y = self.y_lattice.reshape(self.points_num, 1)
        self.z = self.z_lattice.reshape(self.points_num, 1)

        self.control_points = np.hstack((self.x, self.y, self.z))

    

    # define a function to compute the 
    def random_transform_generator(self, randomness):

        # define an affine transformation matrix
        Maff = np.eye(4)
        Maff[0,0] = 1 + 0.05 * np.random.randn(1) * randomness
        Maff[0,1] = 0.05 * np.random.randn(1) * randomness
        Maff[0,2] = 0.05 * np.random.randn(1) * randomness
        Maff[1,0] = 0.05 * np.random.randn(1) * randomness
        Maff[1,1] = 1 + 0.05 * np.random.randn(1) * randomness
        Maff[1,2] = 0.05 * np.random.randn(1) * randomness
        Maff[2,0] = 0.05 * np.random.randn(1) * randomness
        Maff[2,1] = 0.05 * np.random.randn(1) * randomness
        Maff[2,2] = 1 * randomness
        Maff[0,3] = 10 * np.random.randn(1) * randomness
        Maff[1,3] = 10 * np.random.randn(1) * randomness

        # calculate the transformed control points
        transformation = Maff @ np.concatenate((self.control_points, np.ones((self.control_points.shape[0], 1))), axis=1).T
        transformed_control_points = transformation[0:3,:].T

        return transformed_control_points

    # define a function to compute a warped 3D image
    def warp_image(self, Image3D, RBFSpline, randomness, lambda1, sigma):

        voxdims = Image3D.vox_dimension
        shape = Image3D.image_size
        
        # compute the coordinates of query points
        x_max = voxdims[:, 0] * shape[0]
        y_max = voxdims[:, 1] * shape[1]
        z_max = voxdims[:, 2] * shape[2]

        query_x_lattice, query_y_lattice, query_z_lattice = np.mgrid[0:x_max:voxdims[:, 0], 0:y_max:voxdims[:, 1], 0:z_max:voxdims[:, 2]]
        query_x = query_x_lattice.reshape(shape[0] * shape[1] * shape[2], 1)
        query_y = query_y_lattice.reshape(shape[0] * shape[1] * shape[2], 1)
        query_z = query_z_lattice.reshape(shape[0] * shape[1] * shape[2], 1)

        query_points = np.hstack((query_x, query_y, query_z))

        transformed_control_points = self.random_transform_generator(randomness) # first compute transformed control points

        alpha = RBFSpline.fit(self.control_points, transformed_control_points, lambda1, sigma) # then fit alpha

        transformed_query_points = RBFSpline.evaluate(query_points, self.control_points, alpha, sigma) # lastly evalute query points

        transformed_x_lattice = transformed_query_points[:, 0].reshape(shape[0], shape[1], shape[2])
        transformed_y_lattice = transformed_query_points[:, 1].reshape(shape[0], shape[1], shape[2])
        transformed_z_lattice = transformed_query_points[:, 2].reshape(shape[0], shape[1], shape[2])

        return transformed_x_lattice, transformed_y_lattice, transformed_z_lattice

    ## define a function to output randomly warped 3D images
    def random_transform(self, Image3D, RBFSpline, randomness, sigma, lambda1, z):

        transformed_x_lattice, transformed_y_lattice, _ = self.warp_image(Image3D, RBFSpline, randomness, lambda1, sigma)

        # get the points set from original image
        points_i = np.arange(0, Image3D.intensity.shape[0]) * Image3D.vox_dimension[:, 0]
        points_j = np.arange(0, Image3D.intensity.shape[1]) * Image3D.vox_dimension[:, 1]
        points = (points_i, points_j)

        values = Image3D.intensity[:, :, z] # get the value of original image

        # get the points set from transformed image
        transformed_points_i = transformed_x_lattice[:, :, z]
        transformed_points_j = transformed_y_lattice[:, :, z]
        transformed_points = np.concatenate((transformed_points_i.reshape(-1, 1), transformed_points_j.reshape(-1, 1)), axis=1)

        # get the warped image
        image_interpn_flatten = interpn(points, values, transformed_points, bounds_error=False, fill_value=0)
        image_interpn = image_interpn_flatten.reshape(Image3D.image_size[0], Image3D.image_size[1])

        return image_interpn
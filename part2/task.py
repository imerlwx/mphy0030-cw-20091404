import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interpn
import scipy.io as scio

class RBFSpline():

    # define a function to get alpha
    def fit(self, source_points, target_points, lambda1, sigma):

        p = source_points
        q = target_points

        n = p.shape[0]  # number of source points in each dimension  
        W = np.identity(n) # matrix for landmark localization errors
        K = np.zeros((n, n)) # initial kernel value of source points

        # compute the kernel value K
        K = self.kernel_gaussian(p, p, sigma) 
        
        # use the linear least square algorithm to compute alpha
        A = K + lambda1 * W
        U, S, VT = np.linalg.svd(A)
        a = np.linalg.inv(np.diag(S) @ VT) @ (U.T) @ q[:, 0]
        b = np.linalg.inv(np.diag(S) @ VT) @ (U.T) @ q[:, 1]
        c = np.linalg.inv(np.diag(S) @ VT) @ (U.T) @ q[:, 2]
        alpha = np.hstack((a.reshape(-1, 1), b.reshape(-1, 1), c.reshape(-1, 1)))
        
        return alpha

    # define a funtion to evaluate the query points to the transformed version
    def evaluate(self, query_points, control_points, alpha, sigma):

        m = query_points.shape[0] # number of query points in each dimension
        n = control_points.shape[0] # number of control points in each dimension
        K = np.zeros((m, n)) # initial kernel value between query points and control points
        transformation = np.zeros((m, 3)) # initial transformation
        
        # compute kernel values between query points and control points
        K = self.kernel_gaussian(query_points, control_points, sigma) 
        
        # compute the transformed query points
        for i in range(3):
            transformation[:, i] = K @ alpha[:, i]
            
        transformed_query_points = transformation

        return transformed_query_points

    # define a function to output kernel values between query points and control points
    def kernel_gaussian(self, query_points, control_points, sigma):
       
        # use the vectorization method to compute the squared distance
        first = np.sum(np.square(query_points), axis=1).reshape(-1, 1)
        second = np.sum(np.square(control_points), axis=1).reshape(1, -1)
        third = - 2 * query_points @ (control_points.transpose())

        r_squared = first + second + third
        K = np.exp( - r_squared / (2 * sigma ** 2))
        
        return K

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

    @classmethod
    def opcons(cls, Image3D, Nx, Ny, Nz):
        
        points_num = Nx * Ny * Nz
        max_x = Image3D.image_size[0] * Image3D.vox_dimension[:, 0]
        max_y = Image3D.image_size[1] * Image3D.vox_dimension[:, 1]
        max_z = Image3D.image_size[2] * Image3D.vox_dimension[:, 2]

        dx = max_x / Nx
        dy = max_y / Ny
        dz = max_z / Nz

        x_lattice, y_lattice, z_lattice = np.mgrid[0:max_x:dx, 0:max_y:dy, 0:max_z:dz]

        # reshape the lattice to get the coordinate of each control point
        x = x_lattice.reshape(points_num, 1) 
        y = y_lattice.reshape(points_num, 1)
        z = z_lattice.reshape(points_num, 1)

        b = np.hstack((x, y, z))
        control_points = cls(b)
        
        return control_points

    # define a function to compute the 
    def random_transform_generator(self, randomness):

        # define an affine transformation matrix
        Maff = np.eye(4)
        Maff[0,0] = 1 + 0.1 * np.random.randn(1) * randomness
        Maff[0,1] = 0.1 * np.random.randn(1) * randomness
        Maff[0,2] = 0.1 * np.random.randn(1) * randomness
        Maff[1,0] = 0.1 * np.random.randn(1) * randomness
        Maff[1,1] = 1 + 0.1 * np.random.randn(1) * randomness
        Maff[1,2] = 0.1 * np.random.randn(1) * randomness
        Maff[2,0] = 0.1 * np.random.randn(1) * randomness
        Maff[2,1] = 0.1 * np.random.randn(1) * randomness
        Maff[2,2] = 1
        Maff[0,3] = 10 * np.random.randn(1) * randomness
        Maff[1,3] = 10 * np.random.randn(1) * randomness

        # calculate the transformed control points
        transformation = Maff @ np.concatenate((self.control_points, np.ones((self.control_points.shape[0], 1))), axis=1).T
        transformed_control_points = transformation[0:3,:].T

        return transformed_control_points

    # define a function to compute a warped 3D image
    def warp_image(self, Image3D, RBFSpline, transformed_control_points, lambda1, sigma):

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

        alpha = RBFSpline.fit(self.control_points, transformed_control_points, lambda1, sigma) # then fit alpha

        transformed_query_points = RBFSpline.evaluate(query_points, self.control_points, alpha, sigma) # lastly evalute query points

        transformed_x_lattice = transformed_query_points[:, 0].reshape(shape[0], shape[1], shape[2])
        transformed_y_lattice = transformed_query_points[:, 1].reshape(shape[0], shape[1], shape[2])
        transformed_z_lattice = transformed_query_points[:, 2].reshape(shape[0], shape[1], shape[2])

        return transformed_x_lattice, transformed_y_lattice, transformed_z_lattice

    ## define a function to output randomly warped 3D images
    def random_transform(self, Image3D, RBFSpline, randomness, sigma, lambda1, z):

        transformed_control_points = self.random_transform_generator(randomness) # first compute transformed control points

        transformed_x_lattice, transformed_y_lattice, _ = self.warp_image(Image3D, RBFSpline, transformed_control_points, lambda1, sigma)

        # get the points set from original image
        points_i = np.arange(0, Image3D.intensity.shape[0]) * Image3D.vox_dimension[:, 0]
        points_j = np.arange(0, Image3D.intensity.shape[1]) * Image3D.vox_dimension[:, 1]
        points = (points_i, points_j)

        fig, ax = plt.subplots(1, len(z))

        # get the points set from transformed image
        for i in range(len(z)):
            
            values = Image3D.intensity[:, :, z[i]] # get the value of original image
            
            transformed_points_i = transformed_x_lattice[:, :, z[i]]
            transformed_points_j = transformed_y_lattice[:, :, z[i]]
            transformed_points = np.concatenate((transformed_points_i.reshape(-1, 1), transformed_points_j.reshape(-1, 1)), axis=1)
            
            # get the warped image
            image_interpn_flatten = interpn(points, values, transformed_points, bounds_error=False, fill_value=0)
            image_interpn = image_interpn_flatten.reshape(Image3D.image_size[0], Image3D.image_size[1])

            # show the result
            ax[i].imshow(image_interpn, cmap='gray')
            ax[i].title.set_text(f"z = {z[i]}")
            
        plt.show()

# get data from a mat file
dataFile = '../data/example_image.mat'
data = scio.loadmat(dataFile)

image = Image3D(data)
gs_spline = RBFSpline()

randomness = 1
sigma = 100
lambda1 = 0.01
z = [5, 10, 15, 20, 25]

# max should be less than voxdim * vox shape
a = FreeFormDeformation(4, 4, 4, 0, 200, 0, 200, 10, 90)

a.random_transform(image, gs_spline, randomness, sigma, lambda1, z)


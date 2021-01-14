## class file for Gaussian spline

import numpy as np


class RBFSpline():

    # define a function to get alpha
    def fit(self, source_points, target_points, lambda1, sigma):

        p = source_points
        q = target_points

        n = np.shape(p)[0]  # number of source points in each dimension  
        W = np.identity(n) # matrix for landmark localization errors
        K = np.zeros((n, n, 3)) # initial kernel value of source points
        alpha = np.zeros((n,3)) # initialize alpha

        # compute the kernel value K
        for i in range(3):
            K[:, :, i] = self.kernel_gaussian(p[:, i], p[:, i], sigma) 
        
        # use the leaner least square algorithm to compute alpha
        for i in range(3):
            alpha[:, 0] = np.linalg.lstsq(K[:, :, i] + lambda1 @ W, q[:, i], rcond = None)
        
        return alpha

    # define a funtion to evaluate the query points to the transformed version
    def evaluate(self, query_points, control_points, alpha, sigma):

        m = np.shape(query_points)[0] # number of query points in each dimension
        n = np.shape(control_points)[0] # number of control points in each dimension
        K = np.zeros((m, n, 3)) # initial kernel value between query points and control points
        transformed_query_points = np.zeros((m, 3)) # initial query points after transformation
        
        for i in range(3):
            K[:, :, i] = self.kernel_gaussian(query_points[:, i], control_points[:, i], sigma) 
        
        for i in range(3):
            transformed_query_points[:, i] = K[:, :, 1] @ alpha[:, i]

        return transformed_query_points

    def kernel_gaussian(self, query_points, control_points, sigma):

        r = np.linalg.norm(query_points - control_points)
        K = np.exp( - r ** 2 / (2 * sigma ** 2))
        
        return K

        #[px, py, pz] = [p[:,0], p[:,1], p[:,2]]
        #[qx, qy, qz] = [q[:,0], q[:,1], q[:,2]]
        #sigma = np.zeros((np.shape(p))) # set an intial array for sigma 
        #m = np.size(sigma)
        #alpha = np.linalg.inv(K + lambda1 * np.linalg.inv(W)) @ q
        # compute the landmarks localization errors sigma
        #for n in range(m):
            #for i in range(m):
                #sigma[n] += np.sum((p[n] - p[i]) ** 2 / (m - 1))
        #K[:, :, 1] = self.kernel_gaussian(p[:, 1], p[:, 1], sigma)
        #K[:, :, 2] = self.kernel_gaussian(p[:, 2], p[:, 2], sigma)
        #alpha[:, 1] = np.linalg.lstsq(Ky + lambda1 @ W, q[:, 1], rcond = None)
        #alpha[:, 2] = np.linalg.lstsq(Kz + lambda1 @ W, q[:, 2], rcond = None)

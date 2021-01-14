## class file for Gaussian spline

import numpy as np


class RBFSpline():

    # define a function to get alpha
    def fit(self, source_points, target_points, lambda1, sigma):

        p = source_points
        q = target_points

        n = p.shape[0]  # number of source points in each dimension  
        W = np.identity(n) # matrix for landmark localization errors
        K = np.zeros((n, n)) # initial kernel value of source points
        alpha = np.zeros((n,3)) # initialize alpha

        # compute the kernel value K
        K = self.kernel_gaussian(p, p, sigma) 
        
        # use the leaner least square algorithm to compute alpha
        for i in range(3):
            alpha[:, 0] = np.linalg.lstsq(K + lambda1 @ W, q[:, i], rcond = None)
        
        return alpha

    # define a funtion to evaluate the query points to the transformed version
    def evaluate(self, query_points, control_points, alpha, sigma):

        m = query_points.shape[0] # number of query points in each dimension
        n = control_points.shape[0] # number of control points in each dimension
        K = np.zeros((m, n)) # initial kernel value between query points and control points
        transformed_query_points = np.zeros((m, 3)) # initial query points after transformation
        
        # compute kernel values between query points and control points
        K = self.kernel_gaussian(query_points, control_points, sigma) 
        
        for i in range(3):
            transformed_query_points[:, i] = K @ alpha[:, i]

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

        

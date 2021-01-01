import math
import numpy as np
from matplotlib import pyplot as plt

## define a function to output the probability densities 
def bi_gaussian_pdf(x, miu, sigma):
    
    p = (np.exp(-((x - miu).transpose()) @ (np.linalg.inv(sigma)) @ 
        (x - miu) / 2)) / (2 * math.pi * math.sqrt(np.linalg.det(sigma)))

    return p

x = np.random.rand(2, 1000) # set a random x
miu = np.mean(x, axis=1).reshape(2, 1) # compute the average of x in each row
n = x.shape[1] # get the number of column in x

# compute the covariance matrix
sigma = np.zeros((2,2))
sigma_x1 = np.sum((x[0] - miu[0]) ** 2) / (n - 1)
sigma_x2 = np.sum((x[1] - miu[1]) ** 2) / (n - 1)
sigma_x1x2 = np.sum((x[0] - miu[0]) * (x[1] - miu[1])) / (n - 1)
sigma[0, 0] = sigma_x1
sigma[0, 1] = sigma_x1x2
sigma[1, 0] = sigma_x1x2
sigma[1, 1] = sigma_x2

p = bi_gaussian_pdf(x, miu, sigma)


plt.plot(x[0], p[0, :])

#fig = plt.figure()
#ax = fig.gca(projection='3d')
#X, Y = np.meshgrid(x[0], x[1])
#ax.plot_surface(X, Y, p)
#ax.legend()
plt.show()
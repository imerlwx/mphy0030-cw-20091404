import numpy as np
import scipy.io as scio
from matplotlib import pyplot as plt
from ffd import Image3D
from ffd import FreeFormDeformation
from gs import RBFSpline

# get data from a mat file
dataFile = '../data/example_image.mat'
data = scio.loadmat(dataFile)

image = Image3D(data)
gs_spline = RBFSpline()

randomness = 0.5
sigma = 100
lambda1 = 0.01

# max should be less than voxdim * vox shape
a = FreeFormDeformation(4, 4, 4, 0, 200, 0, 200, 10, 90)

image_interpn = a.random_transform(image, gs_spline, randomness, sigma, lambda1, 5)

# show the result
fig, ax = plt.subplots(1, 2)
ax[0].imshow(image.intensity[:, :, 5], cmap='gray')
ax[1].imshow(image_interpn, cmap='gray')
ax[0].title.set_text('Original')
ax[1].title.set_text('sigma = 100')

plt.show()
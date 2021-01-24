import numpy as np
import scipy.io as scio
from ffd import Image3D
from ffd import FreeFormDeformation
from gs import RBFSpline

# get data from a mat file
dataFile = './data/example_image.mat'
data = scio.loadmat(dataFile)

image = Image3D(data)
gs_spline = RBFSpline()

randomness = 0
sigma = 1
lambda1 = 0.01

# max should be less than voxdim * vox shape
a = FreeFormDeformation(4, 4, 4, 110, 150, 110, 150, 20, 90)

a.random_transform(image, gs_spline, randomness, sigma, lambda1, 5)
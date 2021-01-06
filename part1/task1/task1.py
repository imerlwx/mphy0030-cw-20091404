import numpy as np
import scipy.io as scio
import struct
from matplotlib import pyplot as plt

## write any file into a binary file
def simple_image_write(data):
    
    [intensity, vox_dimension] = [data['vol'], data['voxdims']] # read information from data
    
    # open a new file to write in binary format
    with open('./data/image.sim', 'wb+') as fw:
        
        # store the image size to the header of file
        image_size = struct.pack('i', np.size(intensity))
        fw.write(image_size)
        
        a = np.shape(intensity)  # get the shape of image
        
        # store the intensity value one by one to the file
        for i in range(a[0]):
            for j in range(a[1]):
                for k in range(a[2]):
                    s1 = struct.pack('h', intensity[i, j, k])
                    fw.write(s1)
    
        b = np.shape(vox_dimension) # get the dimension of voxel dimensions

        # store the voxel dimension one by one to the file
        for i in range(b[0]):
            for j in range(b[1]):
                s2 = struct.pack('f', vox_dimension[i, j])
                fw.write(s2)

    return a, b

## read any binary file
def simple_image_read(file_name, a, b):

    with open(file_name, 'rb+') as fr:
        
        # read the image size first
        image_size, = struct.unpack('i', fr.read(4))

        # read intensity value as an array
        intensity_read_tuple = struct.unpack('{}h'.format(image_size), fr.read(2 * image_size))
        intensity_read = np.array(intensity_read_tuple)
        intensity = intensity_read.reshape(a)

        # read voxel dimension as an array
        vox_dimension_read_tuple = struct.unpack('{}f'.format(b[0] * b[1]), fr.read(4 * b[0] * b[1]))
        vox_dimension_read = np.array(vox_dimension_read_tuple)
        vox_dimension = vox_dimension_read.reshape(b)

    return intensity, vox_dimension

# get data from a mat file
dataFile = './data/example_image.mat'
data = scio.loadmat(dataFile)

# write and read the binary file
a, b = simple_image_write(data)
intensity, vox_dimension = simple_image_read('./data/image.sim', a, b)    

# plot three images at different z-coodinates
fig = plt.figure()
ax1 = fig.add_subplot(131)
ax1.imshow(intensity[:, :, 5], cmap = plt.cm.gray)
plt.title('z = 5')

ax2 = fig.add_subplot(132)
ax2.imshow(intensity[:, :, 15], cmap = plt.cm.gray)
plt.title('z = 15')

ax2 = fig.add_subplot(133)
ax2.imshow(intensity[:, :, 25], cmap = plt.cm.gray)
plt.title('z = 25')

plt.show()
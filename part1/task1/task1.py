import numpy as np
import scipy.io as scio
import struct

## write any file into a binary file
def simple_image_write(intensity, vox_dimension):
    
    # open a new file to write in binary format
    with open('.//data//image.sim', 'wb+') as fw:
        
        a = np.shape(intensity)  # get the size of image
        
        for i in range(len(a)):
            image_size = struct.pack('h', a)
            fw.write(image_size)
        
        for i in range(a[0]):
            for j in range(a[1]):
                for k in range(a[2]):
                    s1 = struct.pack('h', intensity[i, j, k])
                    fw.write(s1)
    
        b = np.shape(vox_dimension) # get the dimension of voxel dimensions

        for i in range(b[0]):
            for j in range(b[1]):
                s2 = struct.pack('f', vox_dimension[i, j])
                fw.write(s2)


## read any binary file
def simple_image_read(file_name):

    with open('.//data//image.sim', 'rb+') as fr:
        
        image_size = struct.unpack





dataFile = './/data//example_image.mat'
data = scio.loadmat(dataFile)
[intensity, vox_dimension] = [data['vol'], data['voxdims']]
    
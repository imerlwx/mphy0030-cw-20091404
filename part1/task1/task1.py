import numpy as np
import scipy.io as scio
import struct

## write any file into a binary file
def simple_image_write(intensity, vox_dimension):
    
    # open a new file to write in binary format
    with open('.//data//image.sim', 'wb+') as fw:
        
        # store the image size to the header of file
        image_size = struct.pack('i', np.size(intensity))
        fw.write(image_size)
        
        a = np.shape(intensity)  # get the shape of image
        
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

    return a, b

## read any binary file
def simple_image_read(file_name, a, b):

    with open('.//data//image.sim', 'rb+') as fr:
        
        
        image_size, = struct.unpack('i', fr.read(4))

        # read intensity as an array
        intensity_read_tuple = struct.unpack('{}h'.format(image_size), fr.read(2 * image_size))
        intensity_read = np.array(intensity_read_tuple)
        intensity = intensity_read.reshape(a)

        # read voxel dimension as an array
        vox_dimension_read_tuple = struct.unpack('{}f'.format(b[1]), fr.read(4 * b[0] * b[1]))
        vox_dimension_read = np.array(vox_dimension_read_tuple)
        vox_dimension = vox_dimension_read.reshape(b)





dataFile = './/data//example_image.mat'
data = scio.loadmat(dataFile)
[intensity, vox_dimension] = [data['vol'], data['voxdims']]
    
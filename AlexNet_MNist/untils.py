import struct
import numpy as np
def decode_idx3_ubyte(train_image):
    with open(train_image, 'rb') as f:
        print('----file---',train_image)
        f_data = f.read()
    offset = 0
    fmt_header = '>iiii'
    number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, f_data, offset)
    print('num:{}, images_data:{}'.format(number, num_images))
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(num_rows*num_cols) + 'B'
    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        im = struct.unpack_from(fmt_image,f_data, offset)
        images[i] = np.array(im).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    return images

def decode_idx1_ubyte(train_label):
    with open(train_label, 'rb') as f:
        print('----train_label_file---', train_label)
        f_data = f.read()
    offset = 0
    fmt_header = '>ii'
    number, num_label = struct.unpack_from(fmt_header, f_data, offset)
    print('num:{}, images_data:{}'.format(number, num_label))
    offset += struct.calcsize(fmt_header)
    label=[]
    fmt_label = '>B'

    for i in range(num_label):
        label.append(struct.unpack_from(fmt_label, f_data, offset)[0])
        offset += struct.calcsize(fmt_label)
    return label
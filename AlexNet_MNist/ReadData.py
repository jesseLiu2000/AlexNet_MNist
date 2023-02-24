import os
import cv2
from untils import decode_idx3_ubyte, decode_idx1_ubyte
def check(file):
    if not os.path.exists(file):
        os.mkdir(file)
        print(file)
    else:
        if not os.path.isdir(file):
            os.mkdir(file)

def read_image(train_dir, train_image, train_label):
    check(train_dir)
    images = decode_idx3_ubyte(train_image)
    labels = decode_idx1_ubyte(train_label)

    nums = len(labels)
    for i in range(nums):
        img_dir = os.path.join(train_dir, str(labels[i]))
        check(img_dir)
        img_file = os.path.join(img_dir, str(i)+'.png')
        imarr_data = images[i]
        cv2.imwrite(img_file,imarr_data)


def parse_minst(file):
    train_dir = os.path.join(file, 'train')
    train_image = os.path.join(file, 'train-images.idx3-ubyte')
    train_label = os.path.join(file, 'train-labels.idx1-ubyte')
    read_image(train_dir, train_image, train_label)


if __name__ == "__main__":
    file_path='data/train'
    parse_minst(file_path)
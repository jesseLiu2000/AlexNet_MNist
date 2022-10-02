import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np
if __name__ == "__main__":
    #图片格式的预处理，将图片转化为灰度图片且输出矩阵形式，根据矩阵值设置阀值来讲矩阵转化为二值矩阵后输出图片且保存
    image_file_path = os.getcwd() + "/data/pictures/"
    file_path = os.listdir(image_file_path)
    for file_path in file_path:
        img = cv2.imread('data/pictures/'+file_path,)
        Graying = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # print(Graying)
        # print(Graying.shape)
        ret, thresh = cv2.threshold(Graying,150,255,cv2.THRESH_BINARY)
        threshold = 100
        table = []
        for i in range(28):
            for j in range(28):
             if Graying[i][j] < threshold:
                 table.append(0)
             else:
                 table.append(255)
        table1 = np.mat(table)
        table2 = table1.reshape(28,28)
        plt.show()

        cv2.imwrite('data/pre/' + file_path.split('.png')[0] + '.png', table2, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        #print(table2.shape)
        # print(table2)
        # ret,s = Image.fromarray(table1.astype(np.uint8))
        # print(s)
        # plt.imshow(table2)
        # plt.savefig('2333.png')

    #s = cv2.imread('data/pictures/233.png')
    # cv2.imshow('img2',s)
    # cv2.waitKey(0)
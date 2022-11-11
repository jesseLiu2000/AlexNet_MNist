# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 11:10:00 2020

------------------------------------------------------------------------
实验要求：
# 1.根据第四章最后一个实验，在MNIST数据集上面，完成剩余部分代码：要求输出每个epoch下的训练损失，训练准确率，测试损失，测试准确率
# 2.读入图像并输入预测标签
# 3.感兴趣的同学可以尝试把代码修改为GPU模式
# 4.需要注意的是，MNIST数据每幅图像大小为28*28，而LeNet5的输入图像大小为32*32，注意大小变换


说明：
LeNet-5 （1998， Yann LeCun 的 LeNet5）
卷积神经网路的开山之作，麻雀虽小，但五脏俱全，卷积层、pooling层、全连接层，这些都是现代CNN网络的基本组件
用卷积提取空间特征；
由空间平均得到子样本；
用 tanh 或 sigmoid 得到非线性；
用 multi-layer neural network（MLP）作为最终分类器；
层层之间用稀疏的连接矩阵，以避免大的计算成本。

输入：图像Size为32*32。这要比mnist数据库中最大的字母(28*28)还大。这样做的目的是希望潜在的明显特征，如笔画断续、角点能够出现在最高层特征监测子感受野的中心。
输出：10个类别，分别为0-9数字的概率

C1层是一个卷积层，有6个卷积核（提取6种局部特征），核大小为5 * 5
S2层是pooling层，下采样（区域:2 * 2 ）降低网络训练参数及模型的过拟合程度。
C3层是第二个卷积层，使用16个卷积核，核大小:5 * 5 提取特征
S4层也是一个pooling层，区域:2*2
C5层是最后一个卷积层，卷积核大小:5 * 5 卷积核种类:120
最后使用全连接层，将C5的120个特征进行分类，最后输出0-9的概率
以下代码来自官方教程
**********************************************************
注意代码与教材上的区别
1.教材上所有激活函数都是sigmoid函数，并且位于汇聚层（S2和S4），但本代码中激活函数全部使用ReLU，并位于卷积层（C1和C3）和全连接层
2.教材中S2和S4层使用的是平均汇聚，这里使用的是最大汇聚。
3.教材中第五层是卷积层，通过卷积把矩阵变为向量；这里则是把直接把第四层的输出拉成向量，然后使用全连接层。
**********************************************************
------------------------------------------------------------------------
"""

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.autograd import Variable
from PIL import Image
import torch.utils.data as Data
import os


batch_size = 64
learning_rate = 0.1 #q1: 在哪里用到了？什么用？ 优化函数中用到了这个； 增大效果？减少效果？
num_epochs = 5 #1个epoch表示过了1遍训练集中的所有样本,包含多个iteration。------>
size = 32




class LeNet5(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # 这里论文上写的是conv,官方教程用了线性层
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
 
    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


my_net = LeNet5()
print(my_net)



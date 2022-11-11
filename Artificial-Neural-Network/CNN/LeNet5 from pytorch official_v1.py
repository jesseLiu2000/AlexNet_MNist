# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 11:10:00 2020

------------------------------------------------------------------------
实验要求：
# 1.理解 1.1.2 中数据预处理的具体步骤
# 2.在1.2部分完成第二层网络的定义
# 3.理解1.2部分，第三层网络的定义中为什么没有使用softmax函数
# 4.完成1.2部分forward函数中对应部分的代码
# 5.理解整个算法的实现过程
# 6.统计不同学习率 learn_rate、迭代次数 epoch 对分类准确率的影响
# 7.统计隐藏层神经元的数量对实验结果的影响
# 8.增加一个隐藏层，效果怎样？
# 9.感兴趣的同学可以尝试把代码修改为GPU模式



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

data_preprocess = transforms.Compose([
    transforms.Resize(size),  # 缩放
    transforms.CenterCrop((size, size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])]) # 使用data_process实例化函数transforms.Compose ？

train_dataset = datasets.MNIST(root='./data', train=True, transform=data_preprocess, download=False) #载入下载后的文件，但下载的文件需要先解压
test_dataset = datasets.MNIST(root='./data', train=False, transform=data_preprocess, download=False)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size= batch_size,shuffle=True) #返回的变量train_loader是什么？含有所有图像，但是已经把他们分成了若干个batch？
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


print("Number of train_dataset:", len(train_dataset))
print("Number of test_dataset:", len(test_dataset))
img,target = train_dataset[3]  #……获取数据集中第n个元素的的图片信息和类别……#
print("Image Size: ", img.size())  #按照格式“Image Size: ...”输出该图片大小
print("Image Target: ", target)  #按照格式“Image Target: ...”输出该图片的类别


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
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(my_net.parameters(), lr=learning_rate)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)  #判断是否存在GPU，如果存在就输出“cuda:0"，否则输出”cpu“
my_net.to(device)

# net.parameters（）：返回模型的可学习参数
#params = list(net.parameters())
#print(len(params))
#print(params[7].size(), params[8].size(), params[9].size())  # conv1's .weight. 包括5个权重W和5个偏置b


def train_my_net(model, loss_fn, optimizer, num_epochs):
    #使用PyTorch进行训练和测试时一定注意要把实例化的model指定train/eval，eval（）时，...
    #   ...框架会自动把BN和DropOut固定住，不会取平均，而是用训练好的值，不然的话，
    #   ...一旦test的batch_size过小，很容易就会被BN层导致生成图片颜色失真极大！！！！！！
    model.train()  # 训练模式，启用 BatchNormalization 和 Dropout
    for epoch in range(num_epochs):     #每个epoch中，会对训练集中的所有图片都训练一遍。
        print('epoch {}/{}'.format(epoch, num_epochs - 1))  #format()格式化输出，把（）里面的内容按顺序带入到前面{}中
        print('-' * 10)

        ## training------------------
        train_loss = 0.0
        train_acc = 0.0

        # 获取数据输入和标签，封装成变量
        for data in train_loader:  # 每次取一个batch_size张图片
            img, label = data  # 获得图片和标签，img.size:64*1*28*28 = [Batch(size), Channels, Height, Width]

            # 对数据进行封装，将其放入Variable容器中。
            #    之所以需要将tensor转化成variable是因为pytorch中tensor(张量)只能
            #    放在CPU上运算，而(variable)变量是可以只用GPU进行加速计算的。 所以
            #    说这就是为什么pytorch加载图像的时候一般都会使用(variable)变量.
            img = Variable(img)
            label = Variable(label)
            img, label = img.to(device), label.to(device)
            # ……前向传播+反向传播……#
            # 1.梯度参数清零
            optimizer.zero_grad()

            # 2.前向传播
            # 模型预测，得到预测值
            out = model(img)
            #print(out.shape, sum(out.data[0,:]))  #这里还没有代入交叉熵损失函数，所以每个样本输出之和不等于1
            # 计算损失，这里得到的其实是平均loss
            loss = loss_fn(out, label)
            # 求每个样本预测的所有类别概率中最大的一个，返回值是最大概率和其所在位置（类别数）
            _, pred = torch.max(out.data, 1)

            # 3.反向传播
            loss.backward() #反向传播计算得到每个参数的梯度值
            optimizer.step() #通过梯度下降执行一步参数更新

            # 统计
            # 计算总loss（平均loss×样本数）
            # loss.item()：取出loss变量中的数值。
            #print(loss, loss.item())
            train_loss += loss.item() * label.size(0)  # lable.size(0)=64
            # 计算预测值与真实值相等的个数，也就是accuracy
            train_correct = (pred == label).sum()
            # 对每个batch的accuracy进行累加
            train_acc += train_correct.item()
        print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(train_dataset)),
                                                       train_acc / (len(train_dataset))))

        ##在测试集上检验效果
        model.eval()  # 将模型改为测试模式；不启用 BatchNormalization 和 Dropout
        eval_loss = 0.0
        eval_acc = 0.0

        # 获取数据输入和标签，封装成变量
        for data in test_loader:  # 每次取一个batch_size张图片
            img, label = data  # 获得图片和标签，img.size:64*1*28*28

            # 对数据进行封装，将其放入Variable容器中
            img = Variable(img)
            label = Variable(label)
            img, label = img.to(device), label.to(device)

            # ……前向传播，测试集不需要反向传播……#
            # 模型预测，得到预测值
            out = model(img)
            # 计算损失
            loss = loss_fn(out, label)
            # 计算总loss（平均loss×样本数）
            eval_loss += loss.item() * label.size(0)  # lable.size(0)=64
            ##求每个样本预测的所有类别概率中最大的一个，返回值是最大概率和其所在位置（类别数）
            _, pred = torch.max(out, 1)

            # 统计
            # 计算预测值与真实值相等的个数，也就是accuracy
            num_correct = (pred == label).sum()
            # 对每个batch的accuracy进行累加
            eval_acc += num_correct.item()
        print('Test Loss:{:.6f}, Acc: {:.6f}'.format(eval_loss / (len(test_dataset)), eval_acc / (len(test_dataset))))


train_my_net(my_net, loss_fn, optimizer, num_epochs)
torch.save(my_net, "./FNN.pkl")
model_from_file = torch.load('./FNN.pkl')#……保存和加载网络中的参数……#

#……保存和加载网络中的参数……#
torch.save(my_net.state_dict(), "./data/model_2.pkl")
model_parameter = torch.load('./data/model_2.pkl')
my_net.load_state_dict(model_parameter)
#os.getcwd()
#os.listdir("/data/")


#1.7 测试
#自己手写数字或网上下载数字图片，对图片进行灰度化、缩放、切割、转换等操作，并将其上传至平台的数据集中，复制链接地址到变量path,通过定义的readImage()函数读取图片，进行识别测试。
def readImage(path='./data/timg.jpg'):
    mode = Image.open(path).convert('L')  # 转换成灰度图

    # ……对图片进行缩放、切割、转换等操作……#
    transform1 = transforms.Compose([
        transforms.Resize(size),  # 缩放
        transforms.CenterCrop((size, size)),  # 切割
        transforms.ToTensor(),  # 转换为Tensor
        transforms.Normalize([0.5], [0.5])
        ])

    mode = transform1(mode)
    return mode


# 读取图片，调用模型进行训练
figure = readImage()  #得到大小为1*32*32的张量
figure = figure.to(device)
#print(figure.shape)
figure = figure.unsqueeze(0) #因为网络的输入需要是4维张量，所以在最前面增加一个维度，变为1*1*32*32
print(figure.shape)
y_pred = my_net(figure)
print(y_pred)
_, pred = torch.max(y_pred, 1)
print('prediction = ', pred.item())



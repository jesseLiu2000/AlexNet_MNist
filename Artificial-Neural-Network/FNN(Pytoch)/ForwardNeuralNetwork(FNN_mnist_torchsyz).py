# -*- coding: utf-8 -*-
"""
Created on Sat Nov 7 18:47:00 2020

------------------------------------------------------------------------
实验要求：
# 1.理解 1.1.2 中数据预处理的具体步骤
    torchvision.ToTensor：将shape为(H, W, C)的ndarray数组或图像转为shape为(C, H, W)的tensor即张量；
    transforms.Normalize(mean, std)：按照公式output = (input - mean)/std, 把每个channal归一化到[-1,1]之间
# 2.在1.2部分完成第二层网络的定义
# 3.理解1.2部分，第三层网络的定义中为什么没有使用softmax函数
    注意这里没有最后的softmax激活，这是因为下面的交叉熵损失函数CrossEntropyLoss()在计算时会在内部自己把 y 转化成softmax(y) 然后再进行交叉熵loss的运算.
# 4.完成1.2部分forward函数中对应部分的代码
# 5.理解整个算法的实现过程
# 6.统计不同学习率 learn_rate、迭代次数 epoch 对分类准确率的影响
    0.01
    Train Loss: 0.375566, Acc: 0.892450
    Test Loss:0.322073, Acc: 0.908600
    0.03
    Train Loss: 0.273967, Acc: 0.919067
    Test Loss:0.242308, Acc: 0.927100
    0.06
    Train Loss: 0.191257, Acc: 0.942317
    Test Loss:0.150505, Acc: 0.953800
    0.1
    Train Loss: 0.164704, Acc: 0.949267
    Test Loss:0.162060, Acc: 0.946600
    5
    Train Loss: 0.254040, Acc: 0.925883
    Test Loss:0.245705, Acc: 0.925300
    8
    Train Loss: 0.191896, Acc: 0.944533
    Test Loss:0.186643, Acc: 0.943300
    figure_number=5 bingo!
# 7.统计隐藏层神经元的数量对实验结果的影响
    300 100
    Train Loss: 0.375566, Acc: 0.892450
    Test Loss:0.322073, Acc: 0.908600
    200 100
    Train Loss: 0.377766, Acc: 0.891250
    Test Loss:0.325366, Acc: 0.906400
    400 100
    Train Loss: 0.371615, Acc: 0.893000
    Test Loss:0.319211, Acc: 0.908400
    400 200
    Train Loss: 0.367679, Acc: 0.894283
    Test Loss:0.334536, Acc: 0.900700
    100 50
    Train Loss: 0.385654, Acc: 0.888333
    Test Loss:0.338486, Acc: 0.903100
# 8.增加一个隐藏层，效果怎样？
    Train Loss: 0.433928, Acc: 0.872467
    Test Loss:0.347957, Acc: 0.895700
    torch.Size([1, 784])
    prediction =  7
# 9.感兴趣的同学可以尝试把代码修改为GPU模式
------------------------------------------------------------------------


数据介绍：
MNIST 数据集来自美国国家标准与技术研究所, National Institute of
Standards and Technology (NIST). 训练集 (training set) 由
来自 250 个不同人手写的数字构成, 其中 50% 是高中学生, 50% 来自人口
普查局 (the Census Bureau) 的工作人员. 测试集(test set) 也是同
样比例的手写数字数据.
在 MNIST数据集中训练数据集包含 60,000 个样本, 测试数据集包含 10,000
样本. 每张图片由 28 x 28 个像素点构成, 每个像素点用一个灰度值表示.
在这里, 我们将 28 x 28 的像素展开为一个一维的行向量, 这些行向量就是图
片数组里的行(每行 784 个值, 或者说每行就是代表了一张图片)。
"""


#1.1 载入数据
#1.1.1
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.autograd import Variable
from PIL import Image
from numpy import *
import torch.utils.data as Data
import os

# 加载数据集之前，设置模型的一些超参数
#……batch_size，批训练的数据个数……#
batch_size = 64

#……learning_rate，学习率……#
learning_rate = 0.01 #q1: 在哪里用到了？什么用？ 优化函数中用到了这个

#……num_epoches迭代次数，epoch 可以大致当成神经网络把训练集所有的照片从头看到尾都过了一遍……#
num_epochs = 8 #1个epoch表示过了1遍训练集中的所有样本,包含多个iteration。------>
# (iteration：表示1次迭代（也叫training step），每次迭代更新1次网络结构的参数。batch-size：1次迭代所使用的样本量；)

#1.1.2  数据预处理
# ……变量data_preprocess,进行数据的转换及标准化处理 ……#
#torchvision.ToTensor：将shape为(H, W, C)的ndarray数组或图像转为shape为(C, H, W)的tensor即张量；然后通过除以255归一化到[0,1]之间
#transforms.Normalize(mean, std)：按照公式output = (input - mean)/std, 把每个channal归一化到[-1,1]之间
data_preprocess = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])]) # 使用data_process实例化函数transforms.Compose ？

#1.1.3
# ……下载MNIST训练集,设置 root = path,train=Ture , transform=data_preprocess ,download=True,赋值给变量train_dataset……#
# train_dataset：最后的输出，包含数据的名字及位置？
train_dataset = datasets.MNIST(root='./data', train=True, transform=data_preprocess, download=False) #载入下载后的文件，但下载的文件需要先解压
# ……下载MNIST 测试集,设置 root = path,train= False , transform=data_preprocess ,download=True,赋值给变量test_dataset ……#
test_dataset = datasets.MNIST(root='./data', train=False, transform=data_preprocess, download=False)

#……创建batch_size=batch_size, shuffle=True的DataLoader变量data_loader, shuffle = True 表明提取数据时，随机打乱顺序。……#
#train_loader：函数实例化？
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size= batch_size,shuffle=True) #返回的变量train_loader是什么？含有所有图像，但是已经把他们分成了若干个batch？
#……创建batch_size=batch_size, shuffle=Flase的DataLoader变量test_loader,因为我们都是基于随机梯度下降的方式进行训练优化，但测试的时候因为不需要更新参数，所以就无须打乱顺序了。……#
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


#1.1.4
#……按照格式“INumber of train_dataset: ...”输出train_dataset和test_dataset样本大小……#
print("Number of train_dataset:", len(train_dataset))
print("Number of test_dataset:", len(test_dataset))
#……获取数据集中第n个元素的的图片信息和类别……#
img,target = train_dataset[3]
#按照格式“Image Size: ...”输出该图片大小
print("Image Size: ", img.size())
#按照格式“Image Target: ...”输出该图片的类别
print("Image Target: ", target)


#1.2  构建模型
# 定义前馈神经网络
class FNN(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        # 调用父类的初始化函数
        super(FNN, self).__init__()

        # ……定义三层网络……#
        # nn.Sequential() 将nn.Linear()和nn.ReLU()组合到一起
        # in_dim个节点连接到n_hidden_1个节点上,nn.ReLU(True))
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.ReLU(True))

        # 2、n_hidden_1个节点连接到n_hidden_2个节点上,nn.ReLU(True))
        ####################
        '''根据提示，并参考第一层，完成第二层网络的定义'''
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.ReLU(True))
        #self.layer2_1 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_2), nn.ReLU(True))
        ####################

        # 3、n_hidden_2个节点连接到out_dim个节点上
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))   #注意这里没有最后的softmax激活，这是因为下面的交叉熵损失函数CrossEntropyLoss()在计算时会在内部自己把 y 转化成softmax(y) 然后再进行交叉熵loss的运算.

    def forward(self, x):
        # ……定义对输入的依次操作，在三个网络层中进行线性映射……#
        # 1.输入x->layer1，更新到x
        x = self.layer1(x)
        # 2.输入x->layer2，更新到x
        ####################
        '''根据提示，并参考第一层，完成第二层网络的定义'''
        x=self.layer2(x)

        ####################
        #x = self.layer2_1(x)
        # 3.输入x->layer3，更新到x
        ####################
        '''根据提示，并参考第一层，完成第二层网络的定义'''
        x=self.layer3(x)
        ####################
        return x


#……导入网络，参数为28 * 28，300，100，10……#
model = FNN(28*28, 300, 100, 10)


#1.3  定义损失函数
#因为是多分类所以使用 nn.CrossEntropyLoss()，nn.BCELoss是二分类的损失函数，这里我们选择利用交叉熵函数作为损失函数。
#……交叉熵损失函数更新到变量loss_fn……#
loss_fn = nn.CrossEntropyLoss()

#1.4  定义优化函数
# 常用的优化函数有随机梯度下降SGD，Adam等。在训练时，优化方法采用SGD方法。下面代码中的optim.SGD初始化需要接受网络中待优化的Parameter列表（或是迭代器），以及学习率lr。
#……优化函数更新到变量optimizer……#
'''
SGD
Train Loss: 0.375566, Acc: 0.892450
Test Loss:0.322073, Acc: 0.908600

Adam:
Train Loss: 0.250889, Acc: 0.926700
Test Loss:0.233393, Acc: 0.932500
'''
optimizer = optim.SGD(model.parameters(), lr=learning_rate)


#1.5 训练模型
# 接下来开始训练模型，我们只需要遍历数据集，同时在每次迭代中，清空待优化参数的梯度，前向计算，反向传播以及优化器的迭代求解即可，最后输出在训练集和测试集上的准确率。
# 训练模型
# 训练模型
def train_model(model, loss_fn, optimizer, num_epochs):
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
            img, label = data  # 获得图片和标签，img.size:64*1*28*28
            print(img.size())
            img = img.view(img.size(0), -1)  # 将图片进行img的转换，展开成64 *784（28*28）

            # 对数据进行封装，将其放入Variable容器中。
            #    之所以需要将tensor转化成variable是因为pytorch中tensor(张量)只能
            #    放在CPU上运算，而(variable)变量是可以只用GPU进行加速计算的。 所以
            #    说这就是为什么pytorch加载图像的时候一般都会使用(variable)变量.
            img = Variable(img)
            label = Variable(label)

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
            img = img.view(img.size(0), -1)  # 将图片进行img的转换，展开成64 *784（28*28）

            # 对数据进行封装，将其放入Variable容器中
            img = Variable(img)
            label = Variable(label)

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


#运行训练模型的函数
train_model(model, loss_fn, optimizer, num_epochs)
#print("--")
#1.6 保存和加载模型
# 保存和加载整个模型，包括: 网络结构, 模型参数等
torch.save(model, "./FNN.pkl")
model_from_file = torch.load('./FNN.pkl')#……保存和加载网络中的参数……#

#……保存和加载网络中的参数……#
torch.save(model.state_dict(), "./data/model_2.pkl")
model_parameter = torch.load('./data/model_2.pkl')
model.load_state_dict(model_parameter)
#os.getcwd()
#os.listdir("/data/")


#1.7 测试
#自己手写数字或网上下载数字图片，对图片进行灰度化、缩放、切割、转换等操作，并将其上传至平台的数据集中，复制链接地址到变量path,通过定义的readImage()函数读取图片，进行识别测试。
def readImage(path='./data/timg.jpg', size=28):
    mode = Image.open(path).convert('L')  # 转换成灰度图

    # ……对图片进行缩放、切割、转换等操作……#
    transform1 = transforms.Compose([
        transforms.Resize(size),  # 缩放
        transforms.CenterCrop((size, size)),  # 切割
        transforms.ToTensor()  # 转换为Tensor
        ])

    mode = transform1(mode)
    mode = mode.view(mode.size(0), -1)
    return mode


# 读取图片，调用模型进行训练
figure = readImage(size=28)
#figure = figure.cuda()
print(figure.size())
y_pred = model(figure)
_, pred = torch.max(y_pred, 1)
print('prediction = ', pred.item())










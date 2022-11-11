# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 20:27:00 2020

------------------------------------------------------------------------
实验要求：
# 1.在对应位置分别实现：ReLU函数的梯度、交叉熵损失函数、分类准确率的计算
# 2.理解算法的实现过程
# 3.将代码在本地实现，并统计计算结果
# 4.统计不同学习率 learn_rate、迭代次数 epoch 对分类准确率的影响
# 5.统计隐藏层神经元的数量如果增加一倍、或者减少一倍会怎么？ 更多倍呢？
------------------------------------------------------------------------

说明：
一个两层网络（不含输入层）。
计算梯度的时候没有使用反向传播算法，而是直接基于链式法则进行的。
模型计算过程：X --> h1 -->  h1_relu -->  h2 --> h2_soft
h1 = X*W1;   h1_relu = relu(h1);
h2 = h1_relu*W2;   h2_soft = softmax(h2);

h2_log = log(h2_soft): 用于计算交叉熵


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




import os
import numpy as np
import torch   #用来载入数据

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}


def mnist_dataset():
    x = torch.load('train_images.pkl') #注意数据所在的路径格式
    y = torch.load('train_labels.pkl')
    x_test = torch.load('test_images.pkl')
    y_test = torch.load('test_labels.pkl')
    # x: (60000, 784); y:(60000,); x_test: (10000, 784); y_test: (10000,)

    # normalize
    x = x / 255.0
    x_test = x_test / 255.0

    return (x, y), (x_test, y_test)   #输出为元组（x，y）


# Demo numpy based auto differentiation
class Matmul:  #矩阵乘法，用于计算净输入
    def __init__(self):
        self.mem = {}

    def forward(self, x, W):  #净输入h = x*W
        h = np.matmul(x, W)
        self.mem = {'x': x, 'W': W}  #变量放到字典中？
        return h

    def backward(self, grad_y): #计算损失函数L关于权重及对应输入的梯度
        '''
        x: shape(N, d)
        w: shape(d, d')
        grad_y: shape(N, d')
        '''
        x = self.mem['x']
        W = self.mem['W']

        ####################
        '''计算矩阵乘法的对应的梯度'''
        grad_x = np.matmul(grad_y, W.T)  # 这里涉及到矩阵对矩阵求导数，shape(N, b)
        grad_W = np.matmul(x.T, grad_y)  # 同样涉及到矩阵对矩阵求导
        ####################

        return grad_x, grad_W


class Relu:
    def __init__(self):
        self.mem = {}

    def forward(self, x):
        self.mem['x'] = x
        return np.where(x > 0, x, np.zeros_like(x))  #判断x是否大于0，如果是输出x，否则输出0
        # np.zeros_like(x)：产生一个维度与x完全一样的全0数组

    def backward(self, grad_y):
        '''
        grad_y: same shape as x
        '''
        ####################
        '''计算relu 激活函数对应的梯度'''
        x = self.mem['x']
        grad_x = (x>0).astype(np.float32) * grad_y
        ####################
        return grad_x


class Softmax:
    '''
    softmax over last dimention
    '''

    def __init__(self):
        self.epsilon = 1e-12
        self.mem = {}

    def forward(self, x):
        '''
        x: shape(N, c) (60000, 10)
        '''
        x_exp = np.exp(x)
        partition = np.sum(x_exp, axis=1, keepdims=True)  #以竖轴（axis=1）为基准 ，同行相加，keepdims主要用于保持矩阵的二维特性
        out = x_exp / (partition + self.epsilon)  #点除运算，x.exp中第i行的元素除以partition中的第i行元素

        self.mem['out'] = out
        self.mem['x_exp'] = x_exp
        return out

    def backward(self, grad_y): # 计算损失函数关于h2的梯度，这里输入为链式法则右侧的部分
        '''
        grad_y: same shape as x
        '''
        s = self.mem['out']
        sisj = np.matmul(np.expand_dims(s, axis=2), np.expand_dims(s, axis=1))  # (N, c, c)
        g_y_exp = np.expand_dims(grad_y, axis=1)  #在1位置扩展数据的形状
        tmp = np.matmul(g_y_exp, sisj)  # (N, 1, c)
        tmp = np.squeeze(tmp, axis=1)  #从数组的形状中删除单维度条目，即把shape中为1的维度去掉；axis用于指定需要删除的维度，但是指定的维度必须为单维度，否则将会报错；
        tmp = -tmp + grad_y * s
        return tmp


class Log:   #计算交叉熵loss的时候会用到
    '''
    softmax over last dimention
    '''

    def __init__(self):
        self.epsilon = 1e-12
        self.mem = {}

    def forward(self, x):
        '''
        x: shape(N, c)
        '''
        out = np.log(x + self.epsilon)

        self.mem['x'] = x
        return out

    # 这一步是计算交叉熵损失关于输出标签y^hat的梯度；grad_y：这里是-y, 即训练样本的标签。
    # s_L/s_(y^hat) = - y* (1./ y^hat)
    def backward(self, grad_y):
        '''
        grad_y: same shape as x
        '''
        x = self.mem['x']

        return 1. / (x + 1e-12) * grad_y


#建立模型
class myModel:
    def __init__(self):
        self.W1 = np.random.normal(size=[28 * 28 + 1, 100])
        self.W2 = np.random.normal(size=[100, 10])

        self.mul_h1 = Matmul()
        self.mul_h2 = Matmul()
        self.relu = Relu()
        self.softmax = Softmax()
        self.log = Log()

    def forward(self, x):
        x = x.reshape(-1, 28 * 28) #对于输入x /in N*C, 把他reshape为M*(28*28)的形式，其中M根据N*C/(28*28)自行计算
        bias = np.ones(shape=[x.shape[0], 1])
        x = np.concatenate([x, bias], axis=1)

        self.h1 = self.mul_h1.forward(x, self.W1)  # 计算第一层的净输入，shape(5, 4)
        self.h1_relu = self.relu.forward(self.h1)  #激活
        self.h2 = self.mul_h2.forward(self.h1_relu, self.W2)
        self.h2_soft = self.softmax.forward(self.h2)
        self.h2_log = self.log.forward(self.h2_soft)  #把交叉熵中的log运算放到了这里

    def backward(self, label): #label: 训练数据的标签
        self.h2_log_grad = self.log.backward(-label) # 损失函数关于y_hat，这里y_hat就等于h2_soft的梯度
        self.h2_soft_grad = self.softmax.backward(self.h2_log_grad) # 损失函数关于h2的梯度
        self.h2_grad, self.W2_grad = self.mul_h2.backward(self.h2_soft_grad) # 损失函数关于h1_relu，W2的梯度
        self.h1_relu_grad = self.relu.backward(self.h2_grad)  # 损失函数关于h1的梯度
        self.h1_grad, self.W1_grad = self.mul_h1.backward(self.h1_relu_grad)  #计算关于输入, W1的梯度


model = myModel()


#计算 loss
def compute_loss(log_prob, labels):
    ####################
    '''计算交叉熵损失函数'''

    ####################
    return np.mean(np.sum(-log_prob * labels, axis=1))


def compute_accuracy(log_prob, labels):   #输入为one-hot形式的标签
    predictions = np.argmax(log_prob, axis=1) #转为正常的标签
    truth = np.argmax(labels, axis=1)
    ####################
    '''计算分类准确率'''

    ####################
    return np.mean(predictions == truth)


def train_one_step(model, x, y, learn_rate):  #x:训练数据，y:训练数据对应的标签。
    model.forward(x)  #正向传播，得到最终的输出 h2_log， 用于下面计算损失函数
    model.backward(y) #反向传播，得到每个变量的梯度，用于下面的更新
    model.W1 -= learn_rate * model.W1_grad  #批量梯度下降，每次代入所有训练数据，学习率为1e-5
    model.W2 -= learn_rate * model.W2_grad
    loss = compute_loss(model.h2_log, y)
    accuracy = compute_accuracy(model.h2_log, y)
    return loss, accuracy


def test(model, x, y):
    model.forward(x)
    loss = compute_loss(model.h2_log, y)
    accuracy = compute_accuracy(model.h2_log, y)
    return loss, accuracy


#实际训练
train_data, test_data = mnist_dataset()

#标签转为one-hot形式
train_label = np.zeros(shape=[train_data[0].shape[0], 10])
test_label = np.zeros(shape=[test_data[0].shape[0], 10])
train_label[np.arange(train_data[0].shape[0]), np.array(train_data[1])] = 1.
test_label[np.arange(test_data[0].shape[0]), np.array(test_data[1])] = 1.

for epoch in range(50):
    loss, accuracy = train_one_step(model, train_data[0], train_label, learn_rate=1e-5)  #每个epoch都使用所有训练数据
    print('epoch', epoch, ': loss', loss, '; accuracy', accuracy)

loss, accuracy = test(model, test_data[0], test_label)  #使用上面训练好的model，对测试数据进行测试

print('test loss', loss, '; accuracy', accuracy)
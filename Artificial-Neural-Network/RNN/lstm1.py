# -*- coding: utf-8 -*-
"""
Created on Tue Dec 1 19:46:00 2020

------------------------------------------------------------------------
实验要求：
# 1.读懂下面的模型，了解其中参数的含义。（可以参考课本中的 “序列到类别模式”）
# 2.根据第四章及第五章最后一个实验，在MNIST数据集上面，完成剩余部分代码：要求输出每个epoch下的训练损失，训练准确率，测试损失，测试准确率
# 3.读入图像并输入预测标签
# 4.注意：在训练以及测试的时候，一定要注意模型输入数据的类型。也就是如何把4维数据变为三维。
#   参考：img = img.view(-1, 28, 28)  #把4维的img转换为3维 [batch, seq, feature]

========== 50 ===== ===== train loss is  0.16933134198188782 ==========
========== 100 ===== ===== train loss is  0.020747549831867218 ==========
========== 150 ===== ===== train loss is  0.012706916779279709 ==========
========== 200 ===== ===== train loss is  0.0020406581461429596 ==========
========== 250 ===== ===== train loss is  0.04599371924996376 ==========
========== 300 ===== ===== train loss is  0.00536196306347847 ==========
========== 350 ===== ===== train loss is  0.11277002841234207 ==========
========== 400 ===== ===== train loss is  0.11507198214530945 ==========
========== 450 ===== ===== train loss is  0.020300595089793205 ==========
========== 500 ===== ===== train loss is  0.06683354079723358 ==========
========== 550 ===== ===== train loss is  0.016564378514885902 ==========
========== 600 ===== ===== train loss is  0.007648340426385403 ==========
========== 650 ===== ===== train loss is  0.009225009009242058 ==========
========== 700 ===== ===== train loss is  0.01413658820092678 ==========
========== 750 ===== ===== train loss is  0.008559022098779678 ==========
========== 800 ===== ===== train loss is  0.006032626610249281 ==========
========== 850 ===== ===== train loss is  0.005754915066063404 ==========
========== 900 ===== ===== train loss is  0.0596923753619194 ==========
Train loss: 0.047590,Acc:0.985700
========== 50 ===== ===== test loss is  0.10589856654405594 ==========
========== 100 ===== ===== test loss is  0.009977668523788452 ==========
========== 150 ===== ===== test loss is  0.10748825967311859 ==========
Test loss: 0.039042,Acc:0.987500

说明：
LSTM用于MNIST数据分类（序列到类别模式）
我们定义的 input_size为 28，这是因为输入的手写数据的宽高为 28 × 28,所以可以将每一张图片看作长度
为 28 的序列数据，每个数据的维度为28维。 模型最后输出的结果是用作分类的，所以仍然需要输出 10 个数据，

输入：图像Size为28*28。
输出：10个类别，分别为属于数字“0-9”的概率
------------------------------------------------------------------------
"""

import torch
import numpy
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import os
import matplotlib . pyplot as plt
#matplotlib inline

batch_size = 64
learning_rate = 0.003 #学习率
num_epochs = 5 #1个epoch表示过了1遍训练集中的所有样本,包含多个iteration。

data_preprocess=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])
image_path='./data'
train_dataset = datasets.MNIST(root=image_path, train=True, transform=data_preprocess,download=False)
test_dataset = datasets.MNIST(root=image_path, train=False, transform=data_preprocess,download=False)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

img,target=train_dataset[13]
img=img.numpy().transpose(1,2,0)
# 定义 Recurrent Network 模型
#输入参数：
# (1) in_dim：用于指定输入数据的特征数。
# (2) hidden_dim： 用于指定最后隐藏层的输出特征数。
# (3) n_layer： 用于指定循环层堆叠的数量，默认会使用 1。
# (4) n_class：样本类别数量（或输出神经元数量）
# (5) batch first: 在我们的循环神经网络模型中输入层和输出层用到的数据的默认维度是（seq, batch, feature），
#     其中， seq为序列的长度， batch为数据批次的数量， feature为输入或输出的特征数。如果我们将该参数指定为True，
#     那么输入层和输出层的数据维度将重新对应为（batch, seq, feature）。
class Rnn(nn.Module):

    def __init__(self, in_dim, hidden_dim, n_layer, n_class):
        super(Rnn, self).__init__()
        self.n_layer=n_layer
        self.hidden_dim=hidden_dim
        # self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True)
        self.lstm = nn.LSTM(    # 这里可以换成nn.RNN(), 但不建议这么做，因为RNN很难训练
            input_size=in_dim,
            hidden_size=hidden_dim,  # LSTM隐藏层神经元的数量
            num_layers=n_layer,  # LSTM的层数
            batch_first=True,  # input & output 的第一维变为 batch_size. e.g. (batch_size, seq, input_size)
        )
        self.classifier = nn.Linear(hidden_dim, n_class)

    def forward(self, x):
        h0=torch.zeros(self.n_layer,x.size(0),self.hidden_dim).to(device="cpu")
        c0=torch.zeros(self.n_layer,x.size(0),self.hidden_dim).to(device="cpu")
        out, _ = self.lstm(x, (h0,c0))
        out = out[:, -1, :]   # 因为我们的模型处理的是分类问题，所以需要提取序列最后一个样本的输出结果作为主当前循环神经网络模型的输出。
        out = self.classifier(out)
        return out


my_net = Rnn(28, 128, 2, 10)  # 图片大小是28x28,这里将每一张图片看作序列长度为 28 的序列数据，每个序列数据的维度为28维。
print(my_net)
loss_fn=nn.CrossEntropyLoss()
optimizer=optim.Adam(my_net.parameters(),lr=learning_rate)

device=torch.device("cpu")
print(device)
my_net.to(device)



def train_my_net(model,loss_fn,optimizer,num_epochs):
    for epoch in range(num_epochs):
        print("-"*10)

        train_loss=0.0
        train_acc=0.0

        model.train()

        for step,data in enumerate(trainloader):
            img,label=data
            img=img.view(-1,28,28)
            img,label=img.to(device),label.to(device)
            optimizer.zero_grad()
            out=model(img)
            loss=loss_fn(out,label)
            _,pred=torch.max(out.data,1)
            loss.backward()
            optimizer.step()

            train_loss+=loss.item()*label.size(0)
            train_correct=(pred==label).sum()
            train_acc+=train_correct.item()
            if step!=0 and step%50 ==0:
                print("="*10,step,"="*5,"="*5,"train loss is ",loss.item(),"="*10)
        print("Train loss: {:.6f},Acc:{:.6f}".format(train_loss/(len(train_dataset)),train_acc/(len(train_dataset))))

        model.eval()
        eval_loss=0.0
        eval_acc = 0.0
        for step,(img,label) in enumerate(testloader):
            img=img.view(-1,28,28)
            img,label=img.to(device),label.to(device)
            optimizer.zero_grad()
            out=model(img)
            loss=loss_fn(out,label)
            _,pred=torch.max(out.data,1)
            loss.backward()
            optimizer.step()

            eval_loss+=loss.item()*label.size(0)
            num_correct=(pred==label).sum()
            eval_acc+=num_correct.item()
            if step!=0 and step%50 ==0:
                print("="*10,step,"="*5,"="*5,"test loss is ",loss.item(),"="*10)
        print("Test loss: {:.6f},Acc:{:.6f}".format(eval_loss/(len(test_dataset)),eval_acc/(len(test_dataset))))


train_my_net(my_net,loss_fn,optimizer,num_epochs)
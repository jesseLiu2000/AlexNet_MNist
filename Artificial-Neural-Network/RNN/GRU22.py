import torch
import torch.nn as nn
'''
Author:Eric_TRF
Date:2020 /09 /03
'''


class HomorNetv2(torch.nn.Module):
    def __init__(self, input_dim, hidden_size, out_size, n_layers=1, batch_size=1):
        super(HomorNetv2, self).__init__()

        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.out_size = out_size

        # 这里指定了BATCH FIRST,所以输入时BATCH应该在第一维度
        self.gru = torch.nn.GRU(input_dim, hidden_size, n_layers, batch_first=True, bidirectional=True)

        # 加了一个线性层，全连接
        self.fc1 = torch.nn.Linear(hidden_size * 2, 300)
        # 加入了第二个全连接层
        self.fc2 = torch.nn.Linear(300, out_size)

    def forward(self, word_inputs, hidden):
        # hidden 就是上下文输出，output 就是 RNN 输出
        output, hidden = self.gru(word_inputs, hidden)
        # output是所有隐藏层的状态，hidden是最后一层隐藏层的状态
        output = self.fc1(output)
        output = self.fc2(output)

        # 仅仅获取 time seq 维度中的最后一个向量
        # the last of time_seq
        output = output[:, -1, :]

        return output, hidden

    def init_hidden(self):
        # 这个函数写在这里，有一定迷惑性，这个不是模型的一部分，是每次第一个向量没有上下文，在这里捞一个上下文，仅此而已。
        hidden = torch.autograd.Variable(
            torch.zeros(2 * self.n_layers, self.batch_size, self.hidden_size, device='cuda'))
        return hidden


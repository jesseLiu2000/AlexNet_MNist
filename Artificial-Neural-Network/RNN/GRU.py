import torch
# from torch import nn,optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
import torch.nn as nn
import torch.functional as F
import math
# https://github.com/emadRad/lstm-gru-pytorch/blob/master/lstm_gru.ipynb
class GRUCell(nn.Module):
    """
    An implementation of GRUCell.

    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        x = x.view(-1, x.size(1))

        gate_x = self.x2h(x)
        gate_h = self.h2h(hidden)

        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()

        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)

        # 公式1
        resetgate = F.sigmoid(i_r + h_r)
        # 公式2
        inputgate = F.sigmoid(i_i + h_i)
        # 公式3
        newgate = F.tanh(i_n + (resetgate * h_n))
        # 公式4，不过稍微调整了一下公式形式
        hy = newgate + inputgate * (hidden - newgate)

        return hy


class GRUConvCell(nn.Module):

    def __init__(self, input_channel, output_channel):
        super(GRUConvCell, self).__init__()

        # filters used for gates
        gru_input_channel = input_channel + output_channel
        self.output_channel = output_channel

        self.gate_conv = nn.Conv2d(gru_input_channel, output_channel * 2, kernel_size=3, padding=1)
        self.reset_gate_norm = nn.GroupNorm(1, output_channel, 1e-6, True)
        self.update_gate_norm = nn.GroupNorm(1, output_channel, 1e-6, True)

        # filters used for outputs
        self.output_conv = nn.Conv2d(gru_input_channel, output_channel, kernel_size=3, padding=1)
        self.output_norm = nn.GroupNorm(1, output_channel, 1e-6, True)

        self.activation = nn.Tanh()

    # 公式1，2
    def gates(self, x, h):
        # x = N x C x H x W
        # h = N x C x H x W

        # c = N x C*2 x H x W
        c = torch.cat((x, h), dim=1)
        f = self.gate_conv(c)

        # r = reset gate, u = update gate
        # both are N x O x H x W
        C = f.shape[1]
        r, u = torch.split(f, C // 2, 1)

        rn = self.reset_gate_norm(r)
        un = self.update_gate_norm(u)
        rns = torch.sigmoid(rn)
        uns = torch.sigmoid(un)
        return rns, uns

    # 公式3
    def output(self, x, h, r, u):
        f = torch.cat((x, r * h), dim=1)
        o = self.output_conv(f)
        on = self.output_norm(o)
        return on

    def forward(self, x, h=None):
        N, C, H, W = x.shape
        HC = self.output_channel
        if (h is None):
            h = torch.zeros((N, HC, H, W), dtype=torch.float, device=x.device)
        r, u = self.gates(x, h)
        o = self.output(x, h, r, u)
        y = self.activation(o)

        # 公式4
        return u * h + (1 - u) * y


class GRUNet(nn.Module):

    def __init__(self, hidden_size=64):
        super(GRUNet, self).__init__()

        self.gru_1 = GRUConvCell(input_channel=4, output_channel=hidden_size)
        self.gru_2 = GRUConvCell(input_channel=hidden_size, output_channel=hidden_size)
        self.gru_3 = GRUConvCell(input_channel=hidden_size, output_channel=hidden_size)

        self.fc = nn.Conv2d(in_channels=hidden_size, out_channels=1, kernel_size=3, padding=1)

    def forward(self, x, h):
        if h is None:
            h = [None, None, None]

        h1 = self.gru_1(x, h[0])
        h2 = self.gru_2(h1, h[1])
        h3 = self.gru_3(h2, h[2])

        o = self.fc(h3)

        return o, [h1, h2, h3]


if __name__ == '__main__':
    # from utils import *

    device = 'cpu'

    x = torch.rand(1, 1, 10, 20).to(device)

    grunet = GRUNet()
    grunet = grunet.to(device)
    grunet.eval()

    h = None
    o, h_n = grunet(x, h)

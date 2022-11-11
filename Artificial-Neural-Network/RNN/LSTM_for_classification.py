import torch
from torch import nn,optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets,transforms

# 超参数
batch_size = 100    # 批大小
learning_rate = 0.01 # 学习率
num_epoches = 20  # 训练次数

data_tf = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])

train_dataset = datasets.MNIST(root='./data',train=True,transform=data_tf,download=False)
test_dataset = datasets.MNIST(root='./data',train=False,transform=data_tf,download=False)
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)


class Rnn(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, n_class):
        super(Rnn, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, n_class)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.classifier(out)
        return out


model = Rnn(28, 128, 2, 10)  # 图片大小是28x28
use_gpu = torch.cuda.is_available()  # 判断是否有GPU加速
if use_gpu:
    model = model.cuda()

# 定义loss和optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 开始训练
for epoch in range(num_epoches):
    running_loss = 0.0
    running_acc = 0.0
    for i, data in enumerate(train_loader, 1):
        img, label = data
        img = img.squeeze(1)
        if torch.cuda.is_available():
            img = img.cuda()
            label = label.cuda()
        else:
            img = Variable(img)
            label = Variable(label)

            # 向前传播
            out = model(img)
            loss = criterion(out, label)
            running_loss += loss.item() * label.size(0)
            _, pred = torch.max(out, 1)
            num_correct = (pred == label).sum()
            running_acc += num_correct.item()

            # 向后传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 300 == 0:
                print('[{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format(
                    epoch + 1, num_epoches, running_loss / (batch_size * i),
                    running_acc / (batch_size * i)))
        #print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
            #epoch + 1, running_loss / (len(train_dataset)), running_acc / (len(train_dataset))))

    model.eval()
    eval_loss = 0.
    eval_acc = 0.
    for data in test_loader:
        img, label = data
        b, c, h, w = img.size()
        assert c == 1, 'channel must be 1'
        img = img.squeeze(1)
        # img = img.view(b*h, w)
        # img = torch.transpose(img, 1, 0)
        # img = img.contiguous().view(w, b, h)
        if use_gpu:
            with torch.no_grad():
                img = Variable(img, volatile=True).cuda()
                label = Variable(label, volatile=True).cuda()
        else:
            with torch.no_grad():
                img = Variable(img, volatile=True)
                label = Variable(label, volatile=True)
        out = model(img)
        loss = criterion(out, label)
        eval_loss += loss.item() * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        eval_acc += num_correct.item()
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
        test_dataset)), eval_acc / (len(test_dataset))))

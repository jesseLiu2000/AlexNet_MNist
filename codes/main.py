# coding:utf-8

import torch as t
from torch.utils.data import DataLoader
from torchnet import meter
from tqdm import tqdm
import models
from config import opt
from data.dataset import Minst
# from utils.visualize import Visualizer
import matplotlib.pyplot as plt
import cv2
import os
def train(**kwargs):
    """
    训练
    :param kwargs:
    :return:
    """
    opt.parse(kwargs)
    # vis = Visualizer(opt.env)

    # model加载模型以及模型的初始化
    model = getattr(models, opt.model)().eval()       #返回对象属性值
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu:
        model.cuda()

    # data
    #加载训练数据集和验证集
    train_data = Minst(data_root=opt.train_image_path, label_root=opt.train_label_path, train=True)
    val_data = Minst(data_root=opt.train_image_path, label_root=opt.train_label_path, train=False)

    #这个就是利用Dataloader去读取
    train_dataloader = DataLoader(train_data, opt.batch_size,
                                  shuffle=True,             #将元素随机排序
                                  num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data, opt.batch_size,
                                shuffle=False,
                                num_workers=opt.num_workers)

    # 目标函数和优化器
    criterion = t.nn.CrossEntropyLoss()
    lr = opt.lr   #learning rate
    optimizer = t.optim.Adam(model.parameters(),
                             lr=lr,
                             weight_decay=opt.weight_decay)

    # 统计指标，平滑处理之后的损失，还有混淆矩阵
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(10)    #分类结果的精度放在矩阵中
    previous_loss = 1e100   #模糊处理，降噪处理

    for epoch in range(opt.max_epoch):
        #重新传参
        loss_meter.reset()
        confusion_matrix.reset()

        for ii, (data, target) in enumerate(train_dataloader):  #枚举
            if opt.use_gpu:
                data = data.cuda()
                target = target.cuda()

            # 优化器重置导数
            optimizer.zero_grad()
            # 模型计算
            score = model(data)
            # 根据结果计算损失
            loss = criterion(score, target)
            # 损失回传
            loss.backward()
            # 更新参数
            optimizer.step()

            loss_meter.add(loss.item())
            confusion_matrix.add(score.data, target.data)



        model.save()

        # 计算验证集上的指标
        val_cm, val_accuracy = val(model, val_dataloader)
        print('val_accuracy', val_accuracy)
        print("epoch:{epoch}, lr:{lr}, loss:{loss},val_accuracy:{val_accuracy} ".format(
            epoch=epoch,
            loss=loss_meter.value()[0],
            lr=lr,
            val_accuracy=val_accuracy
        ))

        # 如果损失不再下降，降低学习率
        if loss_meter.value()[0] > previous_loss:
            lr = lr * opt.lr_decay
            for param_groups in optimizer.param_groups:
                param_groups['lr'] = lr
        previous_loss = loss_meter.value()[0]



def val(model, dataloader):
    """
    计算模型在验证集上的准确率等信息，用于辅助训练
    :param model:
    :param dataloader:
    :return:
    """
    model.eval()
    confusion_matrix = meter.ConfusionMeter(10)
    for ii, (input, target) in tqdm(enumerate(dataloader)):       #可视化进程条
        if opt.use_gpu:
            input = input.cuda()
            target = target.cuda()
        score = model(input)
        confusion_matrix.add(score.detach().squeeze(), target.type(t.LongTensor))

    model.train()
    cm_value = confusion_matrix.value()
    #经过混淆矩阵迭代，计算Top1
    accuracy = 100. * (cm_value.diagonal().sum()) / (cm_value.sum())       #查看对角元素求和
    return confusion_matrix, accuracy


def test(**kwargs):
    """
    测试
    :param kwargs:
    :return:
    """
    opt.parse(kwargs)

    # configure model
    # t.device('cuda') if opt.use_gpu else t.device('cpu')
    model = getattr(models, opt.model)().eval()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    model.to(t.device('cuda') if opt.use_gpu else t.device('cpu'))

    # data
    train_data = Minst(data_root=opt.test_image_path, test=True)#这句话就是读取test文件里面的数据，因为它本身是一个image和label分开的

    test_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    results = []
    for ii, (input, path) in tqdm(enumerate(test_dataloader)):
        if opt.use_gpu:
            input = input.cuda()
        score = model(input)
        #这个score就是预测的值
        # print(score.data)
        probability = t.nn.functional.softmax(score, dim=1)[:, 0].detach().tolist()
        label = score.max(dim = 1)[1].detach().tolist()
        #然后这句话就是获得预测的标签 ，实际就是看你的结果准不准


        batch_results = [(path_.item(), probability_) for path_, probability_ in zip(path, probability)]

        results += batch_results
        print("---------------------")
        print(results)
        print("----label---",label)
        print("---------------------")
    # write_csv(results, opt.result_file)


    return results


def help():
    """
    :return: 
    """
    pass


import torch.nn.functional as F


def predict():
    '''
    加载模型之前要初始化模型
    1、加载训练好的模型
    2、图片转换为对应的格式
    3、传输到模型中
    4、出来预测的结果
    :return:
    '''
    # model = getattr(models, opt.model)().eval()
    # device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    # model = t.load('save/1221Save.pth')
    # # model=model.to(device)
    # model.eval()
    # image_file_path = os.getcwd() + "/data/predict/"
    # file_path = os.listdir(image_file_path)
    # for file_path in file_path:
    #     img = Image.open('data/predict/'+file_path)
    #     transform = transforms.Compose(
    #         [transforms.CenterCrop(224),
    #          transforms.Resize(256),
    #          transforms.ToTensor(), ]
    #     )
    #     img = img.convert('RGB')
    #     img = transform(img)
    #     img =  img.unsqueeze(0)
    #     img = img.to(device)
    #     with t.no_grad():
    #         out =model(img)
    #         _, pre = t.max(out.data, 1)
    #         classIndex =pre.item()
    #         print(classIndex )



#------------------------------------------------
    model = getattr(models, opt.model)().eval()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    model.to(t.device('cuda') if opt.use_gpu else t.device('cpu'))
    image_file_path = os.getcwd() + "/data/pre/"
    file_path = os.listdir(image_file_path)



    for file_path in file_path:
        print(file_path.split('.png')[0])
        img = cv2.imread('data/pre/'+file_path, cv2.IMREAD_GRAYSCALE)            #图像转化为灰度
        plt.axis('off')
        plt.imshow(img)
        #图片预处理
        img = cv2.resize(img, (28, 28))
        img = t.from_numpy(img).float()
        img = img.view(1, 1, 28, 28)
        img = img.to(t.device('cuda') if opt.use_gpu else t.device('cpu'))
        # img = transform(img)
        # img = img.unsqueeze(0)
        # model.to(t.device('cuda') if opt.use_gpu else t.device('cpu'))
        out = model(img)
        _, pre = t.max(out.data, 1)
        print(file_path,"---->",pre.item())
        plt.title("predict:"+str(pre.item()))
        plt.savefig('predict/'+file_path.split('.png')[0]+".png")
        plt.show()

#----------------------------------------------------------
    # model = getattr(models, opt.model)().eval()
    # if opt.load_model_path:



    # pass


if __name__ == "__main__":
     # train()
     #test()
     predict()

# -*- coding: UTF-8 -*-
# environment requirement
# pandas == 1.5.3
# numpy == 1.21.2
# pytorch == 1.9.0
# scikit-learn == 1.2.0
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F

# define wave length;定义波形长度
wave = 8192
wavetoChn = int(wave/8)

# Initialize the weight of the model;初始化模型的权重
torch.manual_seed(10)


def set_random_seed(state=1):
    """
    设定随机种子
    :param state: 随机种子值
    :return: None
    """
    gens = (np.random.seed, torch.manual_seed)
    for set_state in gens:
        set_state(state)


def process_data(data):
    """
    处理加载的训练数据DataFrame
    :return: np array
    """
    res = []
    for i in range(data.shape[0]):
        x_res = data.iloc[i, 1].split(',')
        label = data.iloc[i, 2]
        x_res.append(label)
        res.append(x_res)
    return np.array(res, dtype=np.float64)


def train_loop(dataloader, model, loss_fn, optimizer):
    """
    模型训练部分
    :param dataloader: 训练数据集
    :param model: 训练用到的模型
    :param loss_fn: 评估用的损失函数
    :param optimizer: 优化器
    :return: None
    """
    for batch, x_y in enumerate(dataloader):
        X, y = x_y[:, :wave].type(torch.float64), torch.tensor(
            x_y[:, 8192], dtype=torch.long, device='cuda:1')
        # 开启梯度
        with torch.set_grad_enabled(True):
            # Compute prediction and loss
            pred = model(X.float())
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            # Backpropagation
            loss.backward()
            optimizer.step()


def valid_loop(dataloader, model, loss_fn):
    """
    模型测试部分
    :param dataloader: 测试数据集
    :param model: 测试模型
    :param loss_fn: 损失函数
    :return: None
    """
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0
    with torch.no_grad():   # 关掉梯度
        model.eval()
        for x_y in dataloader:
            X, y = x_y[:, :wave].type(torch.float64), torch.tensor(
                x_y[:, 8192], dtype=torch.long, device='cuda:1')
            Y = torch.zeros(size=(len(y), 6), device='cuda:1')
            for i in range(len(Y)):
                Y[i][y[i]] = 1

            pred = model(X.float())
            test_loss += loss_fn(pred, y).item()
            # 计算准确率
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(
        f"Test Results:\nAccuracy: {(100*correct):>0.3f} %\
        CroEtr loss: {test_loss:>8f}")


def prediction(dataloader, model, loss_fn):
    """
    模型测试部分
    :param dataloader: 测试数据集
    :param model: 测试模型
    :param loss_fn: 损失函数
    :return: None
    """
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0
    with torch.no_grad():   # 关掉梯度
        model.eval()
        for x_y in dataloader:
            X, y = x_y[:, :wave].type(torch.float64), torch.tensor(
                x_y[:, 8192], dtype=torch.long, device='cuda:1')
            # 注意Y和y的区别, Y用来计算L1 loss, y是CrossEntropy loss.
            Y = torch.zeros(size=(len(y), 6), device='cuda:1')
            for i in range(len(Y)):
                Y[i][y[i]] = 1

            pred = model(X.float())
            test_loss += loss_fn(pred, y).item()
            # 计算准确率
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(
        f"Test111 Results:\nAccuracy: {(100*correct):>0.3f}% \
            CroEtr loss: {test_loss:>8f}")


class Model(nn.Module):
    """
    模型
    """

    def __init__(self):
        """
        CNN模型构造
        """
        super(Model, self).__init__()
        self.conv_layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3,
                      padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        # 下采样down-sampling
        self.sampling_layer1 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32,
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.conv_layer2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3,
                      padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.sampling_layer2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128,
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.conv_layer3 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256,
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        self.sampling_layer3 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=512,
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.full_layer = nn.Sequential(
            nn.Linear(in_features=512*wavetoChn, out_features=6)
        )
        # 输出label预测概率
        self.pred_layer = nn.Softmax(dim=1)

    def forward(self, x):
        """
        前向传播
        """
        x = x.unsqueeze(
            dim=1)  # 升维. input shape(32, 205), output shape(32, 1, 205)
        x = self.conv_layer1(x)
        x = self.sampling_layer1(x)
        x = self.conv_layer2(x)
        x = self.sampling_layer2(x)
        x = self.conv_layer3(x)
        x = self.sampling_layer3(x)
        x = x.view(x.size(0), -1)   # output(32, 12800)
        x = self.full_layer(x)
        return self.pred_layer(x)


class AbsSumLoss(nn.Module):
    def __init__(self):
        super(AbsSumLoss, self).__init__()

    def forward(self, output, target):
        loss = F.l1_loss(target, output, reduction='sum')
        return loss


if __name__ == '__main__':
    set_random_seed(2023)   # 设定随机种子
    data = pd.read_csv(
        './dataset/train.csv')
    data = process_data(data)
    test_data = pd.read_csv(
        './dataset/test.csv')
    test_data = process_data(test_data)
    # 拆分训练测试集
    train, valid = train_test_split(data, test_size=0.2)

    # ini
    lr_rate = 1.5e-8
    w_decay = 1e-6
    n_epoch = 100
    b_size = 64
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    net = Model()
    net.to(device)
    optimizer = torch.optim.Adam(
        params=net.parameters(), lr=lr_rate, weight_decay=w_decay)
    loss_fn = nn.CrossEntropyLoss(reduction='sum')
    print(Model)
    test = test_data

    # train, test = traindata, testdata
    train, valid, test = torch.cuda.FloatTensor(
        train), torch.cuda.FloatTensor(valid), torch.cuda.FloatTensor(test)
    train = train.to(device)
    valid = valid.to(device)
    test = test.to(device)
    train_loader = torch.utils.data.DataLoader(
        dataset=train, batch_size=b_size)
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid, batch_size=b_size)
    test_loader = torch.utils.data.DataLoader(
        dataset=test, batch_size=b_size)

    for epoch in range(n_epoch):
        start = time.time()
        print(f"\n----------Epoch {epoch + 1}----------")
        train_loop(train_loader, net, loss_fn, optimizer)
        valid_loop(valid_loader, net, loss_fn)
        prediction(test_loader, net, loss_fn)
        end = time.time()

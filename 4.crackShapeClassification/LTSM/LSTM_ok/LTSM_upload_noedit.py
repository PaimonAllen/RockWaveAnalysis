# -*- coding: UTF-8 -*-
# environment requirement
# pandas == 1.5.3
# numpy == 1.21.2
# pytorch == 1.9.0
# scikit-learn == 1.2.0
import pandas as pd
import torch.nn as nn
import numpy as np
import torch
from sklearn.model_selection import KFold


torch.manual_seed(10)
# Rd = random.uniform(10, 20)

train = pd.read_csv('./dataset/train.csv')
testA = pd.read_csv('./dataset/test.csv')

# 读取数据 read data
train['signals'] = train['signals'].apply(
    lambda x: np.array(x.split(',')).astype('float32'))
train['label'] = train['label'].apply(lambda x: np.array(x).astype('int32'))
testA['signals'] = testA['signals'].apply(
    lambda x: np.array(x.split(',')).astype('float32'))
data = []
for val in train['signals'].values:
    data.append(val)
data = np.array(data)
targets = train['label'].values
# 转换成tensor
data = data.reshape(data.shape[0], 1, 8192)


test = []
for val in testA['signals'].values:
    test.append(val)
test = np.array(test)
test = test.reshape(test.shape[0], 1, 8192)
test = torch.from_numpy(test).to(torch.float32)  # 转换成tensor

test_ids = testA['id']
test_pre = np.zeros([len(test), 6])


class Model(nn.Module):
    '''
    input_size : 输入的特征维度
    hidden_size : 隐状态的特征维度，也就是LSTM输出的节点数目
    num_layers : 层数
    bias : 是否使用偏置，默认为True（使用）
    dropout : 剪枝
    bidirectional : 是否形成双向RNN，本模型设为True增加隐藏层数
    '''

    def __init__(self):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(input_size=8192, hidden_size=2048,
                            num_layers=3)  # 加了双向，输出的节点数翻2倍
        self.l1 = nn.Linear(2048, 1024)  # 特征输入
        self.l2 = nn.ReLU()  # 激活函数
        self.l3 = nn.BatchNorm1d(1024)  # 批标准化
        self.l4 = nn.Linear(1024, 512)
        self.l5 = nn.ReLU()
        self.l6 = nn.BatchNorm1d(512)
        self.l7 = nn.Linear(512, 6)  # 输出6个节点
        self.l8 = nn.BatchNorm1d(6)

    def forward(self, x):
        out, _ = self.lstm(x)
        # 选择最后一个时间点的output
        out = self.l1(out[:, -1, :])
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)
        out = self.l5(out)
        out = self.l6(out)
        out = self.l7(out)
        out = self.l8(out)
        return out


def SoftMax(x):
    return np.e**x/np.sum(np.e**x)


# 迭代次数
training_step = 10
# 每个批次的大小
batch_size = 64
n_splits = 5
random_state = 114
kf = KFold(n_splits=5, shuffle=True, random_state=114)
for fold, (train_idx, test_idx) in enumerate(kf.split(train, targets)):

    x_train, x_val = data[train_idx], data[test_idx]
    y_train, y_val = targets[train_idx], targets[test_idx]

    model = Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # 多分类的任务loss
    loss_func = nn.CrossEntropyLoss()

    model.train()
    ites = 1
    # ites 开始迭代
    for step in range(training_step):
        M_train = len(x_train)
        M_val = len(x_val)
        L_val = -batch_size
        for index in np.arange(0, M_train, batch_size):
            L = index
            R = min(M_train, index+batch_size)
            L_val += batch_size
            L_val %= M_val
            R_val = min(M_val, L_val + batch_size)
            train_pre = model(torch.from_numpy(x_train[L:R, :]).to(
                torch.float32))
            train_loss = loss_func(
                train_pre, torch.from_numpy(y_train[L:R]).to(torch.long))
            val_pre = model(torch.from_numpy(x_val[L_val:R_val, :]).to(
                torch.float32))
            val_loss = loss_func(val_pre, torch.from_numpy(
                y_val[L_val:R_val]).to(torch.long))
            train_acc = np.sum(
                np.argmax(np.array(train_pre.data),
                          axis=1) == y_train[L:R])/(R-L)
            val_acc = np.sum(
                np.argmax(np.array(val_pre.data),
                          axis=1) == y_val[L_val:R_val])/(R_val-L_val)
            if ites % 16 == 0:
                print("train_loss:", float(train_loss.data),
                      "train_acc:", train_acc,
                      "val_loss:", float(val_loss.data),
                      "val_acc:", val_acc,
                      "ites:", int(ites / 16))
            # BP
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            ites += 1
        val_pre = np.array(
            model(torch.from_numpy(x_val).to(torch.float32)).data)
        y_pre = []
        for val in val_pre:
            y_pre.append(SoftMax(val))
        y_pre = np.array(y_pre)

    pre = np.array(model(test).data)
    soft_pre = []
    for val in pre:
        soft_pre.append(SoftMax(val))
    test_pre += np.array(soft_pre)

    del model  # 更新模型

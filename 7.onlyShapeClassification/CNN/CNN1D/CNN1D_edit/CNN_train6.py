import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

wave = 8192
wavetoChn = int(wave/8)
print(wavetoChn)

# https://blog.csdn.net/WildCatFish/article/details/116228950


torch.manual_seed(10)  # 固定每次初始化模型的权重


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
        处理加载的训练数据DataFrame，去掉id，同时对signals以”，“进行拆分。
    :param data: DataFrame, shape(n, 3)
    :return: np array, shape(n, 206)
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
        # print(X)
        # print(y)
        # sys.exit()
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
    test_loss, correct, l1_loss = 0, 0, 0
    # 用来计算abs-sum. 等于PyTorch L1Loss-->
    # https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html#torch.nn.L1Loss
    l1loss_fn = AbsSumLoss()
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
            test_loss += loss_fn(pred, y).item()    # 这个是CrossEntropy loss
            l1_loss += l1loss_fn(pred, Y).item()    # 这个是abs-sum/L1 loss
            # 这个是计算准确率的, 取概率最大值的下标.
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            # print(pred)

    test_loss /= size   # 等于CrossEntropy的reduction='mean', 这里有些多此一举可删掉.
    correct /= size
    print(
        f"Test Results:\nAccuracy: {(100*correct):>0.1f}% abs-sum loss: {l1_loss:>8f} CroEtr loss: {test_loss:>8f}")


def prediction(dataloader, model, loss_fn):
    """
        模型测试部分
    :param dataloader: 测试数据集
    :param model: 测试模型
    :param loss_fn: 损失函数
    :return: None
    """
    size = len(dataloader.dataset)
    test_loss, correct, l1_loss = 0, 0, 0
    # 用来计算abs-sum. 等于PyTorch L1Loss-->
    # https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html#torch.nn.L1Loss
    l1loss_fn = AbsSumLoss()
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
            test_loss += loss_fn(pred, y).item()    # 这个是CrossEntropy loss
            l1_loss += l1loss_fn(pred, Y).item()    # 这个是abs-sum/L1 loss
            # 这个是计算准确率的, 取概率最大值的下标.
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size   # 等于CrossEntropy的reduction='mean', 这里有些多此一举可删掉.
    correct /= size
    print(
        f"Test111 Results:\nAccuracy: {(100*correct):>0.1f}% abs-sum loss: {l1_loss:>8f} CroEtr loss: {test_loss:>8f}")

# def prediction(net, loss):
#     """
#         对数据进行预测
#     :param net: 训练好的模型
#     :param loss: 模型的测试误差值, 不是损失函数. 可以去掉, 这里是用来给预测数据命名方便区分.
#     :return: None
#     """
#     with torch.no_grad():
#         net.eval()
#         pred_loader = torch.utils.data.DataLoader(dataset=pred_data)
#         res = []
#         for x in pred_loader:
#             x = torch.tensor(x, device='cuda:1', dtype=torch.float64)
#             output = net(x.float())
#             res.append(output.cpu().numpy().tolist())

#         res = [i[0] for i in res]
#         res_df = pd.DataFrame(
#             res, columns=['label_1', 'label_2', 'label_3', 'label_4', 'label_5', 'label_6'])
#         res_df.insert(0, 'id', value=range(100, 120))

#         res_df.to_csv('res-loss '+str(loss)+'.csv', index=False)


class Model(nn.Module):
    def __init__(self):
        """
            CNN模型构造
        """
        super(Model, self).__init__()
        self.conv_layer1 = nn.Sequential(
            # input shape(32, 1, 256) -> [batch_size, channel, features] 8192
            # 参考->https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html#torch.nn.Conv1d
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3,
                      padding=1),   # 卷积后(32, 16, 256) 8192
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        # 下采样down-sampling
        self.sampling_layer1 = nn.Sequential(
            # input shape(32, 16, 256) 8192
            nn.Conv1d(in_channels=16, out_channels=32,
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # size随便选的, 这里output应该是(32, 32, 128) 4096
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.conv_layer2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3,
                      padding=1),   # 输出(32, 64, 128) 4096
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.sampling_layer2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128,
                      kernel_size=3, padding=1),  # 输出(32, 128, 128)
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 输出(32, 64, 64)
        )

        self.conv_layer3 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256,
                      kernel_size=3, padding=1),  # 输出(32, 256, 64)
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        self.sampling_layer3 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=512,
                      kernel_size=3, padding=1),  # 输出(32, 512, 64)
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 输出(32, 512, 32)
        )
        # 全连接层
        # self.full_layer = nn.Sequential(
        #     nn.Linear(in_features=512*wavetoChn, out_features=256*wavetoChn),
        #     nn.ReLU(),
        #     nn.Linear(in_features=256*wavetoChn, out_features=128*wavetoChn),
        #     nn.ReLU(),
        #     nn.Linear(in_features=128*wavetoChn, out_features=64*wavetoChn),
        #     nn.ReLU(),
        #     nn.Linear(in_features=64*wavetoChn, out_features=6)
        # )

        self.full_layer = nn.Sequential(
            nn.Linear(in_features=512*wavetoChn, out_features=6)
        )
        # 这个是输出label预测概率, 不知道这写法对不对
        self.pred_layer = nn.Softmax(dim=1)

    def forward(self, x):
        """
            前向传播
        :param x: batch
        :return: training == Ture 返回的是全连接层输出， training == False 加上一个Softmax(), 返回各个label概率.
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

        # if self.training:
        #     # CrossEntropyLoss自带LogSoftmax, 训练的时候不用输出概率(我也不知道这个写法对不对, 我是试错出来的.)
        #     return x
        # else:
        #     return self.pred_layer(x)
        return self.pred_layer(x)


class AbsSumLoss(nn.Module):
    def __init__(self):
        """
            可以直接用PyTorch的nn.L1Loss, 这个我写的时候不知道。
        """
        super(AbsSumLoss, self).__init__()

    def forward(self, output, target):
        loss = F.l1_loss(target, output, reduction='sum')

        return loss


if __name__ == '__main__':
    set_random_seed(1996)   # 设定随机种子
    # 加载数据集
    # traindata = pd.read_csv(
    #     '/home/tzr/DataLinux-SSD/Dataset/7.onlyShapeClassification/CNN/dataset/train3-CNN/train3-All.csv')
    # traindata = process_data(traindata)
    # print(traindata.shape)
    # testdata = pd.read_csv(
    #     '/home/tzr/DataLinux-SSD/Dataset/7.onlyShapeClassification/CNN/dataset/train2-CNN/valid2.csv')
    # testdata = process_data(testdata)
    # train = traindata
    # test = testdata

    data = pd.read_csv(
        '/home/tzr/DataLinux-SSD/Dataset/7.onlyShapeClassification/CNN/dataset/train6-CNN/train6-All.csv')
    data = process_data(data)
    test_data = pd.read_csv(
        '/home/tzr/DataLinux-SSD/Dataset/7.onlyShapeClassification/CNN/dataset/train6-CNN/test6.csv')
    test_data = process_data(test_data)
    # pred_data = pd.read_csv('./dataset/testA.csv')
    # pred_data = get_pred_x(pred_data)
    # 拆分训练测试集
    train, valid = train_test_split(data, test_size=0.2)

    # 初始化模型
    lr_rate = 1e-5
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
        print('training time: ', end-start)

    # predict

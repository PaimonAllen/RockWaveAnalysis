# -*- coding: UTF-8 -*-
# environment requirement
# numpy == 1.23.5
# scikit-learn == 1.1.3
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as kNN


def vector(filename):
    """
    向量转化。
    """
    returnVect = np.zeros((1, 8192))
    fr = open(filename)
    for i in range(8192):
        lineStr = fr.readline().strip("\n")
        returnVect[0][i] = lineStr
    return returnVect


# train序列集合
trainSet = []
# test序列集合
testSet = []
# 结果序列集合
resSet = []

data_train = np.loadtxt(
    "./dataset/train.txt", delimiter=",")
data_test = np.loadtxt(
    "./dataset/test.txt", delimiter=",")

# data split;数据分割
x_train, y_train = np.split(data_train, (17,), axis=1)
y_train = np.array(y_train, dtype=int)
x_train = x_train[:, 0:17]

x_test, y_test = np.split(data_test, (17,), axis=1)
y_test = np.array(y_test, dtype=int)
x_test = x_test[:, 0:17]


def KNNclassify():
    """
    KNN主函数
    """
    # 构建kNN分类器
    neigh = kNN(n_neighbors=5, algorithm='ball_tree')
    # 拟合模型 
    neigh.fit(x_train, y_train)
    # 错误检测计数
    errorCount = 0.0
    # 解析类别
    m = len(y_test)
    print("len(y_test):", len(y_test))
    for i in range(0, m):
        classNumber = int(y_test[i])
        x_test_now = x_test[i]
        x_test_now = x_test_now.reshape(1, -1)
        print(x_test_now.shape)
        classifierResult = neigh.predict(x_test_now)
        print(classifierResult)
        print("分类返回结果为%s\t真实结果为%s" % (classifierResult, classNumber))
        if (classifierResult != classNumber):
            errorCount += 1.0
        print("目前错误个数：", errorCount)
        print("process:{:.2f}%".format((i/m)*100))
    print("总共错了%d个数据\n错误率为%f%%，正确率为%f%%" %
          (errorCount, errorCount/m * 100, (1 - errorCount/m) * 100))


if __name__ == '__main__':
    """
    main函数入口
    """
    KNNclassify()


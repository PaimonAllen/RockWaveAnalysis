# -*- coding: UTF-8 -*-
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
    "./train.txt", delimiter=",")
data_test = np.loadtxt(
    "./test.txt", delimiter=",")

# data split;数据分割
x_train, y_train = np.split(data_train,
                            (17,),
                            axis=1)
y_train = np.array(y_train, dtype=int)

x_train = x_train[:, 0:17]

x_test, y_test = np.split(data_test,
                          (17,),
                          axis=1)
y_test = np.array(y_test, dtype=int)
x_test = x_test[:, 0:17]


def KNNclassify():
    """
    KNN主函数
    """
    # 构建kNN分类器
    neigh = kNN(n_neighbors=5, algorithm='ball_tree')
    # 拟合模型, trainingMat为训练矩阵,Labels为对应的标签
    neigh.fit(x_train, y_train)
    # 错误检测计数
    errorCount = 0.0
    # 从文件中解析出测试集的类别并进行分类测试
    m = len(y_test)
    print("len(y_test):", len(y_test))
    for i in range(0, m):
        classNumber = int(y_test[i])
        x_test_now = x_test[i]
        x_test_now = x_test_now.reshape(1, -1)
        print(x_test_now.shape)
        classifierResult = neigh.predict(x_test_now)
        print(classifierResult)
        if classifierResult == 1:
            res = '1.1*5mm'
        if classifierResult == 2:
            res = '2.1*10mm'
        if classifierResult == 3:
            res = '3.1*15mm'
        if classifierResult == 4:
            res = '4.1*20mm'
        if classifierResult == 5:
            res = '5.2*5mm'
        if classifierResult == 6:
            res = '6.2*10mm'
        if classifierResult == 7:
            res = '7.4*5mm'
        if classifierResult == 8:
            res = '8.2*5mm'
        if classifierResult == 9:
            res = '9.3*5mm'
        if classifierResult == 10:
            res = '10.4*5mm'
        if classifierResult == 11:
            res = '11.5*5mm'
        if classifierResult == 12:
            res = '12.noCrack'
        if classNumber == 1:
            rockclass = '1.1*5mm'
        if classNumber == 2:
            rockclass = '2.1*10mm'
        if classNumber == 3:
            rockclass = '3.1*15mm'
        if classNumber == 4:
            rockclass = '4.1*20mm'
        if classNumber == 5:
            rockclass = '5.2*5mm'
        if classNumber == 6:
            rockclass = '6.2*10mm'
        if classNumber == 7:
            rockclass = '7.4*5mm'
        if classNumber == 8:
            rockclass = '8.2*5mm'
        if classNumber == 9:
            rockclass = '9.3*5mm'
        if classNumber == 10:
            rockclass = '10.4*5mm'
        if classNumber == 11:
            rockclass = '11.5*5mm'
        if classNumber == 12:
            rockclass = '12.noCrack'
        print("分类返回结果为%s\t真实结果为%s" % (res, rockclass))
        if (classifierResult != classNumber):
            errorCount += 1.0
        print("目前错误个数：", errorCount)
        print("process:{:.2f}%".format((i/m)*100))
    print("总共错了%d个数据\n错误率为%f%%，正确率为%f%%" %
          (errorCount, errorCount/m * 100, (1 - errorCount/m) * 100))


"""
main函数入口
"""
if __name__ == '__main__':
    KNNclassify()

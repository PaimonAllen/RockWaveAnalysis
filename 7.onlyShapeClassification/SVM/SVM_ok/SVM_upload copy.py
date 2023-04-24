# -*- coding: UTF-8 -*-
import numpy as np
# import operator
from sklearn.svm import SVC


def vector(filename):
    # 创建向量
    returnVect = np.zeros((1, 8192))
    # print(returnVect)
    # 打开文件
    fr = open(filename)
    # 按行读取
    for i in range(8192):
        # 读一行数据
        lineStr = fr.readline().strip("\n")
        # print(lineStr)
        # 每一行的前32个元素依次添加到returnVect中
        returnVect[0][i] = lineStr
    # 返回转换后的向量
    # print(returnVect)
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
print("data_train:", data_train.shape)
print("data_test:", data_test.shape)

# 数据分割
x_train, y_train = np.split(data_train,  # 要切分的数组
                            (17,),  # 沿轴切分的位置，第7列开始往后为y
                            axis=1)  # 1代表纵向分割，按列分割

x_train = x_train[:, 0:17]

x_test, y_test = np.split(data_test,  # 要切分的数组
                          (17,),  # 沿轴切分的位置，第7列开始往后为y
                          axis=1)  # 1代表纵向分割，按列分割

x_test = x_test[:, 0:17]


def SVMclassify():
    """
    SVM主函数
    """
    # 构建SVM分类器
    clf = SVC(C=200, kernel='rbf')
    clf.fit(x_train, y_train)
    # 错误检测计数
    errorCount = 0.0
    m = len(y_test)
    print("len(y_test):", len(y_test))
    for i in range(0, m):
        classNumber = int(y_test[i])
        x_test_now = x_test[i]
        x_test_now = x_test_now.reshape(1, -1)
        print(x_test_now.shape)
        classifierResult = clf.predict(x_test_now)
        print(classifierResult)
        if classifierResult == 1:
            res = 'circle'
        if classifierResult == 2:
            res = 'invertedRegularTriangle'
        if classifierResult == 3:
            res = 'noCrack'
        if classifierResult == 4:
            res = 'regularTriangle'
        if classifierResult == 5:
            res = 'rhombus'
            # 加入结果矩阵
            # testSet.append([fileNameStr.split('-')[0], '2'])
        if classifierResult == 6:
            res = 'square'
            # 加入结果矩阵
            # testSet.append([fileNameStr.split('-')[0], '2'])
        if classNumber == 1:
            rockclass = 'circle'
        if classNumber == 2:
            rockclass = 'invertedRegularTriangle'
        if classNumber == 3:
            rockclass = 'noCrack'
        if classNumber == 4:
            rockclass = 'regularTriangle'
        if classNumber == 5:
            rockclass = 'rhombus'
        if classNumber == 6:
            rockclass = 'square'
        print("分类返回结果为%s\t真实结果为%s" % (res, rockclass))
        if (classifierResult != classNumber):
            errorCount += 1.0
        print("目前错误个数：", errorCount)
        print("process:{:.2f}%".format((i/m)*100))
    print("总共错了%d个数据\n错误率为%f%%，正确率为%f%%" %
          (errorCount, errorCount/m * 100, (1 - errorCount/m) * 100))


"""
函数说明:main函数

Parameters:
    无
Returns:
    无

Modify:
    ---
"""
if __name__ == '__main__':
    SVMclassify()

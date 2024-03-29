# -*- coding: UTF-8 -*-
import numpy as np
# import operator
from os import listdir
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


def SignalClassTest():
    """
    分类
    Parameters:
            无
    Returns:
            无
    """
    # 测试集的Labels
    Labels = []
    # 返回trainingDigits目录下的文件名
    trainingFileList = listdir('../../KNN/dataset/train1/')
    # 返回文件夹下文件的个数
    m = len(trainingFileList)
    # 初始化训练的Mat矩阵,测试集
    trainingMat = np.zeros((m, 8192))
    # 从文件名中解析出训练集的类别
    for i in range(m):
        # 获得文件的名字
        fileNameStr = trainingFileList[i]
        # 获得分类的数字
        classNumber = int(fileNameStr.split('-')[1])
        # 将获得的类别添加到hwLabels中
        Labels.append(classNumber)
        # 将每一个文件的数据存储到trainingMat矩阵中
        trainingMat[i, :] = vector(
            '../../KNN/dataset/train1/%s' % (fileNameStr))
    clf = SVC(C=0.5, kernel='rbf')
    clf.fit(trainingMat, Labels)
    # 返回testDigits目录下的文件列表
    testFileList = listdir('../../KNN/dataset/test1/')
    # 错误检测计数
    errorCount = 0.0
    # 测试数据的数量
    mTest = len(testFileList)
    print(mTest)
    # 从文件中解析出测试集的类别并进行分类测试
    for i in range(mTest):
        # 获得文件的名字
        fileNameStr = testFileList[i]
        # 获得分类的数字
        classNumber = int(fileNameStr.split('-')[1])
        # 获得测试集的1x1024向量,用于训练
        vectorUnderTest = vector('../../KNN/dataset/test1/%s' % (fileNameStr))
        # 获得预测结果
        # classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels,\
        # 3)
        classifierResult = clf.predict(vectorUnderTest)

        # print("分类返回结果为%d\t真实结果为%d" % (classifierResult, classNumber))
        # if(classifierResult != classNumber):
        #     errorCount += 1.0

        # print res
        print(classifierResult)
        if classifierResult == 1:
            res = '1.1*5mm'
            # 加入结果矩阵
            # testSet.append([fileNameStr.split('-')[0], '1'])
        if classifierResult == 2:
            res = '2.1*10mm'
            # 加入结果矩阵
            # testSet.append([fileNameStr.split('-')[0], '2'])
        if classifierResult == 3:
            res = '3.1*15mm'
            # 加入结果矩阵
            # testSet.append([fileNameStr.split('-')[0], '2'])
        if classifierResult == 4:
            res = '4.1*20mm'
            # 加入结果矩阵
            # testSet.append([fileNameStr.split('-')[0], '2'])
        if classifierResult == 5:
            res = '5.2*5mm'
            # 加入结果矩阵
            # testSet.append([fileNameStr.split('-')[0], '2'])
        if classifierResult == 6:
            res = '6.2*10mm'
            # 加入结果矩阵
            # testSet.append([fileNameStr.split('-')[0], '2'])
        if classifierResult == 7:
            res = '7.4*5mm'
            # 加入结果矩阵
            # testSet.append([fileNameStr.split('-')[0], '1'])
        if classifierResult == 8:
            res = '8.2*5mm'
            # 加入结果矩阵
            # testSet.append([fileNameStr.split('-')[0], '2'])
        if classifierResult == 9:
            res = '9.3*5mm'
            # 加入结果矩阵
            # testSet.append([fileNameStr.split('-')[0], '2'])
        if classifierResult == 10:
            res = '10.4*5mm'
            # 加入结果矩阵
            # testSet.append([fileNameStr.split('-')[0], '2'])
        if classifierResult == 11:
            res = '11.5*5mm'
            # 加入结果矩阵
            # testSet.append([fileNameStr.split('-')[0], '2'])
        if classifierResult == 12:
            res = '12.noCrack'
            # 加入结果矩阵
            # testSet.append([fileNameStr.split('-')[0], '2'])
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
        print("process:{:.2f}%".format((i/mTest)*100))

    print("总共错了%d个数据\n错误率为%f%%，正确率为%f%%" %
          (errorCount, errorCount/mTest * 100, (1 - errorCount/mTest) * 100))


if __name__ == '__main__':
    SignalClassTest()

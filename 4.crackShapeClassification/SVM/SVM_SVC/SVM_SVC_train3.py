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
        # 每一行的元素依次添加到returnVect中
        returnVect[0][i] = lineStr
    # 返回转换后的向量
    # print(returnVect)
    return returnVect


"""
函数说明:分类测试

Parameters:
    无
Returns:
    无

Modify:
    ---
"""
# train序列集合
trainSet = []
# test序列集合
testSet = []
# 结果序列集合
resSet = []

data_train = np.loadtxt(
    "/home/tzr/DataLinux-SSD/Dataset/6.crackLengthAndShapeClassification/KNN/dataset/train4/train4.txt", delimiter=",")
data_test = np.loadtxt(
    "/home/tzr/DataLinux-SSD/Dataset/6.crackLengthAndShapeClassification/KNN/dataset/test4/test4.txt", delimiter=",")
print("data_train:", data_train.shape)
print("data_test:", data_test.shape)

#

# 数据分割
x_train, y_train = np.split(data_train,  # 要切分的数组
                            (14,),  # 沿轴切分的位置，第7列开始往后为y
                            axis=1)  # 1代表纵向分割，按列分割
y_train = np.array(y_train, dtype=int)

x_train = x_train[:, 0:14]

x_test, y_test = np.split(data_test,  # 要切分的数组
                          (14,),  # 沿轴切分的位置，第7列开始往后为y
                          axis=1)  # 1代表纵向分割，按列分割
y_test = np.array(y_test, dtype=int)
x_test = x_test[:, 0:14]

# x_train, x_test, y_train, y_test = model_selection.train_test_split(
#     x,  # 所要划分的样本特征集
#     y,  # 所要划分的样本结果
#     random_state=1,  # 随机数种子确保产生的随机数组相同
#     test_size=0.3)  # 测试样本占比
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# sys.exit()


def KNNclassify():
    # 测试集的Labels
    # Labels = data[:,6]
    # 返回trainingDigits目录下的文件名
    # trainingFileList = listdir('../dataset/train1-test/')

    # print(trainingFileList)
    # 返回文件夹下文件的个数
    # m = len(trainingFileList)
    # print(m)
    # 初始化训练的Mat矩阵,测试集
    # trainingMat = np.zeros((m, 8192))
    # 从文件名中解析出训练集的类别
    # for i in range(m):
    #     # 获得文件的名字
    #     fileNameStr = trainingFileList[i]
    #     # 获得分类的数字
    #     classNumber = int(fileNameStr.split('-')[1])
    #     # 将获得的类别添加到Labels中
    #     Labels.append(classNumber)
    #     # 加入train矩阵
    #     trainSet.append([fileNameStr.split('-')[0], fileNameStr.split('-')[1]])
    #     # 将每一个文件的数据存储到trainingMat矩阵中
    #     trainingMat[i, :] = vector(
    #         '../dataset/train1-test/%s' % (fileNameStr))
    #     # print(i)
    # pca = PCA(n_components=features)
    # trainingMatPAC = pca.fit_transform(trainingMat)
    # # trainingMatPAC = trainingMat
    # print(trainingMatPAC.shape)
    # print(trainingMat.shape)
    # print(trainSet)
    # 构建kNN分类器

    # 初始化testMat
    # testFileList = listdir('../dataset/test1-test/')
    # n = len(testFileList)
    # print(n)
    # 初始化训练的Mat矩阵,测试集
    # testMat = np.zeros((n, 8192))
    # # 从文件名中解析出训练集的类别
    # for i in range(n):
    #     # 获得文件的名字
    #     fileNameStr = testFileList[i]
    #     # 获得分类的数字
    #     classNumber = int(fileNameStr.split('-')[1])
    #     # 将获得的类别添加到Labels中
    #     # Labels.append(classNumber)
    #     # 加入train矩阵
    #     trainSet.append([fileNameStr.split('-')[0], fileNameStr.split('-')[1]])
    #     # 将每一个文件的数据存储到trainingMat矩阵中
    #     testMat[i, :] = vector(
    #         '../dataset/test1-test/%s' % (fileNameStr))
    #     # print(i)
    # pca = PCA(n_components=features)
    # testMatPAC = pca.fit_transform(trainingMat)
    # # testMatPAC = testMat
    # print(testMatPAC.shape)
    # print('Labels:', len(Labels))

    # 构建kNN分类器
    clf = SVC(C=0.5, kernel='rbf')
    clf.fit(x_train, y_train)
    # sys.exit()
    # 错误检测计数
    errorCount = 0.0
    # 测试数据的数量
    # mTest = len(testFileList)
    # print(mTest)
    # 从文件中解析出测试集的类别并进行分类测试
    m = len(y_test)
    print("len(y_test):", len(y_test))
    for i in range(0, m):
        # print(i)
        # # 获得文件的名字
        # fileNameStr = testFileList[i]
        # 获得分类的数字
        classNumber = int(y_test[i])
        # print(classNumber)
        # 获得测试集的1x1024向量,用于训练
        # print(testMatPAC[i])
        # vectorUnderTest = testMatPAC[i]
        # vectorUnderTest = vectorUnderTest.reshape(1, -1)
        # print("vectorUnderTest.shape:", vectorUnderTest.shape)

        # 获得预测结果
        # classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels,
        # 3)
        x_test_now = x_test[i]
        x_test_now = x_test_now.reshape(1, -1)
        print(x_test_now.shape)
        classifierResult = clf.predict(x_test_now)
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
    KNNclassify()

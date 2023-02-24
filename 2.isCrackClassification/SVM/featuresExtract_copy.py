# features extract
import numpy as np
from os import listdir
FileList = listdir('/home/tzr/DataLinux/Documents/GitHubSYNC/RockWaveAnalysis/2.isCrackClassification/KNN/dataset/train1/')
# 返回文件夹下文件的个数
m = len(FileList)
# print(FileList)
print(m)
for i in range(m):
    # 获得文件的名字
    fileNameStr = FileList[i]
    # 获得分类的数字
    classNumber = int(fileNameStr.split('-')[1])
    # if classNumber == 1:
    #     data = np.loadtxt(f"./dataset/rockshape/train/{fileNameStr}")
    #     # print(data)
    #     max_value = max(data)
    #     min_value = min(data)
    #     # print(min(data))
    #     # print(max_value)
    #     # print(min_value)
    #     saveArr = [max_value, min_value, 1]
    #     saveArr = np.array(saveArr, dtype=float).reshape(1, 3)
    #     print(saveArr)
    #     print(saveArr.ndim)
    #     # saveClassName = np.array([1], dtype=int)
    #     with open("./dataset/rockshape/features_extract/test1.csv", 'a+') \
    # as f:
    #         np.savetxt(f, saveArr, delimiter=',')
    #         # # np.savetxt(f,)
    #         # np.savetxt(f, saveClassName)

    #     # with open("./dataset/rockshape/features_extract/test1.csv", \
    # 'a+') as f:
    data = np.loadtxt(f"/home/tzr/DataLinux/Documents/GitHubSYNC/RockWaveAnalysis/2.isCrackClassification/KNN/dataset/train1/{fileNameStr}")
    # print(data)
    max_value = max(data)
    min_value = min(data)
    mean_value = np.mean(data)
    range_value = max_value-min_value
    # print(min(data))
    # print(max_value)
    # print(min_value)
    saveArr = [max_value, min_value, mean_value, range_value, classNumber]
    saveArr = np.array(saveArr, dtype=float).reshape(1, 5)
    print(saveArr)
    print(saveArr.ndim)
    # saveClassName = np.array([1], dtype=int)
    with open("/home/tzr/DataLinux/Documents/GitHubSYNC/RockWaveAnalysis/2.isCrackClassification/KNN/dataset/train7/train7.txt", 'a+') as f:
        np.savetxt(f, saveArr, delimiter=',')
        # # np.savetxt(f,)
        # np.savetxt(f, saveClassName)

    # with open("./dataset/rockshape/features_extract/test1.csv", 'a+') as f:
            
# features extract
import numpy as np
from os import listdir
FileList = listdir('/home/tzr/DataLinux/Documents/\
GitHubSYNC/RockWaveAnalysis/6.crackLengthAndShapeClassification/\
KNN/dataset/process/6.crackLengthAndShapeTotal/')
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
    #     with open("./dataset/rockshape/features_extract/test1.csv",\
    # 'a+') as f:
    #         np.savetxt(f, saveArr, delimiter=',')
    #         # # np.savetxt(f,)
    #         # np.savetxt(f, saveClassName)

    #     # with open("./dataset/rockshape/features_extract/test1.csv", \
    # 'a+') as f:
    data = np.loadtxt(f"/home/tzr/DataLinux/Documents/GitHubSYNC/\
RockWaveAnalysis/6.crackLengthAndShapeClassification/KNN/dataset/\
process/6.crackLengthAndShapeTotal/{fileNameStr}")
    # print(data)
    max_value = max(data)
    min_value = min(data)
    mean_value = np.mean(data)
    # 峰峰值
    range_value = max_value-min_value
    # 方差和标准差
    var_value = data.var()
    std_value = data.std()
    # 均方根
    rms_value = np.sqrt(pow(mean_value, 2) + pow(std_value, 2))
    # print(min(data))
    # print(max_value)
    # print(min_value)
    saveArr = [max_value, min_value, mean_value, range_value, std_value,
               rms_value, classNumber]
    saveArr = np.array(saveArr, dtype=float).reshape(1, 7)
    print(saveArr)
    print(saveArr.ndim)
    # saveClassName = np.array([1], dtype=int)
    with open("./dataset/train2/train2.txt", 'a+') as f:
        np.savetxt(f, saveArr, delimiter=',')
        # # np.savetxt(f,)
        # np.savetxt(f, saveClassName)

    # with open("./dataset/rockshape/features_extract/test1.csv", 'a+') as f:
            

# -*- coding: utf-8 -*-

from os import listdir
import os
import shutil
import sys
import numpy as np


# current_directory = os.path.dirname(os.path.abspath('__file__'))

# 使用绝对路径
PATH = "/home/tzr/DataLinux/AIDataset/RockWave/Process/ExtractAll/2.isCrack/"
print(PATH)
# sys.exit()


def rename_file(tagname, oldname, val, num):
    tag = tagname
    print('tag:', tag)
    oldname = os.path.join(PATH, oldname)
    test = np.loadtxt(oldname)
    print(test)
    print(test.shape)
    newname = "/home/tzr/DataLinux/Documents/GitHubSYNC/\
RockWaveAnalysis/2.isCrackClassification/\
KNN/dataset/process/2.isCrackTotalExtractAll/" + val + '-' + str(num)

    print("oldname -------------> newname")
    print(oldname)
    print(newname)
    shutil.copy(oldname, newname)


# Rename main function
listArr = listdir(PATH)
'''
# print(listArr)
splitArr = []
resArr = []
# print('-----------')
for i in listArr:
    splitArr.append(i.split("."))
    # i.split(".")

# print(splitArr)
# print('-----------')
splitArr.sort(key=lambda x: int(x[0]))
# pri   nt(splitArr)

# print('-----------')
for i in splitArr:
    resArr.append(".".join(i))
# print(resArr)

# sys.exit()
'''
num = 1
totalNum = 0
listArr.sort()
for i in listArr:
    print(i)
    for j in listdir(PATH + '/'+i):
        rename_file(i, i+'/'+j, j, num)
        # sys.exit()
        pass
    # print(num)
    num += 1


sys.exit()

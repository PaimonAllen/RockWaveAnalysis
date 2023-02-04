
# -*- coding: utf-8 -*-

from os import listdir
import os
import shutil
import sys
import numpy as np


# current_directory = os.path.dirname(os.path.abspath('__file__'))

# 基于当前文件所在路径，转化为绝对路径
PATH = os.path.abspath(os.path.join(
    os.getcwd(), "4.crackShape"))


def rename_file(tagname, oldname, val, num):
    tag = tagname
    print('tag:', tag)
    oldname = os.path.join(PATH, oldname)
    test = np.loadtxt(oldname)
    print(test)
    print(test.shape)
    newname = oldname.replace('/process/4.crackShape',
                              '/process').replace(tag,
                                                  '4.crackShapeTotal') \
        + '-' + str(num)
    # os.rename(oldname, newname)
    shutil.copy(oldname, newname)
    print("oldname -------------> newname")
    print(oldname)
    print(newname)


# Rename main function
num = 1
ListArr = listdir(PATH)
ListArr.sort()
print(ListArr)
for i in ListArr:
    # print(i)
    for j in listdir(PATH + '/'+i):
        rename_file(i, i+'/'+j, j, num)
        # sys.exit()
    num += 1


sys.exit()


def rename_file(oldname, val):
    oldname = os.path.join(PATH, oldname)
    # newname = oldname
    # print(oldname)
    # newname = f'{val}-2'
    # if "-3" in oldname:
    # print(val['a'])
    # a = val['a']
    # newname = oldname + '-1'
    # newname = os.path.join(PATH,newname)+'.txt'
    # newname = os.path.join(PATH, newname)
    # # rename
    # newname = oldname.replace("-3","-2")
    # os.rename(oldname, newname)
    # print("oldname -------------> newname")
    # print(oldname)
    # print(newname)
    # val['a'] = val['a'] + 1
    # shutil.move(oldname, newname)
    # if "-4" in oldname:
    # print(val['a'])
    # a = val['a']
    # newname = oldname + '-1'
    # newname = os.path.join(PATH,newname)+'.txt'
    # newname = os.path.join(PATH, newname)
    # # rename
    # newname = oldname.replace("-4","-1")
    # os.rename(oldname, newname)
    # print(oldname)
    # print(newname)
    # val['a'] = val['a'] + 1

    newname = oldname.replace("-1", "-2")
    val['a'] = val['a'] + 1
    if "1000" in oldname:
        print(oldname)
        print(newname)
    os.rename(oldname, newname)
    # print("oldname -------------> newname")
    # print(oldname)
    # print(newname)


if __name__ == "__main__":
    fileList = os.listdir(PATH)
    fileList.sort()
    # print(fileList)
    # i = 1
    val = {'a': 0}
    for name in fileList:
        # print(i)
        rename_file(name, val)
        # i += 1
    print(val["a"])

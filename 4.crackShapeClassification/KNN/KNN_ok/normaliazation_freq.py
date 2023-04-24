import numpy as np

data = np.loadtxt(
    "/home/tzr/DataLinux-SSD/Dataset/7.onlyShapeClassification/KNN/dataset/train12/test-fre.txt", delimiter=",")

np.set_printoptions(threshold=np.inf)

print(data.shape)

# data1 = data[:, 0:1]
# print(data1.shape)
# data2 = data[:, 0:1]


def normalization(data_nor):
    _range = np.max(data_nor) - np.min(data_nor)
    data_nor -= np.min(data_nor)
    data_nor = data_nor / _range
    # data_nor = 2*data_nor - 1
    return data_nor


# data_nor1 = normalization(data1)
# print(data_nor1)
# data3 = np.column_stack((data1, data2))
# print(data3)
resdata = normalization(data[:, 0])
for i in range(1, 4):
    # print(i)
    data1 = data[:, i]
    data1 = normalization(data1)
    resdata = np.column_stack((resdata, data1))

resdata = np.column_stack((resdata, data[:, 4]))
print(resdata.shape)

np.savetxt("/home/tzr/DataLinux-SSD/Dataset/7.onlyShapeClassification/KNN/dataset/train12/normaliazation/test-fre-nor.txt", resdata, delimiter=",")

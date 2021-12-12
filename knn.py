# -*- coding = utf-8 -*-
# @Time : 2021/12/12 15:13
# @Author : Luxlios
# @File : KNN.py
# @Software : PyCharm

import struct
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os

# KNN
class knn():
    def __init__(self, x, y, num_class):
        # train_feature:x
        # train_label:y
        self.x_train = x
        self.y_train = y
        self.num_class = num_class

    def test(self, x, y, k=11):
        # test_feature:x
        # test_label:y
        # num_neighbor:k
        num_test = x.shape[0]
        num_train = self.x_train.shape[0]
        distance = np.zeros([num_test, num_train])

        # 算法主体
        for i in range(num_test):
            for j in range(num_train):
                # find k neighbor
                # L2_distance
                distance[i, j] = np.linalg.norm(x[i, :] - self.x_train[j, :])
                # Cos_distance
                # distance[i, j] = 1-np.dot(x[i, :], self.x_train[j, :])/(np.linalg.norm(x[i, :]) * np.linalg.norm(self.x_train[j, :]))

        # get index of k neighbors
        distance = np.argsort(distance, axis=1)
        neighbors = np.zeros([num_test, k], dtype=int)
        for i in range(num_test):
            for j in range(k):
                neighbors[i, j] = distance[i, j]

        # get every class_vote
        classvote = np.zeros([num_test, self.num_class], dtype=int)
        for i in range(num_test):
            for j in range(k):
                classvote[i, self.y_train[neighbors[i, j]]] += 1
        # 多数表决规则，找到分类最多的那一类
        _class = np.argmax(classvote, axis=1)

        correct = 0
        for i in range(num_test):
            if _class[i] == y[i]:
                correct = correct + 1
        accuracy = correct / num_test

        # 返回预测class和准确度
        return _class, accuracy

# 标准化
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

# mnist数据集读取函数（IDX1-UBYTE文件）
def load_mnist(path, kind='train'):  # 设置kind方便我们之后打开测试集数据，扩展程序
    """Load MNIST data from path"""
    """os.path.join为合并括号里面的所有路径"""
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)

    images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)

    with open(labels_path, 'rb') as lbpath:
        # 'I'表示一个无符号整数，大小为四个字节
        # '>II'表示读取两个无符号整数，即8个字节
        # 将文件中指针定位到数据集开头处，file.read(8)就是把文件的读取指针放到第九个字节开头处
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    return images, labels

if __name__ == '__main__':
    # 读取mnist数据集
    x_train, y_train = load_mnist('./dataset', kind='train')
    x_train = x_train[0:5000]
    # x_train = normalization(x_train[0:5000])
    y_train = y_train[0:5000]
    x_test, y_test = load_mnist('./dataset', kind='t10k')
    x_test = x_test[0:100]
    # x_test = normalization(x_test[0:100])
    y_test = y_test[0:100]

    # flatten
    x_train = np.array(x_train.reshape(x_train.shape[0], 28*28*1))
    x_test = np.array(x_test.reshape(x_test.shape[0], 28*28*1))

    # mnist有10个class
    KNN = knn(x_train, y_train, num_class=10)
    accuracy_record = []
    for k in range(1, 20, 2):
        _class, _accuracy = KNN.test(x_test, y_test, k)
        accuracy_record.append(_accuracy)

    # figure
    plt.figure()
    plt.scatter(range(1, 20, 2), accuracy_record, marker='.', c='C1', s=150)
    plt.grid()
    plt.xlabel('K')
    plt.ylabel('error_rate')
    plt.title('L2_distance')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()









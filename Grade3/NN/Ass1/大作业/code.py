import numpy as np
import struct
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import torch


def load_mnist_train(path, kind='train'):
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    return images, labels


def load_mnist_test(path, kind='t10k'):
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    return images, labels


def load_data():
    path = ''
    train_images, train_labels = load_mnist_train(path)  # 60000*784 60000*1
    test_images, test_labels = load_mnist_test(path)  # 10000*784 10000*1
    return train_images, train_labels, test_images, test_labels


def to_onehot(y):
    # 输入为向量，转为onehot编码
    y = y.reshape(-1, 1)  # 即为n*1形式
    enc = OneHotEncoder()
    enc.fit(y)
    y = enc.transform(y).toarray()
    return y


def to_label(y):
    return np.argmax(y, axis=1)


def accuracy(y_true, y_pred):
    # 输入真实标记和预测值返回Accuracy，其中真实标记和预测值是维度相同的向量
    return np.mean(y_true == y_pred)


'''sigmoid为激活函数'''


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


'''定义类NNet，里面包括初始化，训练，预测三部分'''


class NeuralNetwork():
    def __init__(self, d, q, l, method):
        # 初始化权重和偏置，输入层d=784个神经元，隐层设q=100个神经元，输出层l=10个神经元
        # weights
        if method == 'random':
            self.w1 = np.random.rand(d, q) - 0.5  # 输入层到隐层
            self.w2 = np.random.rand(q, l) - 0.5  # 隐层到输出层
        elif method == 'zeros':
            self.w1 = np.zeros((d, q))  # 输入层到隐层
            self.w2 = np.zeros((q, l))  # 隐层到输出层

    def predict(self, X):
        q_out = sigmoid(np.dot(X, self.w1))
        return sigmoid(np.dot(q_out, self.w2))

    def train(self, X, y, learning_rate, epochs, func, learning_rate_method, regularzation):
        # X:60000*784,y:60000*10,d=784,q=100,l=10,W1:d*q,,W2；q*l
        acc_list = []  # 记录每轮准确度便于画图
        acc_list_1 = []  # 与baseline对比的
        count = 0
        for epoch in range(epochs):
            for i in range(X.shape[0]):
                q_in = np.dot(X[i], self.w1).reshape(1, -1)  # 1*784 * 784*q
                q_out = sigmoid(q_in).reshape(1, -1)  # 1*q
                y_in = np.dot(q_out, self.w2).reshape(1, -1)  # 1*q * q*10
                y_out = sigmoid(y_in).reshape(1, -1)
                if func == 'mse':
                    '''输出层误差'''
                    delta2 = (y[i] - y_out).reshape(1, -1)  # 1*l
                    '''隐层误差'''
                    delta1 = np.dot(self.w2, delta2.T).T  # 1*q
                    '''调整输出层权重'''
                    self.w2 = self.w2 + learning_rate * np.dot(q_out.T, delta2 * y_out * (1 - y_out))
                    '''调整隐层权重'''
                    self.w1 = self.w1 + learning_rate * np.dot(X[i].reshape(-1, 1), delta1 * q_out * (1 - q_out))

                elif func == 'entropy loss':
                    '''做softmax处理'''
                    exp_y_out = np.exp(y_out)
                    y_out = exp_y_out / np.sum(exp_y_out)

                    w2_loss_func = (y_out - y[i]).reshape(1, -1)  # 1*l
                    w2_grad = np.dot(q_out.T, w2_loss_func)  # q*1 * 1*l = q*l
                    w1_loss_diff = w2_loss_func  # 1*l
                    w1_loss_func = np.multiply(q_out, 1 - q_out) * np.dot(w1_loss_diff, self.w2.T)  # 1*q
                    w1_grad = np.dot(X[i].reshape(-1, 1), w1_loss_func)

                    self.w1 = self.w1 - learning_rate * w1_grad
                    self.w2 = self.w2 - learning_rate * w2_grad

            y_pred = self.predict(X)  # y_pred是概率
            y_pred_class = np.argmax(y_pred, axis=1)
            acc = accuracy(to_label(y), y_pred_class)
            acc_list.append(acc)
            print("Epoch %d accuracy:  %.3f%%" % (epoch, acc * 100))
            if learning_rate_method == 'decrease' and epoch % 10 == 0 and epoch > 0:
                learning_rate = learning_rate - 0.1
            if regularzation == "earlystop" and epoch > 0:
                if acc < acc_list[epoch - 1]:
                    count = count + 1
                    print(count)
                    if count >= 10:
                        break
        '''绘图代码'''
        # plt.plot(range(len(acc_list)), acc_list, marker='o', label="No")
        # plt.plot(range(len(acc_list_1)), acc_list_1, marker='*', label='Early stop')
        # plt.legend()
        # plt.xlabel('Epoch')
        # plt.ylabel('Accuracy')
        # plt.show()


if __name__ == '__main__':
    # X_train, y_train, X_test, y_test = load_data()
    # y_train = to_onehot(y_train)  # 60000*10
    # # 缩放输入数据。0.01的偏移量避免0值输入
    # X_train = (np.asfarray(X_train[0:]) / 255.0 * 0.99) + 0.01
    # # y_test = to_onehot(y_test)
    # n_features = X_train.shape[1]  # 就是784
    # n_hidden_layer_size = 60
    # n_outputs = 10  # 表示数字0-9
    # network = NeuralNetwork(d=n_features, q=n_hidden_layer_size, l=n_outputs, method='random')
    # network.train(X_train, y_train, learning_rate=0.00005, epochs=60, func='entropy loss', learning_rate_method='fixed',
    #               regularzation="earlystop")
    # network.train(X_train, y_train, learning_rate=0.7, epochs=60, func='mse', learning_rate_method='decrease',
    #               regularzation="no")
    # 预测结果
    # y_pred = network.predict(X_test)
    # y_pred_class = np.argmax(y_pred, axis=1)
    # acc = accuracy(y_test, y_pred_class) * 100
    # print("\nTesting Accuracy: {:.3f} %".format(acc))


from pathlib import Path

import numpy as np
from sklearn.svm import SVC, LinearSVC
import matplotlib.pyplot as plt
import math
from data import get_moon_data
from examples import RBF_kernel_fn, linear_kernel_fn, train_kernel_svm_classifier_with_gram
from utils import plot_decision_boundary, plot_data
import time
Path("./results").mkdir(parents=True, exist_ok=True)

data_name = "moon_data"
data_x, data_y = get_moon_data()


def demo_plot_data():
    """
    plot distribution of datasets
    """
    plot_data(data_x, data_y, title=data_name,
              path="./results/{}.jpg".format(data_name))


def demo_linear_svm():
    clf = train_kernel_svm_classifier_with_gram(
        data_x, data_y, kernel_fn=linear_kernel_fn)
    plot_decision_boundary(data_x, data_y, clf, kernel_fn=linear_kernel_fn, boundary=True,
                           title="linear kernel prediction",
                           path="./results/linear_{}.jpg".format(data_name))


def demo_rbf_svm():
    clf = train_kernel_svm_classifier_with_gram(
        data_x, data_y, kernel_fn=RBF_kernel_fn)
    plot_decision_boundary(data_x, data_y, clf, kernel_fn=RBF_kernel_fn, boundary=True,
                           title="rbf kernel prediction",
                           path="./results/rbf_{}.jpg".format(data_name))


def RFF_kernel_fn(x1, x2):
    """
    Approximate RBF kernel with random fourier features.
    Reference:
        Random Features for Large-Scale Kernel Machines
        https://papers.nips.cc/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html

    Input: 
        x1.shape=(x1_n, d)
        x2.shape=(x2_n, d)
    Return:
        gram.shape=(x1_n, x2_n)

    TODO: Complete this function.
    """
    D=500
    d=x1.shape[1]
    MSE=[]
    w=np.random.normal(size=(D,d))#D*d,x=x1*d
    b=2*math.pi*np.random.rand(D)
    z1=math.sqrt(2/D)*np.cos(np.dot(x1,w.T)+b)
    z2=math.sqrt(2/D)*np.cos(np.dot(x2,w.T)+b)
    '''
    #测试D和mse的关系
    for D in range(5000,100001,5000):
        w=np.random.normal(size=(D,d))#D*d,x=x1*d
        b=2*math.pi*np.random.rand(D)
        z1=math.sqrt(2/D)*np.cos(np.dot(x1,w.T)+b)
        z2=math.sqrt(2/D)*np.cos(np.dot(x2,w.T)+b)        
        gram_rbf = RBF_kernel_fn(x1, x2)
        gram_rff = np.dot(z1, z2.T)
        # diff = np.max(np.abs(gram_rbf - gram_rff))
        diff = np.mean((gram_rbf - gram_rff)**2)
        MSE.append(diff)
    plt.plot(range(5000,100001,5000),MSE)
    plt.xlabel("D")
    plt.ylabel("MSE")
    plt.show()
    '''
    return np.dot(z1, z2.T)
    



def test_RFF_kernel_fn():
    """
    TODO:
        1. investigate how the dimension of random fourier features affect the precision of approximation.
        2. investigate how x_dim affect the speed of rbf kernel and rff kernel.

    Reference:
        On the Error of Random Fourier Features, UAI 2015
        https://arxiv.org/abs/1506.02785
    """

    '''#测时间
    T_rff=[]
    T_rbf=[]
    for x_dim in range(100,901,100):
        x1 = np.random.randn(x_dim, 2)
        x2 = np.random.randn(x_dim, 2)
        t1=time.time()
        gram_rbf = RBF_kernel_fn(x1, x2)
        t2=time.time()
        t_rbf=(t2-t1)*1000#ms
        T_rbf.append(t_rbf)
        t3=time.time()
        gram_rff = RFF_kernel_fn(x1, x2)
        t4=time.time()
        t_rff=(t4-t3)*1000#ms
        T_rff.append(t_rff)
    plt.plot(range(100,901,100),T_rbf,label="RBF time")
    plt.plot(range(100,901,100),T_rff,label="RFF time")
    plt.xlabel("x_dim")
    plt.ylabel("Time(ms)")
    plt.legend()
    plt.show()
    '''

    '''
    #绘制MSE图像
    MSE=[]
    for i in range(0,50):
        x_dim = 100
        x1 = np.random.randn(x_dim, 2)
        x2 = np.random.randn(x_dim, 2)
        gram_rbf = RBF_kernel_fn(x1, x2)
        gram_rff = RFF_kernel_fn(x1, x2)
        # diff = np.max(np.abs(gram_rbf - gram_rff))
        diff = np.mean((gram_rbf - gram_rff)**2)
        MSE.append(diff)
    plt.plot(range(0,50),MSE,label="RFF")
    plt.legend()
    plt.xlabel("实验次数")
    plt.ylabel("MSE")
    plt.show()
    '''
    x_dim = 100
    x1 = np.random.randn(x_dim, 2)
    x2 = np.random.randn(x_dim, 2)
    gram_rbf = RBF_kernel_fn(x1, x2)
    gram_rff = RFF_kernel_fn(x1, x2)
    # diff = np.max(np.abs(gram_rbf - gram_rff))
    diff = np.mean((gram_rbf - gram_rff)**2)
    print("MSE of gram matrix: {:.10f}".format(diff))
    # D=100000, MSE ≈ 1e-5


def test_RFF_kernel_svm():
    """Test how your RFF perform.
    """
    clf = train_kernel_svm_classifier_with_gram(
        data_x, data_y, kernel_fn=RFF_kernel_fn)
    plot_decision_boundary(data_x, data_y, clf, kernel_fn=RFF_kernel_fn, boundary=True,
                           title="rff kernel prediction ",
                           path="./results/rff_{}(修改后).jpg".format(data_name))

if __name__ == "__main__":
    demo_plot_data()
    demo_linear_svm()
    demo_rbf_svm()
    test_RFF_kernel_fn()
    test_RFF_kernel_svm()
    
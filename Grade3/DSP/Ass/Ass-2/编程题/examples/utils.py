import numpy as np
import matplotlib.pyplot as plt


def plot_data(points, labels, title="", path=None):
    plt.scatter(points[:, 0], points[:, 1], c=labels)
    plt.title(title)
    if path is not None:
        plt.savefig(path)
    else:
        plt.show()
    plt.clf()
    plt.cla()


def plot_decision_boundary(data_x, data_y, clf, kernel_fn, boundary=False, title="", path=None):
    if boundary:
        x_min, x_max = data_x[:, 0].min() - 0.2, data_x[:, 0].max() + 0.2
        y_min, y_max = data_x[:, 1].min() - 0.2, data_x[:, 1].max() + 0.2
        xx1, xx2 = np.meshgrid(np.arange(x_min, x_max, 0.02),
                            np.arange(y_min, y_max, 0.02))
        data_xx = np.c_[xx1.ravel(), xx2.ravel()]
        gram = kernel_fn(data_xx, data_x)
        pred_yy = clf.predict(gram)
        plt.contourf(xx1, xx2, pred_yy.reshape(xx1.shape), alpha=0.4)

    gram_test = kernel_fn(data_x, data_x)
    pred_y = clf.predict(gram_test)
    plt.scatter(data_x[:, 0], data_x[:, 1], c=pred_y)
    plt.title(title)
    if path is not None:
        plt.savefig(path)
    else:
        plt.show()
    plt.clf()
    plt.cla()


def plot_decision_boundary_sklearn(data_x, data_y, clf, boundary=False, title="", path=None):
    if boundary:
        x_min, x_max = data_x[:, 0].min() - 0.2, data_x[:, 0].max() + 0.2
        y_min, y_max = data_x[:, 1].min() - 0.2, data_x[:, 1].max() + 0.2
        xx1, xx2 = np.meshgrid(np.arange(x_min, x_max, 0.02),
                            np.arange(y_min, y_max, 0.02))
        data_xx = np.c_[xx1.ravel(), xx2.ravel()]
        pred_yy = clf.predict(data_xx)
        plt.contourf(xx1, xx2, pred_yy.reshape(xx1.shape), alpha=0.4)
    pred_y = clf.predict(data_x)
    plt.scatter(data_x[:, 0], data_x[:, 1], c=pred_y)
    plt.title(title)
    if path is not None:
        plt.savefig(path)
    else:
        plt.show()
    plt.clf()
    plt.cla()

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
from sklearn.decomposition import PCA
import os
from matplotlib import pyplot as plt
import lightgbm as lgb
from load_data import load_images


def feature_get():
    feature_test = np.zeros((2000, 4))
    feature_train = np.zeros((2000, 4))
    label_train = np.zeros(2000)
    label_test = np.zeros(2000)
    count = 0
    # 提取三个特征，面积，横线数，竖线数
    # img = Image.open(os.path.join('figs_cut/test', "class7-26.jpg"))
    # mat = np.array(img)
    # print(mat, np.median(mat))
    for fname in os.listdir("figs_cut/test"):
        img = Image.open(os.path.join("figs_cut/test", fname))
        mat = np.array(img)
        S = 0  # 面积
        line1 = 0  # 横线数
        line2 = 0  # 竖线数
        if fname.startswith("class") and fname.endswith(".jpg"):
            label1 = int(fname.split("-")[0][-1])
        if label1 == 7 or label1 == 8:
            maincolor = np.min(mat)
        else:
            maincolor = np.median(mat)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if mat[i][j] > maincolor - 5 and mat[i][j] < maincolor + 5:
                    S = S + 1  #
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if mat[i][j] > maincolor - 5 and mat[i][j] < maincolor + 5 and j < mat.shape[1] - 1:
                    if mat[i][j + 1] > maincolor - 5 and mat[i][j + 1] < maincolor + 5:
                        line1 = line1 + 1
                        break

        for i in range(mat.shape[1]):
            for j in range(mat.shape[0]):
                if mat[j][i] > maincolor - 5 and mat[j][i] < maincolor + 5 and j < mat.shape[0] - 1:
                    if mat[j + 1][i] > maincolor - 5 and mat[j][i] < maincolor + 5:
                        line2 = line2 + 1
                        break

        feature_test[count, 0] = S
        feature_test[count, 1] = line1
        feature_test[count, 2] = line2
        feature_test[count, 3] = line1 + line2
        label_test[count] = label1
        count = count + 1
    count = 0
    for fname in os.listdir("figs_cut/train"):
        img = Image.open(os.path.join("figs_cut/train", fname))
        mat = np.array(img)
        S = 0  # 面积
        line1 = 0  # 横线数
        line2 = 0  # 竖线数
        if fname.startswith("class") and fname.endswith(".jpg"):
            label2 = int(fname.split("-")[0][-1])
        if label2 == 7 or label2 == 8:
            maincolor = np.min(mat)
        else:
            maincolor = np.median(mat)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if mat[i][j] > maincolor - 5 and mat[i][j] < maincolor + 5:
                    S = S + 1
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if mat[i][j] > maincolor - 5 and mat[i][j] < maincolor + 5 and j < mat.shape[1] - 1:
                    if mat[i][j + 1] > maincolor - 5 and mat[i][j + 1] < maincolor + 5:
                        line1 = line1 + 1
                        break

        for i in range(mat.shape[1]):
            for j in range(mat.shape[0]):
                if mat[j][i] > maincolor - 5 and mat[j][i] < maincolor + 5 and j < mat.shape[0] - 1:
                    if mat[j + 1][i] > maincolor - 5 and mat[j][i] < maincolor + 5:
                        line2 = line2 + 1
                        break

        feature_train[count, 0] = S
        feature_train[count, 1] = line1
        feature_train[count, 2] = line2
        feature_train[count, 3] = line1 + line2
        label_train[count] = label2
        count = count + 1
    return feature_train, label_train, feature_test, label_test


def middle_get():
    for fname in os.listdir("figs_processed/test"):
        img = Image.open(os.path.join("figs_processed/test", fname))
        mat = np.array(img)  # mat是2维64*64,灰度L=0.299R+0.587G+0.114B，255为白色
        L = 0
        R = 0
        U = 0
        D = 0
        # 取图像最下一行
        for i in range(0, 28):
            flag = 0
            for j in range(0, 28):
                if mat[i, j] < 230:
                    flag = 1
                    D = i
                    break
            if flag:
                break
        # 取最上一行
        for i in range(27, -1, -1):
            flag = 0
            for j in range(0, 28):
                if mat[i, j] < 230:
                    flag = 1
                    U = i
                    break
            if flag:
                break
        # 取最左一列
        for i in range(0, 28):
            flag = 0
            for j in range(0, 28):
                if mat[j, i] < 230:
                    flag = 1
                    L = i
                    break
            if flag:
                break
        # 取最右一列
        for i in range(27, -1, -1):
            flag = 0
            for j in range(0, 28):
                if mat[j, i] < 230:
                    flag = 1
                    R = i
                    break
            if flag:
                break
        # 裁剪
        if R < L or R == L or U < D or U == D:
            img1 = img
        else:
            img1 = img.crop((L, D, R, U))
        img1.save(os.path.join('figs_cut/test', fname))

    for fname in os.listdir("figs_processed/train"):
        img = Image.open(os.path.join("figs_processed/train", fname))
        mat = np.array(img)  # mat是2维64*64,灰度L=0.299R+0.587G+0.114B，255为白色
        L = 0
        R = 0
        U = 0
        D = 0
        # 取图像最下一行
        for i in range(0, 28):
            flag = 0
            for j in range(0, 28):
                if mat[i, j] < 230:
                    flag = 1
                    D = i
                    break
            if flag:
                break
        # 取最上一行
        for i in range(27, -1, -1):
            flag = 0
            for j in range(0, 28):
                if mat[i, j] < 230:
                    flag = 1
                    U = i
                    break
            if flag:
                break
        # 取最左一列
        for i in range(0, 28):
            flag = 0
            for j in range(0, 28):
                if mat[j, i] < 230:
                    flag = 1
                    L = i
                    break
            if flag:
                break
        # 取最右一列
        for i in range(27, -1, -1):
            flag = 0
            for j in range(0, 28):
                if mat[j, i] < 230:
                    flag = 1
                    R = i
                    break
            if flag:
                break
        # 裁剪
        if R < L or R == L or U < D or U == D:
            img1 = img
        else:
            img1 = img.crop((L, D, R, U))
        img1.save(os.path.join('figs_cut/train', fname))


def preprocess():
    # 变成灰度图
    # for fname in os.listdir("figs/test"):
    #     img = Image.open(os.path.join("figs/test", fname))
    #     img = img.convert('L')
    #     img.save(os.path.join('figs_processed/test', fname))
    # for fname in os.listdir("figs/train"):
    #     img = Image.open(os.path.join("figs/train", fname))
    #     img = img.convert('L')
    #     img.save(os.path.join('figs_processed/train', fname))
    # 先去除整体白边，只能run一次！
    # for fname in os.listdir("figs_processed/test"):
    #     img = Image.open(os.path.join("figs_processed/test", fname))
    #     img = img.crop((21, 16, 49, 44)) # 28*28
    #     img.save(os.path.join('figs_processed/test', fname))
    # for fname in os.listdir("figs_processed/train"):
    #     img = Image.open(os.path.join("figs_processed/train", fname))
    #     img = img.crop((21, 16, 49, 44))
    #     img.save(os.path.join('figs_processed/train', fname))
    middle_get()


def train(algo, pca_dim=None):
    # load data
    # train_xs, train_ys = load_images("train")
    # test_xs, test_ys = load_images("test")

    train_xs, train_ys, test_xs, test_ys = feature_get()
    train_xs = train_xs.reshape(train_xs.shape[0], -1)
    test_xs = test_xs.reshape(test_xs.shape[0], -1)
    n_tr = len(train_xs)

    # PCA
    # if pca_dim is not None:
    #     xs = np.concatenate([train_xs, test_xs], axis=0)
    #     pca = PCA(n_components=pca_dim)
    #     xs = pca.fit_transform(xs)
    #     train_xs, test_xs = xs[0:n_tr], xs[n_tr:]

    if algo == "LR":
        model = LogisticRegression(
            multi_class="multinomial", C=10.0,
            solver="lbfgs", max_iter=5000
        )
    elif algo == "DT":
        model = DecisionTreeClassifier(
            max_depth=10
        )
    elif algo == "RF":
        model = RandomForestClassifier(
            n_estimators=500
        )
    # elif algo == "LGB":
    #     model = lgb.LGBMClassifier(
    #         max_depth=10
    #     )

    model.fit(train_xs, train_ys)
    pred_train_ys = model.predict(train_xs)
    pred_test_ys = model.predict(test_xs)

    train_acc = np.mean(train_ys == pred_train_ys)
    test_acc = np.mean(test_ys == pred_test_ys)

    print("[{},{}] Train Acc:{:.5f}, Test Acc:{:.5f}".format(
        algo, pca_dim, train_acc, test_acc
    ))
    return train_acc, test_acc


if __name__ == "__main__":
    # preprocess()
    feature_get()
    train_accs = []
    test_accs = []
    for algo in ["LR", "DT", "RF"]:
        # for pca_dim in [None, 1000, 500, 200, 50]:
        #     train_acc, test_acc = train(algo, pca_dim)
            train_acc, test_acc = train(algo)
            train_accs.append(train_acc)
            test_accs.append(test_acc)

    train_accs = np.array(train_accs).reshape(3, 1)
    test_accs = np.array(test_accs).reshape(3, 1)

    fig = plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    ax = plt.gca()
    plt.imshow(train_accs)

    for i in range(1):
        for j in range(3):
            ax.text(i, j, "{:.1f}".format(train_accs[j][i] * 100.0))

    # ax.set_xticks(range(5))
    # ax.set_xticklabels(["None", "1000", "500", "200", "50"])
    ax.set_yticks(range(3))
    ax.set_yticklabels(["LR", "DT", "RF"])
    ax.set_title("Train Acc")

    plt.subplot(1, 2, 2)
    ax = plt.gca()
    plt.imshow(test_accs)

    for i in range(1):
        for j in range(3):
            ax.text(i, j, "{:.1f}".format(test_accs[j][i] * 100.0))

    # ax.set_xticks(range(5))
    # ax.set_xticklabels(["None", "1000", "500", "200", "50"])
    ax.set_yticks(range(3))
    ax.set_yticklabels(["LR", "DT", "RF"])
    ax.set_title("Test Acc")

    fig.tight_layout()
    fig.savefig("./feature-get.jpg")
    plt.show()

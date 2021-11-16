import numpy as np
seed = 1
np.random.seed(seed)
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def sigmoid(x):
    # （需要填写的地方，输入x返回sigmoid(x)，x可以是标量、向量或矩阵）
    return 1/(1+np.exp(-x))

def deriv_sigmoid(x):
    # （需要填写的地方，输入x返回sigmoid(x)在x点的梯度，x可以是标量、向量或矩阵）
    return sigmoid(x)*(1-sigmoid(x))

def mse_loss(y_true, y_pred):
    # （需要填写的地方，输入真实标记和预测值返回他们的MSE（均方误差，不需要除以2）,其中真实标记和预测值维度都是(n_samples,) 或 (n_samples, n_outputs)）
    return ((y_true.flatten() - y_pred.flatten()) ** 2).mean()

def accuracy(y_true, y_pred):
    # （需要填写的地方，输入真实标记和预测值返回Accuracy，其中真实标记和预测值是维度相同的向量）
    return np.sum(y_true==y_pred)/len(y_true)

def to_onehot(y):
    # 输入为向量，转为onehot编码
    y = y.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(y)
    y = enc.transform(y).toarray()
    return y

class NeuralNetwork():
    def __init__(self, d, q, l):
        # weights
        self.v = np.random.randn(d, q)
        self.w = np.random.randn(q, l)
        # biases
        self.gamma = np.random.randn(q)
        self.theta = np.random.randn(l)
        # 以上为神经网络中的权重和偏置，其中具体含义见西瓜书P101

    def predict(self, X):
        '''
        X: shape (n_samples, d)
        returns: shape (n_samples, l)
        '''
        # （需要填写的地方，输入样本，输出神经网络最后一层的输出值）
        self.alpha = np.matmul(X, self.v) # (n, q)
        self.b = sigmoid(self.alpha - self.gamma) # (n, q)
        self.beta = np.matmul(self.b, self.w) # (n, l)
        self.y_pred = sigmoid(self.beta - self.theta) # (n, l)
        return self.y_pred
    
    def train(self, X, y, learning_rate = 1, epochs = 500):
        '''
        X: shape (n_samples, d)
        y: shape (n_samples, l)
        输入样本和训练标记，进行网络训练
        '''
        for epoch in range(epochs):
            # （以下部分为向前传播过程，请完成）
            self.predict(X) 
            # （以下部分为计算梯度，请完成）
            d_L_d_ypred = (self.y_pred-y) # (n, l)
            # 输出层梯度
            g = d_L_d_ypred*self.y_pred*(1-self.y_pred) # (n, l)
            d_ypred_d_w = np.matmul(self.b.T,g) # (n, q, l)
            d_ypred_d_theta = -g 
            # 隐层梯度
            e = self.b*(1-self.b)*np.matmul(g, self.w.T) # (n, q)
            d_L_d_v = np.matmul(X.T,e) # (n, d, q)
            d_L_d_gamma = -e 
            # 更新权重和偏置
            self.w -= learning_rate*(d_ypred_d_w/len(X))
            self.theta -= learning_rate*np.average(d_ypred_d_theta, axis = 0) 
            self.v -= learning_rate*(d_L_d_v/len(X))
            self.gamma -= learning_rate*np.average(d_L_d_gamma, axis = 0) 

            # 计算epoch的loss
            if epoch % 10 == 0:
                y_preds = self.predict(X)
                loss = mse_loss(y, y_preds)
                print("Epoch %d loss: %.3f"%(epoch, loss))
    
if __name__ ==  '__main__':
    # 获取数据集，训练集处理成one-hot编码
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
    y_train = to_onehot(y_train)

    # 训练网络
    n_features = X.shape[1]
    n_hidden_layer_size = 60
    n_outputs = len(np.unique(y))
    network = NeuralNetwork(d = n_features, q = n_hidden_layer_size, l = n_outputs)
    # （可以对下面的learning_rate和epochs进行修改，以取得更好性能）
    network.train(X_train, y_train, learning_rate = 0.5, epochs = 10000)

    # 预测结果
    y_pred = network.predict(X_test)
    y_pred_class = np.argmax(y_pred, axis=1)
    mse = mse_loss(to_onehot(y_test), y_pred)
    print("\nTesting MSE: {:.3f}".format(mse))
    acc = accuracy(y_test, y_pred_class) * 100
    print("\nTesting Accuracy: {:.3f} %".format(acc))
    
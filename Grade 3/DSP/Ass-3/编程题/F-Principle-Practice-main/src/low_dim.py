import random

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import rfft
from torch.optim import lr_scheduler


class MLP(nn.Module):

    def __init__(self, input_size, output_size, hidden_sizes):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        for i, hidden_size in enumerate(hidden_sizes):
            if i == 0:
                self.layers.append(nn.Linear(input_size, hidden_size))
            else:
                self.layers.append(
                    nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            # self.layers.append(nn.ReLU())
            self.layers.append(nn.Tanh())
            self.layers.append(nn.BatchNorm1d(hidden_sizes[i]))
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))

    def forward(self, x, _=None):
        for layer in self.layers[:-1]:
            x = layer(x)
        out = self.layers[-1](x)
        return out


def low_dim_data(num=100, period=5):
    """x in range[-10, 10]
    """

    def fn(x):
        res = np.zeros_like(x)
        sin_value = np.sin(x)
        cut = 0.6
        mask1 = sin_value > cut
        maskn1 = sin_value < -cut
        res[mask1] = 1
        res[maskn1] = -1
        return res

    x = np.linspace(-10, 10, num=num)
    _x = x / period * np.pi * 2
    y = fn(_x)
    # plt.plot(_x,y)
    # plt.show()
    x = np.reshape(x, (-1, 1))
    y = np.reshape(y, (-1, 1))

    return x, y


def main(_=None):
    x, y = low_dim_data(num=100)
    model = MLP(
        input_size=1,
        output_size=1,
        hidden_sizes=[16, 128, 16]
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheculer = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                           factor=0.5,
                                                           patience=50,
                                                           verbose=True)
    x = torch.tensor(x, dtype=torch.float32)
    # print(x)
    # print(torch.fft.rfft(x))
    y = torch.tensor(y, dtype=torch.float32)
    # print(y)
    # print(torch.fft.fft(y))
    import matplotlib.pyplot as plt
    w = np.linspace(0, 20, num=100)
    _w = w / 5 * np.pi * 2

    freq = np.fft.rfftfreq(_w.shape[-1], d=1 / 240)
    y_value = y.detach().numpy().flatten()
    y_value_w = np.fft.rfft(y_value)

    for ep in range(15000):
        _y = model(x)
        loss = torch.mean((y - _y) ** 2)

        # y即为待拟合的函数值
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pred_y = _y.detach().numpy()
        if ep == 250 or ep == 500 or ep == 1000 or ep == 1500 or ep == 2000 or ep == 5000 or ep == 10000:
            y1=pred_y.flatten()
            pred_y_w=np.fft.rfft(y1)
            plt.plot(freq,abs(pred_y_w),color='b',linestyle='--') # 拟合曲线
            plt.plot(freq,abs(y_value_w),color='r') # 待拟合的
            plt.show()
        if ep % 500 == 0:
            print(loss)

        '''spacial domain'''
        # if ep == 250 or ep == 500 or ep == 1000 or ep == 1500 or ep == 2000:
        #     plt.plot(x, y, color='r')
        #     plt.plot(x, pred_y, color='b', linestyle='--')
        #     plt.show()
        # if ep % 2500 == 0:
        #     print(ep)
        #     plt.plot(x, y, color='r')
        #     plt.plot(x, pred_y, color='b', linestyle='--')
        #     plt.show()

    #############################################
    #############################################

    """
    Plot spacial domain and Fourier domain.
    """

    #############################################
    #############################################


if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    main()
